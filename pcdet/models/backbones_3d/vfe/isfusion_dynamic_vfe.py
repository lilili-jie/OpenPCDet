"""Native OpenPCDet implementation of ISFusion's DynamicVFEV2.

The released ISFusion model uses dynamic voxelization followed by two
point-wise VFE layers.  This module keeps the original parameter layout
(``vfe_layers.{i}.linear/norm``), but uses only PyTorch, torch_scatter and
OpenPCDet batch dictionaries.  It deliberately has no MMCV/MMDetection3D
dependency.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torch_scatter import scatter_max, scatter_mean

from .vfe_template import VFETemplate


class _DifferentiableAllReduce(Function):
    """Reference IS-Fusion all-reduce with a distributed backward pass."""

    @staticmethod
    def forward(ctx, value):
        gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
        # Keep the original implementation's all-gather + sum behavior.  It
        # avoids relying on an in-place all-reduce inside autograd.
        dist.all_gather(gathered, value, async_op=False)
        return torch.stack(gathered, dim=0).sum(dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    """IS-Fusion's equal-worker-weight synchronized BatchNorm1d.

    Each worker first computes its own channel mean and mean-square, then the
    worker statistics are averaged with equal weight.  This intentionally
    matches ``mmdet3d.ops.NaiveSyncBatchNorm1d`` used by the reference recipe;
    it is not interchangeable with ``torch.nn.SyncBatchNorm``, which follows
    different behavior when workers contain different numbers of points.
    """

    def forward(self, inputs):
        using_dist = dist.is_available() and dist.is_initialized()
        if not using_dist or dist.get_world_size() == 1 or not self.training:
            return super().forward(inputs)
        if inputs.shape[0] == 0:
            raise ValueError("NaiveSyncBatchNorm1d does not support empty inputs")

        # The reference force_fp32 decorator computes statistics in FP32 and
        # casts the result back under mixed precision.  Formal ISFusion runs
        # use FP32, but preserving this behavior makes the module robust.
        output_dtype = inputs.dtype
        values = inputs.float()
        is_two_dimensional = values.dim() == 2
        if is_two_dimensional:
            values = values.unsqueeze(2)

        channels = values.shape[1]
        mean = values.mean(dim=(0, 2))
        mean_square = (values * values).mean(dim=(0, 2))
        statistics = torch.cat((mean, mean_square), dim=0)
        statistics = (
            _DifferentiableAllReduce.apply(statistics) / dist.get_world_size()
        )
        mean, mean_square = torch.split(statistics, channels)
        variance = mean_square - mean * mean

        self.running_mean.add_(
            self.momentum * (mean.detach() - self.running_mean)
        )
        self.running_var.add_(
            self.momentum * (variance.detach() - self.running_var)
        )
        inverse_std = torch.rsqrt(variance + self.eps)
        scale = (self.weight * inverse_std).reshape(1, -1, 1)
        bias = (self.bias - mean * self.weight * inverse_std).reshape(1, -1, 1)
        output = values * scale + bias
        if is_two_dimensional:
            output = output.squeeze(2)
        return output.to(dtype=output_dtype)


class DynamicVFELayer(nn.Module):
    """Point-wise Linear -> BatchNorm1d -> ReLU used by DynamicVFEV2."""

    def __init__(
        self,
        in_channels,
        out_channels,
        eps=1e-3,
        momentum=0.01,
        norm_type="batch_norm",
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        norm_classes = {
            "batch_norm": nn.BatchNorm1d,
            "naive_sync_bn": NaiveSyncBatchNorm1d,
        }
        if norm_type not in norm_classes:
            raise ValueError(
                f"Unsupported DynamicVFE NORM_TYPE={norm_type!r}; "
                f"expected one of {sorted(norm_classes)}"
            )
        self.norm = norm_classes[norm_type](
            out_channels, eps=eps, momentum=momentum
        )

    def forward(self, inputs):
        return F.relu(self.norm(self.linear(inputs)), inplace=False)


class ISFusionDynamicVFE(VFETemplate):
    """DynamicVFEV2 adapted to OpenPCDet's ``batch_dict`` contract.

    Input points use ``[batch, x, y, z, intensity, timestamp]``.  Output
    coordinates use OpenPCDet/spconv order ``[batch, z, y, x]``.
    """

    def __init__(
        self,
        model_cfg,
        num_point_features,
        voxel_size,
        grid_size,
        point_cloud_range,
        **kwargs,
    ):
        super().__init__(model_cfg=model_cfg)

        self.with_cluster_center = model_cfg.get("WITH_CLUSTER_CENTER", True)
        self.with_voxel_center = model_cfg.get("WITH_VOXEL_CENTER", True)
        self.with_distance = model_cfg.get("WITH_DISTANCE", False)
        self.mode = model_cfg.get("MODE", "max")
        if self.mode not in ("max", "avg"):
            raise ValueError(f"Unsupported DynamicVFE reduction: {self.mode}")

        input_channels = int(num_point_features)
        if self.with_cluster_center:
            input_channels += 3
        if self.with_voxel_center:
            input_channels += 3
        # Preserve the released DynamicVFEV2 implementation.  Its constructor
        # reserves three channels when enabled, although the production CCF
        # config keeps WITH_DISTANCE=False.
        if self.with_distance:
            input_channels += 3

        filters = [input_channels] + list(model_cfg.NUM_FILTERS)
        if len(filters) < 2:
            raise ValueError("NUM_FILTERS must contain at least one output size")
        eps = float(model_cfg.get("NORM_EPS", 1e-3))
        momentum = float(model_cfg.get("NORM_MOMENTUM", 0.01))
        self.norm_type = str(
            model_cfg.get("NORM_TYPE", "batch_norm")
        ).lower()
        layers = []
        for index, (in_channels, out_channels) in enumerate(
            zip(filters[:-1], filters[1:])
        ):
            if index > 0:
                in_channels *= 2
            layers.append(
                DynamicVFELayer(
                    in_channels,
                    out_channels,
                    eps,
                    momentum,
                    norm_type=self.norm_type,
                )
            )
        self.vfe_layers = nn.ModuleList(layers)
        self.num_filters = list(model_cfg.NUM_FILTERS)

        self.register_buffer(
            "voxel_size_tensor",
            torch.as_tensor(voxel_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "point_cloud_range_tensor",
            torch.as_tensor(point_cloud_range, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "grid_size_tensor",
            torch.as_tensor(grid_size, dtype=torch.long),
            persistent=False,
        )

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def _voxel_coords_from_linear(linear_coords, scale_xyz, scale_yz, scale_z):
        batch = torch.div(linear_coords, scale_xyz, rounding_mode="floor")
        remainder = linear_coords % scale_xyz
        x = torch.div(remainder, scale_yz, rounding_mode="floor")
        remainder = remainder % scale_yz
        y = torch.div(remainder, scale_z, rounding_mode="floor")
        z = remainder % scale_z
        return torch.stack((batch, z, y, x), dim=1).int()

    def _reduce(self, features, inverse, voxel_count):
        if self.mode == "avg":
            return scatter_mean(features, inverse, dim=0, dim_size=voxel_count)
        return scatter_max(features, inverse, dim=0, dim_size=voxel_count)[0]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict["points"]
        if points.ndim != 2 or points.shape[1] < 1 + 3:
            raise ValueError(f"Expected [N, 1+C] points, got {tuple(points.shape)}")

        xyz = points[:, 1:4]
        point_features = points[:, 1:]
        voxel_size = self.voxel_size_tensor.to(device=points.device, dtype=xyz.dtype)
        point_range = self.point_cloud_range_tensor.to(
            device=points.device, dtype=xyz.dtype
        )
        grid_size = self.grid_size_tensor.to(device=points.device)

        xyz_coords = torch.floor((xyz - point_range[:3]) / voxel_size).long()
        valid = ((xyz_coords >= 0) & (xyz_coords < grid_size[:3])).all(dim=1)
        if not bool(valid.all()):
            points = points[valid]
            xyz = xyz[valid]
            point_features = point_features[valid]
            xyz_coords = xyz_coords[valid]
        if points.shape[0] == 0:
            raise RuntimeError("ISFusionDynamicVFE received no in-range points")

        scale_z = int(grid_size[2].item())
        scale_yz = int((grid_size[1] * grid_size[2]).item())
        scale_xyz = int((grid_size[0] * grid_size[1] * grid_size[2]).item())
        linear_coords = (
            points[:, 0].long() * scale_xyz
            + xyz_coords[:, 0] * scale_yz
            + xyz_coords[:, 1] * scale_z
            + xyz_coords[:, 2]
        )
        unique_coords, inverse = torch.unique(
            linear_coords, sorted=True, return_inverse=True
        )
        voxel_count = unique_coords.shape[0]

        decorated = [point_features]
        if self.with_cluster_center:
            points_mean = scatter_mean(xyz, inverse, dim=0, dim_size=voxel_count)
            decorated.append(xyz - points_mean[inverse])
        if self.with_voxel_center:
            centers = (
                xyz_coords.to(dtype=xyz.dtype) * voxel_size
                + voxel_size / 2
                + point_range[:3]
            )
            decorated.append(xyz - centers)
        if self.with_distance:
            # Kept for configuration validation.  The released model disables
            # this branch; using it would require changing the original
            # three-channel constructor convention.
            distance = torch.norm(xyz, p=2, dim=1, keepdim=True)
            decorated.append(distance.expand(-1, 3))

        features = torch.cat(decorated, dim=-1)
        voxel_features = None
        for index, layer in enumerate(self.vfe_layers):
            pointwise = layer(features)
            voxel_features = self._reduce(pointwise, inverse, voxel_count)
            if index != len(self.vfe_layers) - 1:
                features = torch.cat((pointwise, voxel_features[inverse]), dim=1)

        voxel_coords = self._voxel_coords_from_linear(
            unique_coords, scale_xyz, scale_yz, scale_z
        )
        batch_dict["voxel_features"] = voxel_features
        batch_dict["pillar_features"] = voxel_features
        batch_dict["voxel_coords"] = voxel_coords
        return batch_dict
