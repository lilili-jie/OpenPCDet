"""Native spconv implementation of ISFusion's SparseEncoderV2."""

from __future__ import annotations

from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


def sparse_post_act_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    indice_key=None,
    conv_type="subm",
    norm_fn=None,
):
    if conv_type == "subm":
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False, indice_key=indice_key,
        )
    elif conv_type == "spconv":
        conv = spconv.SparseConv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False, indice_key=indice_key,
        )
    else:
        raise ValueError(f"Unsupported sparse convolution type: {conv_type}")
    return spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())


class ISFusionSparseBasicBlock(spconv.SparseModule):
    """Residual SubMConv3d block with CCF-compatible parameter names."""

    def __init__(self, channels, norm_fn):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = norm_fn(channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = norm_fn(channels)

    def forward(self, inputs):
        identity = inputs
        output = self.conv1(inputs)
        output = replace_feature(output, self.relu(self.bn1(output.features)))
        output = self.conv2(output)
        output = replace_feature(output, self.bn2(output.features))
        output = replace_feature(output, output.features + identity.features)
        return replace_feature(output, self.relu(output.features))


class ISFusionSparseEncoder(nn.Module):
    """SparseEncoderV2 with an OpenPCDet ``batch_dict`` interface."""

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        sparse_shape = model_cfg.get("SPARSE_SHAPE", None)
        self.sparse_shape = (
            list(sparse_shape)
            if sparse_shape is not None
            else [int(grid_size[2]) + 1, int(grid_size[1]), int(grid_size[0])]
        )
        eps = float(model_cfg.get("NORM_EPS", 1e-3))
        momentum = float(model_cfg.get("NORM_MOMENTUM", 0.01))
        norm_fn = partial(nn.BatchNorm1d, eps=eps, momentum=momentum)

        self.conv_input = sparse_post_act_block(
            input_channels, 32, 3, padding=1,
            indice_key="subm1", conv_type="subm", norm_fn=norm_fn,
        )

        stage_specs = (
            (32, 64, 1),
            (64, 128, 1),
            (128, 256, (0, 1, 1)),
            (256, None, None),
        )
        self.encoder_layers = spconv.SparseSequential()
        for stage_index, (channels, next_channels, padding) in enumerate(
            stage_specs, start=1
        ):
            blocks = [
                ISFusionSparseBasicBlock(channels, norm_fn),
                ISFusionSparseBasicBlock(channels, norm_fn),
            ]
            if next_channels is not None:
                blocks.append(sparse_post_act_block(
                    channels,
                    next_channels,
                    3,
                    stride=2,
                    padding=padding,
                    indice_key=f"spconv{stage_index}",
                    conv_type="spconv",
                    norm_fn=norm_fn,
                ))
            self.encoder_layers.add_module(
                f"encoder_layer{stage_index}", spconv.SparseSequential(*blocks)
            )

        self.conv_out = sparse_post_act_block(
            256,
            256,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=0,
            indice_key="spconv_down2",
            conv_type="spconv",
            norm_fn=norm_fn,
        )
        # conv_out keeps depth=2 for the released [41, 1440, 1440] sparse
        # shape; flattening depth into channels therefore produces 512 BEV
        # channels, exactly as the downstream ISFusion encoder expects.
        self.num_point_features = 512
        self.backbone_channels = {
            "encoder_layer1": 64,
            "encoder_layer2": 128,
            "encoder_layer3": 256,
            "encoder_layer4": 256,
        }

    def forward(self, batch_dict):
        sparse = spconv.SparseConvTensor(
            features=batch_dict["voxel_features"],
            indices=batch_dict["voxel_coords"].int(),
            spatial_shape=self.sparse_shape,
            batch_size=int(batch_dict["batch_size"]),
        )
        sparse = self.conv_input(sparse)
        stage_features = [sparse]
        for stage in self.encoder_layers.children():
            sparse = stage(sparse)
            stage_features.append(sparse)
        output = self.conv_out(sparse)
        dense = output.dense()
        batch_size, channels, depth, height, width = dense.shape
        dense = dense.view(batch_size, channels * depth, height, width)

        batch_dict["isfusion_lidar_bev"] = dense
        batch_dict["isfusion_sparse_features"] = stage_features
        batch_dict["encoded_spconv_tensor"] = output
        batch_dict["encoded_spconv_tensor_stride"] = 8
        return batch_dict
