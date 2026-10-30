"""Native pillar construction and multi-camera image sampling for ISFusion."""

from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn import functional as F


class ISFusionPillarImageSampler(nn.Module):
    """Project points in non-empty BEV pillars into all camera feature maps."""

    def __init__(self, model_cfg):
        super().__init__()
        self.num_points = int(model_cfg.get('NUM_POINTS_IN_PILLAR', 20))
        self.num_views = int(model_cfg.get('NUM_VIEWS', 6))
        self.embed_dims = int(model_cfg.get('EMBED_DIMS', 256))
        self.bev_size = int(model_cfg.get('BEV_SIZE', 180))
        self.point_cloud_range = list(model_cfg.POINT_CLOUD_RANGE)
        self.pillar_size = list(model_cfg.get('PILLAR_SIZE', [0.6, 0.6, 8.0]))
        # CCF uses (30k, 60k) in its MMCV Voxelization layer.  The distinct
        # training/test limits matter for exact source-compatible behaviour.
        self.max_voxels_train = int(model_cfg.get('MAX_VOXELS_TRAIN', 30000))
        self.max_voxels_test = int(model_cfg.get('MAX_VOXELS_TEST', model_cfg.get('MAX_VOXELS', 60000)))
        self.feature_level = int(model_cfg.get('FEATURE_LEVEL', 1))
        self.random_noise = float(model_cfg.get('RANDOM_NOISE', 1.0))
    @torch.no_grad()
    def _voxelize_one(self, sample_points):
        """Match MMCV hard voxelization's deterministic first-N rule.

        ``spconv.PointToVoxel`` uses concurrent writes and can retain a
        different subset when a pillar holds more than ``num_points`` points.
        ISFusion projects every retained point into images, so that seemingly
        small distinction changes the fused BEV.  MMCV's CCF operator keeps
        the first points in input order; this vectorized implementation does
        the same without a Python point-by-point loop.
        """
        if sample_points.numel() == 0:
            empty_voxels = sample_points.new_zeros(
                (0, self.num_points, sample_points.shape[1])
            )
            return (
                empty_voxels,
                torch.empty((0, 3), dtype=torch.int32, device=sample_points.device),
                torch.empty((0,), dtype=torch.int32, device=sample_points.device),
            )

        minimum = sample_points.new_tensor(self.point_cloud_range[:3])
        maximum = sample_points.new_tensor(self.point_cloud_range[3:])
        pillar_size = sample_points.new_tensor(self.pillar_size)
        xyz_coords = torch.floor(
            (sample_points[:, :3] - minimum) / pillar_size
        ).long()
        grid_size = torch.round((maximum - minimum) / pillar_size).long()
        valid = ((xyz_coords >= 0) & (xyz_coords < grid_size)).all(dim=1)
        points = sample_points[valid]
        xyz_coords = xyz_coords[valid]
        if points.numel() == 0:
            return self._voxelize_one(points)

        # Stable grouping keeps the original input order within every pillar.
        linear = xyz_coords[:, 0] + grid_size[0] * (
            xyz_coords[:, 1] + grid_size[1] * xyz_coords[:, 2]
        )
        order = torch.argsort(linear, stable=True)
        sorted_linear = linear[order]
        sorted_points = points[order]
        sorted_coords = xyz_coords[order]
        _, point_counts = torch.unique_consecutive(
            sorted_linear, return_counts=True
        )
        starts = torch.cumsum(point_counts, dim=0) - point_counts
        point_rank = torch.arange(order.numel(), device=points.device)
        point_rank -= torch.repeat_interleave(starts, point_counts)
        groups = torch.repeat_interleave(
            torch.arange(point_counts.numel(), device=points.device), point_counts
        )

        # When the number of occupied pillars exceeds CCF's cap, retain the
        # pillars that first occur in the input stream (the hard-voxelizer
        # allocation rule), not lexicographically first coordinates.
        max_voxels = self.max_voxels_train if self.training else self.max_voxels_test
        group_order = torch.argsort(order[starts], stable=True)
        selected_groups = group_order[:max_voxels]
        selected = torch.zeros(point_counts.numel(), dtype=torch.bool, device=points.device)
        selected[selected_groups] = True
        keep = (point_rank < self.num_points) & selected[groups]

        # The output group index follows first occurrence order; pillar order
        # itself does not affect later scatter-by-coordinate operations.
        remap = torch.full_like(groups[:point_counts.numel()], -1)
        remap[selected_groups] = torch.arange(
            selected_groups.numel(), device=points.device
        )
        kept_groups = remap[groups[keep]]
        voxels = points.new_zeros(
            (selected_groups.numel(), self.num_points, points.shape[1])
        )
        voxels[kept_groups, point_rank[keep]] = sorted_points[keep]
        counts = point_counts[selected_groups].clamp(max=self.num_points).to(torch.int32)
        xyz = sorted_coords[starts[selected_groups]]
        coords = torch.stack((xyz[:, 2], xyz[:, 1], xyz[:, 0]), dim=1).to(torch.int32)
        return voxels, coords, counts

    def build_pillars(self, points, batch_size):
        pillars, counts, coordinates = [], [], []
        for batch_index in range(batch_size):
            sample_points = points[points[:, 0].long() == batch_index, 1:]
            voxels, coords, num_points = self._voxelize_one(sample_points)
            batch_column = coords.new_full((coords.shape[0], 1), batch_index)
            pillars.append(voxels)
            counts.append(num_points)
            coordinates.append(torch.cat((batch_column, coords), dim=1))
        return (
            torch.cat(pillars, dim=0),
            torch.cat(counts, dim=0),
            torch.cat(coordinates, dim=0),
        )

    def _project(self, xyz, lidar2image, img_aug_matrix, lidar_aug_matrix,
                 image_height, image_width):
        xyz = xyz.reshape(-1, 3).clone()
        if lidar_aug_matrix is not None:
            xyz -= lidar_aug_matrix[:3, 3]
            xyz = torch.linalg.inv(lidar_aug_matrix[:3, :3]).matmul(xyz.t())
        else:
            xyz = xyz.t()

        projection = lidar2image.float()
        camera_points = projection[:, :3, :3].matmul(xyz)
        camera_points += projection[:, :3, 3].reshape(-1, 3, 1)
        if self.training and self.random_noise > 0 and random.random() > 0.5:
            camera_points += random.uniform(-self.random_noise, self.random_noise)

        depth = camera_points[:, 2].clone()
        camera_points[:, 2].clamp_(1e-5, 1e4)
        camera_points[:, :2] /= camera_points[:, 2:3]
        if img_aug_matrix is not None:
            camera_points = img_aug_matrix[:, :3, :3].matmul(camera_points)
            camera_points += img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)

        grid = camera_points[:, :2].transpose(1, 2)
        grid[..., 0] /= image_width
        grid[..., 1] /= image_height
        grid = (grid - 0.5) * 2
        valid = (
            (depth > 1e-5)
            & (grid[..., 0] > -1.0) & (grid[..., 0] < 1.0)
            & (grid[..., 1] > -1.0) & (grid[..., 1] < 1.0)
        )
        return grid, valid

    @staticmethod
    def _compose_image_augmentation(lidar2image, img_aug_matrix):
        """Fold post-projection image augmentation into lidar2image.

        CCF performs this composition before perspective division. Using the
        same operation order avoids border points moving across the strict
        ``grid_sample`` boundary due solely to floating-point round-off.
        """
        if img_aug_matrix is None:
            return lidar2image
        homography = img_aug_matrix[:, :3, :3].clone()
        homography[:, :, 2] += img_aug_matrix[:, :3, 3]
        composed = lidar2image.clone()
        composed[:, :3, :] = torch.matmul(
            homography.to(lidar2image.dtype), lidar2image[:, :3, :]
        )
        return composed

    def forward(self, batch_dict):
        batch_size = int(batch_dict['batch_size'])
        pillars, counts, coordinates = self.build_pillars(
            batch_dict['points'], batch_size
        )
        image_features = batch_dict['image_fpn'][self.feature_level]
        _, channels, _, _ = image_features.shape
        if channels != self.embed_dims:
            raise ValueError(
                f'Expected {self.embed_dims} image channels, got {channels}'
            )
        image_features = image_features.view(
            batch_size, self.num_views, channels,
            image_features.shape[-2], image_features.shape[-1],
        )
        image_height, image_width = batch_dict['camera_imgs'].shape[-2:]
        image_bev = image_features.new_zeros(
            batch_size, channels, self.bev_size, self.bev_size
        )

        for batch_index in range(batch_size):
            selected = coordinates[:, 0] == batch_index
            sample_pillars = pillars[selected, :, :3]
            sample_coords = coordinates[selected]
            sample_img_aug = (
                batch_dict.get('img_aug_matrix', None)[batch_index]
                if batch_dict.get('img_aug_matrix', None) is not None else None
            )
            sample_lidar2image = self._compose_image_augmentation(
                batch_dict['lidar2image'][batch_index], sample_img_aug
            )
            grid, _ = self._project(
                sample_pillars,
                sample_lidar2image,
                None,
                batch_dict.get('lidar_aug_matrix', None)[batch_index]
                if batch_dict.get('lidar_aug_matrix', None) is not None else None,
                image_height,
                image_width,
            )
            sampled = []
            for camera_index in range(self.num_views):
                camera_grid = grid[camera_index].view(-1, 1, 2)
                camera_features = F.grid_sample(
                    image_features[batch_index, camera_index].unsqueeze(0),
                    camera_grid.to(dtype=image_features.dtype).unsqueeze(0),
                    mode='bilinear', padding_mode='zeros', align_corners=False,
                )
                sampled.append(camera_features.view(channels, -1))
            sampled = torch.stack(sampled, dim=0).sum(dim=0)
            sampled = sampled.view(channels, -1, self.num_points).sum(dim=2)
            image_bev[
                batch_index, :, sample_coords[:, 2].long(),
                sample_coords[:, 3].long()
            ] = sampled.to(dtype=image_bev.dtype)

        batch_dict['isfusion_pillars'] = pillars
        batch_dict['isfusion_pillar_num_points'] = counts
        batch_dict['isfusion_pillar_coords'] = coordinates
        batch_dict['isfusion_image_bev'] = image_bev
        return batch_dict
