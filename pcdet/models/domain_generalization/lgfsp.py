"""LiDAR-guided geometry-aware feature style perturbation (LGFSP).

This module is training-only and parameter free.  For every selected image
backbone output it computes either process-local or cross-DDP statistics:

* S: channel-wise background style sensitivity;
* R: channel-wise class discrimination from LiDAR-anchored object features;
* W: a channel-wise interpolation weight between clean features and NP.

The caller is responsible for invoking the module only on an output branch;
the clean tensor that continues through later backbone stages must not be
replaced by the returned tensor.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils
from .normalization_perturbation import NormalizationPerturbation


class LiDARGuidedFeatureStylePerturbation(nn.Module):
    """Compute LGFSP statistics and perturb image output features."""

    CACHE_KEY = '_lgfsp_geometry_cache'

    def __init__(self, cfg):
        super().__init__()
        self.num_views = int(cfg.get('NUM_VIEWS', 6))
        self.expected_batch_size = int(cfg.get('EXPECTED_BATCH_SIZE', 3))
        self.strict_batch_size = bool(cfg.get('STRICT_BATCH_SIZE', True))
        self.noise_std = float(cfg.get('NOISE_STD', 0.75))
        self.style_std_weight = float(cfg.get('STYLE_STD_WEIGHT', 1.0))
        self.semantic_weight = float(cfg.get('SEMANTIC_WEIGHT', 1.0))
        self.statistics_scope = str(
            cfg.get('STATISTICS_SCOPE', 'local')
        ).lower()
        if self.statistics_scope not in {'local', 'global_ddp'}:
            raise ValueError(
                'STATISTICS_SCOPE must be local or global_ddp, got '
                f'{self.statistics_scope!r}'
            )
        self.expected_world_size = int(cfg.get('EXPECTED_WORLD_SIZE', 1))
        self.strict_world_size = bool(cfg.get('STRICT_WORLD_SIZE', False))
        self.weight_mode = str(cfg.get('WEIGHT_MODE', 'combined')).lower()
        if self.weight_mode not in {
            'combined', 'style_only', 'semantic_only', 'shuffle'
        }:
            raise ValueError(
                'WEIGHT_MODE must be one of combined, style_only, '
                f'semantic_only or shuffle, got {self.weight_mode!r}'
            )
        self.exclude_masked_cameras_from_style = bool(
            cfg.get('EXCLUDE_MASKED_CAMERAS_FROM_STYLE', True)
        )
        self.match_combined_weight_distribution = bool(
            cfg.get('MATCH_COMBINED_WEIGHT_DISTRIBUTION', False)
        )
        self.weight_min = float(cfg.get('WEIGHT_MIN', 0.0))
        self.weight_max = float(cfg.get('WEIGHT_MAX', 1.0))
        self.fallback_r = float(cfg.get('FALLBACK_R', 0.5))
        self.min_background_pixels = int(
            cfg.get('MIN_BACKGROUND_PIXELS', 16)
        )
        self.min_points_per_instance = int(
            cfg.get('MIN_POINTS_PER_INSTANCE', 1)
        )
        self.max_points_per_instance = int(
            cfg.get('MAX_POINTS_PER_INSTANCE', 64)
        )
        self.eps = float(cfg.get('EPS', 1e-6))

        if self.expected_batch_size < 1:
            raise ValueError('EXPECTED_BATCH_SIZE must be positive')
        if self.expected_world_size < 1:
            raise ValueError('EXPECTED_WORLD_SIZE must be positive')
        if not 0.0 <= self.weight_min <= self.weight_max <= 1.0:
            raise ValueError('LGFSP weights must satisfy 0 <= min <= max <= 1')
        if self.max_points_per_instance < self.min_points_per_instance:
            raise ValueError(
                'MAX_POINTS_PER_INSTANCE must be >= MIN_POINTS_PER_INSTANCE'
            )

        self.np = NormalizationPerturbation(
            method='np', noise_std=self.noise_std, eps=self.eps
        )
        self.last_statistics: Dict[int, Dict[str, object]] = {}
        self.fallback_counts = {'style': 0, 'semantic': 0}

    @property
    def uses_global_ddp_statistics(self):
        return self.statistics_scope == 'global_ddp'

    def _distributed_world_size(self):
        initialized = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if initialized else 1
        if self.uses_global_ddp_statistics and self.strict_world_size:
            if world_size != self.expected_world_size:
                raise RuntimeError(
                    'LGFSP global statistics expect DDP world size '
                    f'{self.expected_world_size}, got {world_size}'
                )
        return world_size

    @staticmethod
    def _all_gather_rows(values, world_size):
        """All-gather a variable number of detached 2-D rows."""
        if world_size == 1:
            return values, [int(values.shape[0])]
        count = torch.tensor(
            [values.shape[0]], device=values.device, dtype=torch.long
        )
        counts = [torch.zeros_like(count) for _ in range(world_size)]
        dist.all_gather(counts, count)
        counts = [int(value.item()) for value in counts]
        maximum = max(counts)
        padded = values.new_zeros((maximum, values.shape[1]))
        padded[:values.shape[0]] = values
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat([
            value[:size] for value, size in zip(gathered, counts)
        ]), counts

    @staticmethod
    def _all_gather_labels(labels, counts, world_size):
        if world_size == 1:
            return labels
        maximum = max(counts)
        padded = labels.new_zeros((maximum,))
        padded[:labels.shape[0]] = labels
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat([
            value[:size] for value, size in zip(gathered, counts)
        ])

    @staticmethod
    def _normalize_channels(values, eps):
        values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0,
                                  neginf=0.0)
        minimum = values.amin()
        span = (values.amax() - minimum).clamp_min(eps)
        return ((values - minimum) / span).clamp_(0.0, 1.0)

    @staticmethod
    def blend(clean, perturbed, weights):
        """Interpolate from clean (W=0) to complete NP (W=1)."""
        if weights.ndim != 1 or weights.numel() != clean.shape[1]:
            raise ValueError(
                f'Expected [{clean.shape[1]}] weights, got {tuple(weights.shape)}'
            )
        weights = weights.to(device=clean.device, dtype=clean.dtype).view(
            1, -1, 1, 1
        )
        blended = clean + weights * (perturbed - clean)
        # Preserve the two mathematical endpoints bitwise.  This makes W=0
        # a strict clean-path oracle and W=1 a strict NP oracle despite
        # floating-point cancellation in ``clean + (perturbed - clean)``.
        blended = torch.where(weights == 0, clean, blended)
        return torch.where(weights == 1, perturbed, blended)

    @staticmethod
    def _compose_image_augmentation(lidar2image, img_aug_matrix):
        """Use the same calibrated composition order as native IS-Fusion."""
        if img_aug_matrix is None:
            return lidar2image
        homography = img_aug_matrix[:, :3, :3].clone()
        homography[:, :, 2] += img_aug_matrix[:, :3, 3]
        composed = lidar2image.clone()
        composed[:, :3, :] = torch.matmul(
            homography.to(lidar2image.dtype), lidar2image[:, :3, :]
        )
        return composed

    def _project_points(self, xyz, batch_dict, batch_index):
        """Project augmented-LiDAR xyz to normalized grids for all cameras."""
        xyz = xyz.reshape(-1, 3).float().clone()
        lidar_aug = batch_dict.get('lidar_aug_matrix', None)
        if lidar_aug is not None:
            matrix = lidar_aug[batch_index].float()
            xyz -= matrix[:3, 3]
            xyz = torch.linalg.inv(matrix[:3, :3]).matmul(xyz.t())
        else:
            xyz = xyz.t()

        img_aug = batch_dict.get('img_aug_matrix', None)
        sample_img_aug = (
            img_aug[batch_index].float() if img_aug is not None else None
        )
        projection = self._compose_image_augmentation(
            batch_dict['lidar2image'][batch_index].float(), sample_img_aug
        )
        camera_points = projection[:, :3, :3].matmul(xyz)
        camera_points += projection[:, :3, 3].reshape(-1, 3, 1)
        depth = camera_points[:, 2].clone()
        safe_depth = camera_points[:, 2:3].clamp(1e-5, 1e4)
        pixels = camera_points[:, :2] / safe_depth

        image_height, image_width = batch_dict['camera_imgs'].shape[-2:]
        grid = pixels.transpose(1, 2)
        grid[..., 0] /= float(image_width)
        grid[..., 1] /= float(image_height)
        grid = (grid - 0.5) * 2.0
        in_front = depth > 1e-5
        valid = (
            in_front
            & (grid[..., 0] > -1.0) & (grid[..., 0] < 1.0)
            & (grid[..., 1] > -1.0) & (grid[..., 1] < 1.0)
        )
        return grid, valid, in_front

    @staticmethod
    def _valid_boxes(gt_boxes):
        if gt_boxes.shape[-1] < 8:
            raise ValueError('gt_boxes must include a final class label')
        valid = (
            (gt_boxes[:, 3:6] > 0).all(dim=1)
            & (gt_boxes[:, -1] > 0)
        )
        return gt_boxes[valid, :7].float(), gt_boxes[valid, -1].long()

    @torch.no_grad()
    def _build_geometry_cache(self, batch_dict):
        """Assign points to GT once; all selected feature levels reuse it."""
        if self.CACHE_KEY in batch_dict:
            return batch_dict[self.CACHE_KEY]

        batch_size = int(batch_dict['batch_size'])
        cache: List[Dict[str, torch.Tensor]] = []
        all_points = batch_dict['points']
        all_gt_boxes = batch_dict['gt_boxes']
        for batch_index in range(batch_size):
            boxes, labels = self._valid_boxes(all_gt_boxes[batch_index])
            sample_points = all_points[
                all_points[:, 0].long() == batch_index, 1:4
            ].float()
            empty_ids = torch.empty(
                (0,), dtype=torch.long, device=sample_points.device
            )
            if boxes.numel() == 0 or sample_points.numel() == 0:
                cache.append({
                    'boxes': boxes,
                    'labels': labels,
                    'xyz': sample_points[:0],
                    'instance_ids': empty_ids,
                })
                continue

            assignments = roiaware_pool3d_utils.points_in_boxes_gpu(
                sample_points.contiguous().unsqueeze(0),
                boxes.contiguous().unsqueeze(0),
            )[0].long()
            selected_xyz, selected_ids = [], []
            for instance_index in range(boxes.shape[0]):
                point_indices = torch.nonzero(
                    assignments == instance_index, as_tuple=False
                ).flatten()
                if point_indices.numel() < self.min_points_per_instance:
                    continue
                if point_indices.numel() > self.max_points_per_instance:
                    positions = torch.linspace(
                        0, point_indices.numel() - 1,
                        self.max_points_per_instance,
                        device=point_indices.device,
                    ).round().long()
                    point_indices = point_indices[positions]
                selected_xyz.append(sample_points[point_indices])
                selected_ids.append(torch.full(
                    (point_indices.numel(),), instance_index,
                    dtype=torch.long, device=point_indices.device,
                ))
            cache.append({
                'boxes': boxes,
                'labels': labels,
                'xyz': (
                    torch.cat(selected_xyz, dim=0)
                    if selected_xyz else sample_points[:0]
                ),
                'instance_ids': (
                    torch.cat(selected_ids, dim=0)
                    if selected_ids else empty_ids
                ),
            })

        batch_dict[self.CACHE_KEY] = cache
        return cache

    def _target_masks(self, features_5d, batch_dict, geometry):
        """Rasterize projected 3D-box rectangles at this feature resolution."""
        batch_size, num_views, _, height, width = features_5d.shape
        masks = torch.zeros(
            (batch_size, num_views, height, width), dtype=torch.bool,
            device=features_5d.device,
        )
        y_coords = torch.arange(height, device=features_5d.device).view(
            1, 1, height, 1
        )
        x_coords = torch.arange(width, device=features_5d.device).view(
            1, 1, 1, width
        )
        for batch_index, sample in enumerate(geometry):
            boxes = sample['boxes']
            if boxes.numel() == 0:
                continue
            corners = box_utils.boxes_to_corners_3d(boxes).reshape(-1, 3)
            grid, _, in_front = self._project_points(
                corners, batch_dict, batch_index
            )
            grid = grid.view(num_views, boxes.shape[0], 8, 2)
            in_front = in_front.view(num_views, boxes.shape[0], 8)
            # Clamping front-facing corners to the image border also handles
            # boxes that are only partially visible.
            feature_x = ((grid[..., 0].clamp(-1, 1) + 1) * 0.5 * width)
            feature_y = ((grid[..., 1].clamp(-1, 1) + 1) * 0.5 * height)
            positive_inf = torch.full_like(feature_x, float('inf'))
            negative_inf = torch.full_like(feature_x, float('-inf'))
            x0 = torch.where(in_front, feature_x, positive_inf).amin(dim=2)
            y0 = torch.where(in_front, feature_y, positive_inf).amin(dim=2)
            x1 = torch.where(in_front, feature_x, negative_inf).amax(dim=2)
            y1 = torch.where(in_front, feature_y, negative_inf).amax(dim=2)
            valid_rect = in_front.sum(dim=2) >= 2
            x0 = x0.floor().clamp(0, width - 1)
            x1 = x1.ceil().clamp(0, width - 1)
            y0 = y0.floor().clamp(0, height - 1)
            y1 = y1.ceil().clamp(0, height - 1)
            rectangles = (
                valid_rect[:, :, None, None]
                & (x_coords >= x0[:, :, None, None])
                & (x_coords <= x1[:, :, None, None])
                & (y_coords >= y0[:, :, None, None])
                & (y_coords <= y1[:, :, None, None])
            )
            masks[batch_index] = rectangles.any(dim=1)
        return masks

    def _background_moments(self, features_5d, target_masks,
                            active_camera_masks=None):
        detached = features_5d.detach().float()
        background = (~target_masks).unsqueeze(2).float()
        counts = background.sum(dim=(3, 4))
        valid_images = counts[:, :, 0] >= self.min_background_pixels
        if active_camera_masks is not None:
            if active_camera_masks.shape != valid_images.shape:
                raise ValueError(
                    'active_camera_masks must have shape '
                    f'{tuple(valid_images.shape)}, got '
                    f'{tuple(active_camera_masks.shape)}'
                )
            valid_images = valid_images & active_camera_masks.bool()
        safe_counts = counts.clamp_min(1.0)
        means = (detached * background).sum(dim=(3, 4)) / safe_counts
        centered = detached - means[:, :, :, None, None]
        variances = (
            centered.square() * background
        ).sum(dim=(3, 4)) / safe_counts
        stds = variances.clamp_min(0).sqrt()
        means = means.reshape(-1, detached.shape[2])
        stds = stds.reshape(-1, detached.shape[2])
        valid_images = valid_images.reshape(-1)
        return means[valid_images], stds[valid_images]

    def _style_from_moments(self, means, stds, channels, device):
        if means.shape[0] < 2:
            self.fallback_counts['style'] += 1
            return torch.zeros(
                channels, dtype=torch.float32, device=device
            ), True
        score = (
            means.var(dim=0, unbiased=False)
            + self.style_std_weight * stds.var(dim=0, unbiased=False)
        )
        return self._normalize_channels(score, self.eps), False

    def _style_sensitivity(self, features_5d, target_masks,
                           active_camera_masks=None):
        means, stds = self._background_moments(
            features_5d, target_masks, active_camera_masks
        )
        return self._style_from_moments(
            means, stds, features_5d.shape[2], features_5d.device
        )

    def _active_camera_mask(self, batch_dict, batch_index, device):
        active = torch.ones(self.num_views, dtype=torch.bool, device=device)
        if 'img_mask_idx' not in batch_dict:
            return active
        indices = batch_dict['img_mask_idx'][batch_index]
        indices = indices[indices >= 0].long()
        if indices.numel():
            active[indices] = False
        return active

    def _active_camera_masks(self, batch_dict, batch_size, device):
        return torch.stack([
            self._active_camera_mask(batch_dict, batch_index, device)
            for batch_index in range(batch_size)
        ])

    def _combine_weights(self, style, semantic):
        combined = style / (
            style + self.semantic_weight * semantic + self.eps
        )
        if self.weight_mode == 'style_only':
            weights = style
        elif self.weight_mode == 'semantic_only':
            weights = 1.0 - semantic
        else:
            weights = combined
            if self.weight_mode == 'shuffle':
                cuda_devices = (
                    [weights.device.index]
                    if weights.is_cuda and weights.device.index is not None
                    else []
                )
                # The negative control must not change the global torch RNG
                # consumed by the frozen Swin main path.
                with torch.random.fork_rng(
                    devices=cuda_devices, enabled=True
                ):
                    permutation = torch.randperm(
                        weights.numel(), device=weights.device
                    )
                weights = weights[permutation]
        if (
            self.match_combined_weight_distribution
            and self.weight_mode in {'style_only', 'semantic_only'}
        ):
            # Preserve the exact combined-W multiset and replace only its
            # channel ranking. This isolates S-only/R-only channel selection
            # from the otherwise very different overall perturbation strength.
            rank_order = weights.argsort()
            matched = torch.empty_like(weights)
            matched[rank_order] = combined.sort().values
            weights = matched
        return weights.clamp(self.weight_min, self.weight_max).detach()

    def _instance_features(self, features_5d, batch_dict, geometry):
        """Pool projected LiDAR point features once per GT instance."""
        batch_size, num_views, channels, _, _ = features_5d.shape
        pooled_features, pooled_labels = [], []
        for batch_index in range(batch_size):
            sample = geometry[batch_index]
            xyz = sample['xyz']
            instance_ids = sample['instance_ids']
            num_instances = sample['boxes'].shape[0]
            if xyz.numel() == 0 or num_instances == 0:
                continue
            grid, valid, _ = self._project_points(
                xyz, batch_dict, batch_index
            )
            active_cameras = self._active_camera_mask(
                batch_dict, batch_index, features_5d.device
            )
            sums = features_5d.new_zeros((num_instances, channels)).float()
            counts = features_5d.new_zeros((num_instances, 1)).float()
            for camera_index in range(num_views):
                camera_valid = valid[camera_index] & active_cameras[camera_index]
                if not bool(camera_valid.any()):
                    continue
                sampled = F.grid_sample(
                    features_5d[
                        batch_index, camera_index
                    ].detach().float().unsqueeze(0),
                    grid[camera_index].view(1, -1, 1, 2),
                    mode='bilinear', padding_mode='zeros', align_corners=False,
                ).view(channels, -1).transpose(0, 1)
                ids = instance_ids[camera_valid]
                sums.index_add_(0, ids, sampled[camera_valid])
                counts.index_add_(
                    0, ids, counts.new_ones((ids.numel(), 1))
                )
            has_features = counts[:, 0] > 0
            if has_features.any():
                pooled_features.append(
                    sums[has_features] / counts[has_features]
                )
                pooled_labels.append(sample['labels'][has_features])
        if not pooled_features:
            return (
                features_5d.new_zeros((0, channels)).float(),
                torch.empty(
                    (0,), dtype=torch.long, device=features_5d.device
                ),
            )
        return torch.cat(pooled_features, dim=0), torch.cat(pooled_labels)

    def _semantic_relevance(self, instance_features, labels, channels, device):
        """Class-balanced Fisher score over local-mini-batch instances."""
        unique_labels = torch.unique(labels)
        if instance_features.shape[0] < 2 or unique_labels.numel() < 2:
            self.fallback_counts['semantic'] += 1
            return torch.full(
                (channels,), self.fallback_r, device=device,
                dtype=torch.float32,
            ), True, int(instance_features.shape[0]), int(unique_labels.numel())

        class_means, class_variances = [], []
        for label in unique_labels:
            class_features = instance_features[labels == label]
            class_mean = class_features.mean(dim=0)
            class_means.append(class_mean)
            class_variances.append(
                (class_features - class_mean).square().mean(dim=0)
            )
        class_means = torch.stack(class_means)
        class_variances = torch.stack(class_variances)
        balanced_global_mean = class_means.mean(dim=0)
        between = (
            class_means - balanced_global_mean
        ).square().mean(dim=0)
        within = class_variances.mean(dim=0)
        score = between / (within + self.eps)
        return (
            self._normalize_channels(score, self.eps), False,
            int(instance_features.shape[0]), int(unique_labels.numel()),
        )

    def compute_statistics(self, features, batch_dict, stage_index):
        batch_size = int(batch_dict['batch_size'])
        if self.strict_batch_size and batch_size != self.expected_batch_size:
            raise ValueError(
                'LGFSP formal protocol expects local mini-batch '
                f'{self.expected_batch_size}, got {batch_size}'
            )
        if features.shape[0] != batch_size * self.num_views:
            raise ValueError(
                f'Expected B*N={batch_size * self.num_views} image features, '
                f'got {features.shape[0]}'
            )
        features_5d = features.view(
            batch_size, self.num_views, features.shape[1],
            features.shape[2], features.shape[3],
        )
        geometry = self._build_geometry_cache(batch_dict)
        target_masks = self._target_masks(features_5d, batch_dict, geometry)
        active_camera_masks = self._active_camera_masks(
            batch_dict, batch_size, features.device
        )
        background_means, background_stds = self._background_moments(
            features_5d, target_masks,
            active_camera_masks=(
                active_camera_masks
                if self.exclude_masked_cameras_from_style else None
            ),
        )
        instance_features, labels = self._instance_features(
            features_5d, batch_dict, geometry
        )
        world_size = self._distributed_world_size()
        local_num_active_images = int(background_means.shape[0])
        local_num_instances = int(instance_features.shape[0])
        if self.uses_global_ddp_statistics and world_size > 1:
            background_means, active_image_counts = self._all_gather_rows(
                background_means, world_size
            )
            background_stds, _ = self._all_gather_rows(
                background_stds, world_size
            )
            instance_features, instance_counts = self._all_gather_rows(
                instance_features, world_size
            )
            labels = self._all_gather_labels(
                labels, instance_counts, world_size
            )
        else:
            active_image_counts = [local_num_active_images]
            instance_counts = [local_num_instances]
        style, style_fallback = self._style_from_moments(
            background_means, background_stds,
            features.shape[1], features.device,
        )
        semantic, semantic_fallback, num_instances, num_classes = (
            self._semantic_relevance(
                instance_features, labels, features.shape[1], features.device
            )
        )
        weights = self._combine_weights(style, semantic)
        _, class_counts = torch.unique(labels, return_counts=True)
        singleton_classes = int((class_counts == 1).sum())
        singleton_instances = int(class_counts[class_counts == 1].sum())
        self.last_statistics[int(stage_index)] = {
            'S': style.detach(),
            'R': semantic.detach(),
            'W': weights,
            'target_masks': target_masks.detach(),
            'num_instances': num_instances,
            'num_classes': num_classes,
            'class_counts': class_counts.detach(),
            'singleton_classes': singleton_classes,
            'singleton_instance_fraction': (
                float(singleton_instances / max(num_instances, 1))
            ),
            'num_active_images': int(sum(active_image_counts)),
            'local_num_active_images': local_num_active_images,
            'local_num_instances': local_num_instances,
            'statistics_scope': self.statistics_scope,
            'world_size': world_size,
            'global_batch_size': batch_size * world_size,
            'weight_mode': self.weight_mode,
            'matched_combined_weight_distribution': (
                self.match_combined_weight_distribution
            ),
            'style_fallback': style_fallback,
            'semantic_fallback': semantic_fallback,
            'local_batch_size': batch_size,
            'num_images': batch_size * self.num_views,
        }
        return style, semantic, weights

    def forward(self, features, batch_dict, stage_index):
        _, _, weights = self.compute_statistics(
            features, batch_dict, stage_index
        )
        # Output-branch sampling must not advance the global torch RNG used by
        # later frozen Swin stages (notably DropPath).  ``fork_rng`` gives the
        # branch real random NP samples while restoring the main-path state.
        cuda_devices = (
            [features.device.index]
            if features.is_cuda and features.device.index is not None else []
        )
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            perturbed = self.np(features)
        return self.blend(features, perturbed, weights)
