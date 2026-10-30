"""Native PyTorch implementation of the SST blocks used by ISFusion.

The module intentionally keeps the parameter hierarchy of the original CCF
implementation so that ``isfusion_source.pth`` can be used as a numerical
oracle.  It has no dependency on MMCV, MMDetection or MMDetection3D.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@torch.no_grad()
def get_inner_window_indices(window_indices: torch.Tensor) -> torch.Tensor:
    """Return the occurrence number of every item inside its window.

    The original implementation uses the custom ``ingroup_indices`` CUDA
    extension.  Stable sorting preserves the input order within each window,
    which is exactly the operation performed by that extension.
    """
    if window_indices.numel() == 0:
        return torch.empty_like(window_indices)
    order = torch.argsort(window_indices, stable=True)
    sorted_indices = window_indices[order]
    _, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    inner_sorted = torch.arange(
        window_indices.numel(), device=window_indices.device,
        dtype=window_indices.dtype,
    ) - torch.repeat_interleave(starts, counts)
    inner = torch.empty_like(inner_sorted)
    inner[order] = inner_sorted
    return inner


@torch.no_grad()
def make_continuous_indices(indices: torch.Tensor) -> torch.Tensor:
    unique_indices = torch.unique(indices, sorted=True)
    return torch.searchsorted(unique_indices, indices)


@torch.no_grad()
def get_window_coordinates(coordinates, sparse_shape, window_shape, do_shift):
    if len(window_shape) == 2:
        win_x, win_y = window_shape
        win_z = sparse_shape[-1]
    else:
        win_x, win_y, win_z = window_shape

    sparse_x, sparse_y, sparse_z = sparse_shape
    if sparse_z >= sparse_x:
        raise ValueError('Expected sparse shape in (x, y, z) order')
    max_win_x = int(math.ceil(sparse_x / win_x) + 1)
    max_win_y = int(math.ceil(sparse_y / win_y) + 1)
    max_win_z = int(math.ceil(sparse_z / win_z) + 1)
    max_windows = max_win_x * max_win_y * max_win_z

    if do_shift:
        shift_x, shift_y, shift_z = win_x // 2, win_y // 2, win_z // 2
    else:
        shift_x, shift_y, shift_z = win_x, win_y, win_z
    if sparse_z == win_z:
        shift_z = 0

    shifted_x = coordinates[:, 3] + shift_x
    shifted_y = coordinates[:, 2] + shift_y
    shifted_z = coordinates[:, 1] + shift_z
    window_x = shifted_x // win_x
    window_y = shifted_y // win_y
    window_z = shifted_z // win_z
    batch_window_indices = (
        coordinates[:, 0] * max_windows
        + window_x * max_win_y * max_win_z
        + window_y * max_win_z
        + window_z
    )
    coordinates_in_window = torch.stack(
        (shifted_z % win_z, shifted_y % win_y, shifted_x % win_x), dim=-1
    )
    return batch_window_indices, coordinates_in_window


@torch.no_grad()
def get_flat_to_window_indices(batch_window_indices, drop_levels, drop_info,
                               debug=True):
    result = {}
    for level in drop_info:
        mask = drop_levels == level
        if not mask.any():
            continue
        continuous = make_continuous_indices(batch_window_indices[mask])
        max_tokens = drop_info[level]['max_tokens']
        inner = get_inner_window_indices(continuous)
        flat_indices = continuous * max_tokens + inner
        result[level] = (flat_indices, torch.where(mask))
        if debug:
            assert inner.max() < max_tokens
            assert flat_indices.min() >= 0
    result['voxel_drop_level'] = drop_levels
    result['batching_info'] = drop_info
    return result


def flat_to_window(features, indices_dict):
    drop_levels = indices_dict['voxel_drop_level']
    drop_info = indices_dict['batching_info']
    result = {}
    for level in drop_info:
        mask = drop_levels == level
        if not mask.any():
            continue
        indices = indices_dict[level][0]
        max_tokens = drop_info[level]['max_tokens']
        num_windows = int(indices.max().item() // max_tokens + 1)
        window_features = features.new_zeros(
            num_windows * max_tokens, features.shape[-1]
        )
        window_features[indices] = features[mask]
        result[level] = window_features.view(
            num_windows, max_tokens, features.shape[-1]
        )
    return result


def window_to_flat(window_features, indices_dict):
    num_voxels = indices_dict['voxel_drop_level'].numel()
    first = next(iter(window_features.values()))
    result = first.new_zeros(num_voxels, first.shape[-1])
    for level, features in window_features.items():
        indices, flat_positions = indices_dict[level]
        result[flat_positions] = features.reshape(-1, features.shape[-1])[indices]
    return result


class ISFusionSSTInputLayer(nn.Module):
    """Window grouping, token dropping, positional embedding and masks."""

    def __init__(self, drop_info, window_shape, sparse_shape,
                 shuffle_voxels=True, debug=False, normalize_pos=False,
                 pos_temperature=10000, mute=True, pos_embed=256):
        super().__init__()
        self.meta_drop_info = drop_info
        self.sparse_shape = tuple(sparse_shape)
        self.shuffle_voxels = shuffle_voxels
        self.debug = debug
        self.window_shape = tuple(window_shape)
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature
        self.mute = mute
        self.pos_embed_channels = pos_embed

    @property
    def drop_info(self):
        return self.meta_drop_info

    def forward(self, voxel_feats, voxel_coors, batch_size=None):
        voxel_coors = voxel_coors.long()
        if self.shuffle_voxels:
            shuffle_indices = torch.randperm(
                len(voxel_feats), device=voxel_feats.device
            )
            voxel_feats = voxel_feats[shuffle_indices]
            voxel_coors = voxel_coors[shuffle_indices]

        voxel_info = self.window_partition(voxel_coors)
        voxel_info['voxel_feats'] = voxel_feats
        voxel_info['voxel_coors'] = voxel_coors
        voxel_info = self.drop_voxels(voxel_info)

        for shift in range(2):
            indices = get_flat_to_window_indices(
                voxel_info[f'batch_win_inds_shift{shift}'],
                voxel_info[f'voxel_drop_level_shift{shift}'],
                self.drop_info,
                debug=self.debug,
            )
            voxel_info[f'flat2win_inds_shift{shift}'] = indices
            voxel_info[f'pos_dict_shift{shift}'] = self.get_position_embedding(
                indices, voxel_info[f'coors_in_win_shift{shift}'],
                self.pos_embed_channels, voxel_feats.dtype,
            )
            voxel_info[f'key_mask_shift{shift}'] = self.get_padding_mask(indices)
        if self.shuffle_voxels:
            voxel_info['shuffle_inds'] = shuffle_indices
        return voxel_info

    @torch.no_grad()
    def window_partition(self, coordinates):
        result = {}
        for shift in range(2):
            windows, inner = get_window_coordinates(
                coordinates, self.sparse_shape, self.window_shape, shift == 1
            )
            result[f'batch_win_inds_shift{shift}'] = windows
            result[f'coors_in_win_shift{shift}'] = inner
        return result

    def _drop_single_shift(self, window_indices):
        drop_levels = torch.full_like(window_indices, -1)
        inner = get_inner_window_indices(window_indices)
        counts = torch.bincount(window_indices)
        counts_per_voxel = counts[window_indices]
        target = torch.zeros_like(window_indices)
        for level, info in self.drop_info.items():
            lower, upper = info['drop_range']
            mask = (counts_per_voxel >= lower) & (counts_per_voxel < upper)
            target[mask] = info['max_tokens']
            drop_levels[mask] = level
        if (target <= 0).any() or (drop_levels < 0).any():
            raise ValueError('drop_info does not cover every SST window')
        return inner < target, drop_levels

    def drop_voxels(self, voxel_info):
        initial_count = len(voxel_info['voxel_coors'])
        keep_indices = torch.arange(
            initial_count, device=voxel_info['voxel_coors'].device
        )
        windows0 = voxel_info['batch_win_inds_shift0']
        keep0, levels0 = self._drop_single_shift(windows0)
        keep_indices = keep_indices[keep0]
        windows0 = windows0[keep0]
        levels0 = levels0[keep0]

        windows1 = voxel_info['batch_win_inds_shift1'][keep0]
        keep1, levels1 = self._drop_single_shift(windows1)
        keep_indices = keep_indices[keep1]
        windows0 = windows0[keep1]
        levels0 = levels0[keep1]
        windows1 = windows1[keep1]
        levels1 = levels1[keep1]

        voxel_info['voxel_keep_inds'] = keep_indices
        voxel_info['batch_win_inds_shift0'] = windows0
        voxel_info['voxel_drop_level_shift0'] = levels0
        voxel_info['batch_win_inds_shift1'] = windows1
        voxel_info['voxel_drop_level_shift1'] = levels1
        for key, value in list(voxel_info.items()):
            if isinstance(value, torch.Tensor) and len(value) == initial_count:
                voxel_info[key] = value[keep_indices]
        return voxel_info

    @torch.no_grad()
    def get_position_embedding(self, indices, coordinates, feature_dim, dtype):
        if len(self.window_shape) == 2 or self.window_shape[-1] == 1:
            dimensions = 2
            win_x, win_y = self.window_shape[:2]
            win_z = 0
        else:
            dimensions = 3
            win_x, win_y, win_z = self.window_shape
        z = coordinates[:, 0] - win_z / 2
        y = coordinates[:, 1] - win_y / 2
        x = coordinates[:, 2] - win_x / 2
        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415
            y = y / win_y * 2 * 3.1415
            z = z / win_z * 2 * 3.1415
        length = feature_dim // dimensions
        frequency = torch.arange(
            length, dtype=torch.float32, device=coordinates.device
        )
        frequency = self.pos_temperature ** (2 * (frequency // 2) / length)

        def encode(position):
            embedding = position[:, None] / frequency[None, :]
            return torch.stack(
                (embedding[:, ::2].sin(), embedding[:, 1::2].cos()), dim=-1
            ).flatten(1)

        embeddings = [encode(x), encode(y)]
        if dimensions == 3:
            embeddings.append(encode(z))
        position = torch.cat(embeddings, dim=-1).to(dtype)
        if position.shape[1] < feature_dim:
            position = F.pad(position, (0, feature_dim - position.shape[1]))
        return flat_to_window(position, indices)

    @staticmethod
    @torch.no_grad()
    def get_padding_mask(indices):
        valid = torch.ones(
            (len(indices['voxel_drop_level']), 1), dtype=torch.bool,
            device=indices['voxel_drop_level'].device,
        )
        return {
            level: value.logical_not().squeeze(2)
            for level, value in flat_to_window(valid, indices).items()
        }


class WindowAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, batch_first=False,
                 layer_id=None, layer_cfg=None):
        super().__init__()
        layer_cfg = layer_cfg or {}
        if layer_cfg.get('cosine') or layer_cfg.get('linear'):
            raise NotImplementedError('ISFusion uses standard multi-head attention')
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.layer_id = layer_id

    def forward(self, features, position_dict, index_dict, padding_dict):
        window_features = flat_to_window(features, index_dict)
        outputs = {}
        for level, feature in window_features.items():
            feature = feature.permute(1, 0, 2)
            position = position_dict[level].permute(1, 0, 2)
            query = key = feature + position
            output, _ = self.self_attn(
                query, key, value=feature,
                key_padding_mask=padding_dict[level],
            )
            outputs[level] = output.permute(1, 0, 2)
        return window_to_flat(outputs, index_dict)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', batch_first=False, layer_id=None,
                 mlp_dropout=0, layer_cfg=None):
        super().__init__()
        layer_cfg = layer_cfg or {}
        self.win_attn = WindowAttention(
            d_model, nhead, dropout, batch_first, layer_id, layer_cfg
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        if layer_cfg.get('use_bn', False):
            raise NotImplementedError('Native ISFusion currently supports LayerNorm')
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)
        self.activation = getattr(F, activation)
        self.post_norm = layer_cfg.get('post_norm', True)

    def forward(self, source, position_dict, index_dict, padding_dict):
        if self.post_norm:
            attended = self.win_attn(
                source, position_dict, index_dict, padding_dict
            )
            source = self.norm1(source + self.dropout1(attended))
            feedforward = self.linear2(
                self.dropout(self.activation(self.linear1(source)))
            )
            return self.norm2(source + self.dropout2(feedforward))
        normalized = self.norm1(source)
        attended = self.win_attn(
            normalized, position_dict, index_dict, padding_dict
        )
        source = source + self.dropout1(attended)
        normalized = self.norm2(source)
        feedforward = self.linear2(
            self.dropout(self.activation(self.linear1(normalized)))
        )
        return source + self.dropout2(feedforward)


class BasicShiftBlockV2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', batch_first=False, block_id=-100,
                 layer_cfg=None):
        super().__init__()
        self.encoder_list = nn.ModuleList([
            EncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                batch_first, block_id * 2 + shift, layer_cfg=layer_cfg,
            )
            for shift in range(2)
        ])

    def forward(self, source, position_list, index_list, padding_list,
                using_checkpoint=False):
        output = source
        for index, layer in enumerate(self.encoder_list):
            shift = index % len(position_list)
            if using_checkpoint and self.training:
                output = checkpoint(
                    layer, output, position_list[shift], index_list[shift],
                    padding_list[shift], use_reentrant=False,
                )
            else:
                output = layer(
                    output, position_list[shift], index_list[shift],
                    padding_list[shift],
                )
        return output


class ISFusionSST(nn.Module):
    """Single-stride sparse transformer with CCF-compatible names."""

    def __init__(self, d_model, nhead, num_blocks=1, dim_feedforward=None,
                 dropout=0.0, activation='gelu', output_shape=None,
                 in_channel=None, checkpoint_blocks=None, layer_cfg=None):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model
        self.checkpoint_blocks = checkpoint_blocks or []
        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])
        self.block_list = nn.ModuleList([
            BasicShiftBlockV2(
                d_model[index], nhead[index], dim_feedforward[index], dropout,
                activation, block_id=index, layer_cfg=layer_cfg,
            )
            for index in range(num_blocks)
        ])
        self.output_shape = tuple(output_shape)
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, voxel_info):
        coordinates = voxel_info['voxel_coors']
        batch_size = int(coordinates[:, 0].max().item()) + 1
        indices = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(2)]
        masks = [voxel_info[f'key_mask_shift{i}'] for i in range(2)]
        positions = [voxel_info[f'pos_dict_shift{i}'] for i in range(2)]
        output = voxel_info['voxel_feats']
        if hasattr(self, 'linear0'):
            output = self.linear0(output)
        for index, block in enumerate(self.block_list):
            output = block(
                output, positions, indices, masks,
                using_checkpoint=index in self.checkpoint_blocks,
            )
        return [self.recover_bev(output, coordinates, batch_size)]

    def recover_bev(self, features, coordinates, batch_size):
        height, width = self.output_shape
        output = features.new_zeros(
            batch_size, features.shape[-1], height * width
        )
        for batch_index in range(batch_size):
            mask = coordinates[:, 0] == batch_index
            indices = coordinates[mask, 2] * width + coordinates[mask, 3]
            output[batch_index, :, indices.long()] = features[mask].t()
        return output.view(batch_size, features.shape[-1], height, width)


class _ConvFusion(nn.Module):
    """Conv-BN-ReLU with the parameter names produced by MMCV ConvModule."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(768, 128, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.activate(self.bn(self.conv(features)))


class ISFusionFusionEncoder(nn.Module):
    """Two-stage image/LiDAR SST fusion encoder.

    Pillar image sampling is kept as the preceding OpenPCDet module. This
    class consumes ``isfusion_image_bev`` and ``isfusion_lidar_bev`` and uses
    the supplied progressive SECOND backbone between its two SST stages.
    """

    def __init__(self, model_cfg=None):
        super().__init__()
        self.conv_fusion = _ConvFusion()
        self.get_regions = nn.ModuleList([
            ISFusionSSTInputLayer(
                drop_info={0: {'max_tokens': 36, 'drop_range': (0, 100000)}},
                window_shape=(6, 6, 1), sparse_shape=(180, 180, 1),
                shuffle_voxels=True, normalize_pos=False,
                pos_temperature=1000, pos_embed=128,
            ),
            ISFusionSSTInputLayer(
                drop_info={0: {'max_tokens': 36, 'drop_range': (0, 100000)}},
                window_shape=(6, 6, 1), sparse_shape=(90, 90, 1),
                shuffle_voxels=True, normalize_pos=False,
                pos_temperature=1000, pos_embed=256,
            ),
        ])
        self.grid2region_att = nn.ModuleList([
            ISFusionSST(
                d_model=[128] * 4, nhead=[8] * 4, num_blocks=1,
                dim_feedforward=[128] * 4, output_shape=(180, 180),
                in_channel=128,
            ),
            ISFusionSST(
                d_model=[256] * 4, nhead=[8] * 4, num_blocks=1,
                dim_feedforward=[256] * 4, output_shape=(90, 90),
                in_channel=None,
            ),
        ])

    @staticmethod
    def create_dense_coordinates(height, width, batch_size, device):
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device), indexing='ij',
        )
        coordinates = []
        for batch_index in range(batch_size):
            batch = torch.full_like(y, batch_index)
            coordinates.append(torch.stack(
                (batch, torch.zeros_like(y), y, x), dim=-1
            ).view(-1, 4))
        return torch.cat(coordinates, dim=0).long()

    def forward_features(self, image_bev, lidar_bev, progressive_backbone):
        fused = self.conv_fusion(torch.cat((image_bev, lidar_bev), dim=1))
        batch_size, channels, height, width = fused.shape
        flat = fused.flatten(2).transpose(1, 2).reshape(-1, channels)
        coordinates = self.create_dense_coordinates(
            height, width, batch_size, fused.device
        )
        multi_scale = []
        for stage in range(2):
            voxel_info = self.get_regions[stage](flat, coordinates, batch_size)
            attended = self.grid2region_att[stage](voxel_info)
            flat, coordinates, dense = progressive_backbone(
                attended, f'stage{stage + 1}'
            )
            multi_scale.append(dense)
        return multi_scale

    def forward(self, batch_dict, progressive_backbone=None):
        if progressive_backbone is None:
            progressive_backbone = batch_dict.get('isfusion_progressive_backbone')
        if progressive_backbone is None:
            raise ValueError('ISFusionFusionEncoder requires a progressive backbone')
        batch_dict['isfusion_multi_scale_bev'] = self.forward_features(
            batch_dict['isfusion_image_bev'],
            batch_dict['isfusion_lidar_bev'],
            progressive_backbone,
        )
        return batch_dict
