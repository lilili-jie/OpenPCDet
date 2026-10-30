"""Native SECOND and FPN components used by ISFusion."""

from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1,
                  padding=1):
    return [
        nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    ]


class ISFusionSECOND(nn.Module):
    """Progressive two-stage SECOND backbone used between SST stages.

    ``forward_stage1`` consumes the 180x180 SST feature, returns its dense
    feature plus flattened 90x90 tokens for the next SST stage.
    ``forward_stage2`` consumes the second SST feature and returns its dense
    90x90 feature. Parameter names match CCF's ``SECONDV2``.
    """

    def __init__(self, model_cfg=None, input_channels=128):
        super().__init__()
        if model_cfg is None:
            layer_nums = [5, 5]
            layer_strides = [1, 2]
            out_channels = [128, 256]
        else:
            layer_nums = list(model_cfg.get('LAYER_NUMS', [5, 5]))
            layer_strides = list(model_cfg.get('LAYER_STRIDES', [1, 2]))
            out_channels = list(model_cfg.get('NUM_FILTERS', [128, 256]))
        if layer_nums != [5, 5] or layer_strides != [1, 2]:
            raise ValueError('Native ISFusion currently expects SECOND [5,5]/[1,2]')

        blocks = []
        in_channels = [input_channels, out_channels[0]]
        for stage in range(2):
            layers = []
            if layer_strides[stage] != 2:
                layers.extend(_conv_bn_relu(
                    in_channels[stage], out_channels[stage],
                    stride=layer_strides[stage],
                ))
            for _ in range(layer_nums[stage]):
                layers.extend(_conv_bn_relu(
                    out_channels[stage], out_channels[stage]
                ))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)
        self.ds_layer = nn.Sequential(*_conv_bn_relu(
            out_channels[0], out_channels[1], stride=2
        ))
        self.num_bev_features = out_channels[-1]

    @staticmethod
    def _dense_coordinates(height, width, batch_size, device):
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij',
        )
        coordinates = []
        for batch_index in range(batch_size):
            batch = torch.full_like(y, batch_index)
            z = torch.zeros_like(y)
            coordinates.append(torch.stack((batch, z, y, x), dim=-1).view(-1, 4))
        return torch.cat(coordinates, dim=0).long()

    def forward_stage1(self, features):
        dense = self.blocks[0](features)
        downsampled = self.ds_layer(dense)
        batch_size, channels, height, width = downsampled.shape
        flat = downsampled.flatten(2).transpose(1, 2).reshape(-1, channels)
        coordinates = self._dense_coordinates(
            height, width, batch_size, downsampled.device
        )
        return flat, coordinates, dense

    def forward_stage2(self, features):
        return self.blocks[1](features)

    def forward(self, features, stage=None):
        if stage == 'stage1':
            return self.forward_stage1(features[0])
        if stage == 'stage2':
            return None, None, self.forward_stage2(features[0])
        stage1 = self.blocks[0](features)
        stage2 = self.blocks[1](self.ds_layer(stage1))
        return stage1, stage2


class ISFusionSECONDFPN(nn.Module):
    """Two-level FPN matching CCF's ``SECONDFPNV2``."""

    def __init__(self, model_cfg=None):
        super().__init__()
        if model_cfg is None:
            in_channels = [128, 256]
            out_channels = [256, 256]
            strides = [1, 2]
            use_conv_for_no_stride = True
        else:
            in_channels = list(model_cfg.get('IN_CHANNELS', [128, 256]))
            out_channels = list(model_cfg.get('NUM_UPSAMPLE_FILTERS', [256, 256]))
            strides = list(model_cfg.get('UPSAMPLE_STRIDES', [1, 2]))
            use_conv_for_no_stride = model_cfg.get('USE_CONV_FOR_NO_STRIDE', True)
        deblocks = []
        for in_channel, out_channel, stride in zip(
                in_channels, out_channels, strides):
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                layer = nn.ConvTranspose2d(
                    in_channel, out_channel, kernel_size=stride,
                    stride=stride, bias=False,
                )
            else:
                inverse_stride = round(1 / stride)
                layer = nn.Conv2d(
                    in_channel, out_channel, kernel_size=inverse_stride,
                    stride=inverse_stride, bias=False,
                )
            deblocks.append(nn.Sequential(
                layer,
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ))
        self.deblocks = nn.ModuleList(deblocks)
        self.num_bev_features = sum(out_channels)

    def forward_features(self, features):
        upsampled = [
            deblock(feature)
            for deblock, feature in zip(self.deblocks, features)
        ]
        output = torch.cat(upsampled, dim=1) if len(upsampled) > 1 else upsampled[0]
        # Preserve the original CCF convention consumed by TransFusionHeadV2.
        return output.permute(0, 1, 3, 2).contiguous()

    def forward(self, batch_dict):
        features = batch_dict.get('isfusion_multi_scale_bev')
        if features is None:
            raise KeyError('batch_dict is missing isfusion_multi_scale_bev')
        batch_dict['spatial_features_2d'] = self.forward_features(features)
        return batch_dict
