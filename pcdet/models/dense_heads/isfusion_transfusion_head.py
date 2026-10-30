"""Pure PyTorch forward path of CCF's TransFusionHeadV2.

The parameter hierarchy is intentionally checkpoint-compatible. Training
target assignment and box decoding are integrated separately with OpenPCDet;
this module first establishes exact, differentiable network predictions.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvModule1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.activate(self.bn(self.conv(features)))


class ConvModule2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.activate(self.bn(self.conv(features)))


class MFSeparateHead(nn.Module):
    def __init__(self, in_channels, heads, head_conv=64):
        super().__init__()
        self.heads = heads
        for name, (out_channels, num_conv) in heads.items():
            layers = []
            current_channels = in_channels
            for _ in range(num_conv - 1):
                layers.append(ConvModule1d(current_channels, head_conv))
                current_channels = head_conv
            layers.append(nn.Conv1d(head_conv, out_channels, 1, bias=True))
            setattr(self, name, nn.Sequential(*layers))

    def forward(self, box_features, class_features=None):
        output = {}
        for name in self.heads:
            source = class_features if name == 'heatmap' else box_features
            output[name] = getattr(self, name)(source)
        return output


class PositionEncodingLearned(nn.Module):
    def __init__(self, input_channel=2, num_pos_feats=128):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, 1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, 1),
        )

    def forward(self, coordinates):
        return self.position_embedding_head(coordinates.transpose(1, 2).contiguous())


class ResidualMultiheadAttention(nn.Module):
    """MMCV MultiheadAttention-compatible parameter and residual behavior."""

    def __init__(self, embed_dims=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=False
        )
        self.proj_drop = nn.Dropout(0.0)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_pos=None, attn_mask=None,
                key_padding_mask=None):
        key = query if key is None else key
        value = key if value is None else value
        identity = query if identity is None else identity
        if key_pos is None and query_pos is not None and query_pos.shape == key.shape:
            key_pos = query_pos
        query_with_pos = query if query_pos is None else query + query_pos
        key_with_pos = key if key_pos is None else key + key_pos
        output, _ = self.attn(
            query_with_pos.transpose(0, 1),
            key_with_pos.transpose(0, 1),
            value.transpose(0, 1),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        output = output.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(output))


class FeedForward(nn.Module):
    """MMCV FFN-compatible two-layer residual MLP."""

    def __init__(self, embed_dims=128, feedforward_channels=256, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(dropout),
        )
        self.dropout_layer = nn.Identity()

    def forward(self, features, identity=None):
        identity = features if identity is None else identity
        return identity + self.dropout_layer(self.layers(features))


class CenterTransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ResidualMultiheadAttention()
        self.cross_attn = ResidualMultiheadAttention()
        self.ffn = FeedForward()
        self.norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(3)])
        self.self_posembed = PositionEncodingLearned()
        self.cross_posembed = PositionEncodingLearned()

    def forward(self, query, key, query_pos, key_pos,
                self_attn_mask=None, cross_attn_mask=None,
                key_padding_mask=None):
        query_position = self.self_posembed(query_pos).transpose(1, 2)
        key_position = self.cross_posembed(key_pos).transpose(1, 2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        query = self.self_attn(
            query, query, query + query_position,
            query_pos=query_position, key_pos=query_position,
            attn_mask=self_attn_mask,
        )
        query = self.norms[0](query)
        query = self.cross_attn(
            query, key, key + key_position,
            query_pos=query_position, key_pos=key_position,
            attn_mask=cross_attn_mask, key_padding_mask=key_padding_mask,
        )
        query = self.norms[1](query)
        query = self.ffn(query)
        return self.norms[2](query).transpose(1, 2)


class MFTransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ResidualMultiheadAttention()
        self.cls_cross_attn = ResidualMultiheadAttention()
        self.box_cross_attn = ResidualMultiheadAttention()
        self.cls_ffn = FeedForward()
        self.box_ffn = FeedForward()
        self.norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(5)])
        self.self_posembed = PositionEncodingLearned()
        self.box_cross_posembed = PositionEncodingLearned()
        self.cls_cross_posembed = PositionEncodingLearned()

    def forward(self, query, query_pos, box_key, cls_key,
                box_key_pos, cls_key_pos, self_attn_mask=None,
                box_cross_attn_mask=None, cls_cross_attn_mask=None,
                box_key_padding_mask=None, cls_key_padding_mask=None):
        query_position = self.self_posembed(query_pos).transpose(1, 2)
        box_position = self.box_cross_posembed(box_key_pos).transpose(1, 2)
        class_position = self.cls_cross_posembed(cls_key_pos).transpose(1, 2)
        query = query.transpose(1, 2)
        box_key = box_key.transpose(1, 2)
        cls_key = cls_key.transpose(1, 2)
        query = self.self_attn(
            query, query, query, query_pos=query_position,
            key_pos=query_position, attn_mask=self_attn_mask,
        )
        query = self.norms[0](query)
        box_query = self.box_cross_attn(
            query, box_key, box_key, query_pos=query_position,
            key_pos=box_position, attn_mask=box_cross_attn_mask,
            key_padding_mask=box_key_padding_mask,
        )
        box_query = self.norms[1](box_query)
        box_query = self.norms[2](self.box_ffn(box_query))
        class_query = self.cls_cross_attn(
            query, cls_key, cls_key, query_pos=query_position,
            key_pos=class_position, attn_mask=cls_cross_attn_mask,
            key_padding_mask=cls_key_padding_mask,
        )
        class_query = self.norms[3](class_query)
        class_query = self.norms[4](self.cls_ffn(class_query))
        return box_query.transpose(1, 2), class_query.transpose(1, 2)


class ISFusionTransFusionHead(nn.Module):
    """Checkpoint-compatible neural prediction path of TransFusionHeadV2."""

    def __init__(self, model_cfg=None, input_channels=512, num_class=10,
                 class_names=None, grid_size=None, point_cloud_range=None,
                 voxel_size=None, predict_boxes_when_training=True):
        super().__init__()
        self.num_classes = num_class
        self.num_proposals = 200
        self.num_decoder_layers = 1
        self.nms_kernel_size = 3
        self.auxiliary = True
        self.shared_conv = nn.Conv2d(input_channels, 128, 3, padding=1, bias=True)
        self.heatmap_head = nn.Sequential(
            ConvModule2d(128, 128, 3),
            nn.Conv2d(128, num_class, 3, padding=1, bias=True),
        )
        self.class_encoding = nn.Conv1d(num_class, 128, 1)
        self.center_decoder = CenterTransformerDecoderLayer()
        self.center_head = MFSeparateHead(128, {'center': (2, 2)})
        self.decoder = nn.ModuleList([MFTransformerDecoderLayer()])
        heads = {
            'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2),
            'vel': (2, 2), 'heatmap': (num_class, 2),
        }
        self.prediction_heads = nn.ModuleList([MFSeparateHead(128, heads)])
        # Match TransFusionHeadV2.init_weights(): only the main decoupled
        # decoder is explicitly Xavier-initialized in the CCF implementation.
        for parameter in self.decoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        self.bev_pos = self.create_2d_grid(180, 180)
        self.query_labels = None
        self.model_cfg = model_cfg
        if model_cfg is not None:
            from .target_assigner.hungarian_assigner import HungarianAssigner3D
            from ...utils import loss_utils

            self.grid_size = np.asarray(grid_size)
            self.point_cloud_range = point_cloud_range
            self.voxel_size = voxel_size
            target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
            self.feature_map_stride = target_cfg.FEATURE_MAP_STRIDE
            self.dataset_name = target_cfg.get('DATASET', 'nuScenes')
            self.code_size = 10
            loss_cfg = model_cfg.LOSS_CONFIG
            self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(
                gamma=loss_cfg.LOSS_CLS.gamma,
                alpha=loss_cfg.LOSS_CLS.alpha,
            )
            self.loss_cls_weight = loss_cfg.LOSS_WEIGHTS.cls_weight
            self.loss_bbox = loss_utils.L1Loss()
            self.loss_bbox_weight = loss_cfg.LOSS_WEIGHTS.bbox_weight
            self.loss_heatmap = loss_utils.GaussianFocalLoss()
            self.loss_heatmap_weight = loss_cfg.LOSS_WEIGHTS.hm_weight
            self.bbox_assigner = HungarianAssigner3D(
                **target_cfg.HUNGARIAN_ASSIGNER
            )

    @staticmethod
    def create_2d_grid(x_size, y_size):
        batch_x, batch_y = torch.meshgrid(
            torch.linspace(0, x_size - 1, x_size),
            torch.linspace(0, y_size - 1, y_size),
            indexing='ij',
        )
        coordinates = torch.cat(
            ((batch_x + 0.5)[None], (batch_y + 0.5)[None]), dim=0
        )[None]
        return coordinates.view(1, 2, -1).permute(0, 2, 1)

    def predict(self, inputs):
        batch_size = inputs.shape[0]
        lidar_features = self.shared_conv(inputs)
        flat_features = lidar_features.view(batch_size, 128, -1)
        bev_position = self.bev_pos.repeat(batch_size, 1, 1).to(inputs.device)
        dense_heatmap = self.heatmap_head(lidar_features)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        pooled = F.max_pool2d(heatmap, self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:-padding, padding:-padding] = pooled
        local_max[:, 8] = heatmap[:, 8]
        local_max[:, 9] = heatmap[:, 9]
        heatmap = (heatmap * (heatmap == local_max)).view(batch_size, self.num_classes, -1)
        proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True
        )[..., :self.num_proposals]
        proposal_classes = proposals // heatmap.shape[-1]
        proposal_indices = proposals % heatmap.shape[-1]
        query_position = bev_position.gather(
            1, proposal_indices[:, :, None].expand(-1, -1, 2)
        )
        query_features = flat_features.gather(
            2, proposal_indices[:, None, :].expand(-1, 128, -1)
        )
        self.query_labels = proposal_classes
        one_hot = F.one_hot(
            proposal_classes, num_classes=self.num_classes
        ).permute(0, 2, 1)
        query_features = query_features + self.class_encoding(one_hot.float())
        query_features = self.center_decoder(
            query_features, flat_features, query_position, bev_position
        )
        center = self.center_head(query_features)['center']
        query_position = query_position + center.permute(0, 2, 1)

        outputs = []
        for decoder, prediction_head in zip(self.decoder, self.prediction_heads):
            box_features, class_features = decoder(
                query_features, query_position, flat_features, flat_features,
                bev_position, bev_position,
            )
            result = prediction_head(box_features, class_features)
            result['center'] = query_position.permute(0, 2, 1)
            outputs.append(result)
            query_position = result['center'].detach().clone().permute(0, 2, 1)
        result = outputs[0]
        result['query_heatmap_score'] = heatmap.gather(
            2, proposal_indices[:, None, :].expand(-1, self.num_classes, -1)
        )
        result['dense_heatmap'] = dense_heatmap
        result['lidar_feats'] = lidar_features
        result['query_cat'] = proposal_classes
        result['query_feats'] = query_features
        return result

    def forward(self, batch_dict):
        # The reference IS-Fusion recipe is full FP32 (its config does not
        # enable MMCV's fp16 hook).  We use AMP for the much larger feature
        # extractors so a real per-GPU batch of four fits on 24 GB GPUs, but
        # keep the relatively small TransFusion head in FP32.  Running this
        # head in FP16 produced non-finite predictions near the peak OneCycle
        # learning rate, which subsequently made Hungarian matching fail.
        with torch.autocast(device_type='cuda', enabled=False):
            predictions = self.predict(
                batch_dict['spatial_features_2d'].float()
            )
        batch_dict['isfusion_head_predictions'] = predictions
        if self.model_cfg is None:
            return batch_dict
        if self.training:
            ground_truth = batch_dict['gt_boxes']
            boxes = ground_truth[..., :-1]
            labels = ground_truth[..., -1].long() - 1
            loss, metrics = self.loss(boxes, labels, predictions)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = metrics
        else:
            batch_dict['final_box_dicts'] = self.get_bboxes(predictions)
        return batch_dict

    # Reuse OpenPCDet's native target assignment, loss and nuScenes decoding.
    # These functions contain no trainable layers; the neural architecture
    # above remains CCF's TransFusionHeadV2 rather than standard TransFusion.
    def get_targets(self, *args, **kwargs):
        from .transfusion_head import TransFusionHead
        return TransFusionHead.get_targets(self, *args, **kwargs)

    def get_targets_single(self, *args, **kwargs):
        from .transfusion_head import TransFusionHead
        return TransFusionHead.get_targets_single(self, *args, **kwargs)

    def loss(self, gt_bboxes_3d, gt_labels_3d, predictions, **kwargs):
        # The original MMDetection3D IS-Fusion head decorates ``loss`` with
        # ``@force_fp32(apply_to=('preds_dicts'))``.  OpenPCDet wraps the
        # complete forward/loss call in autocast, so omitting that boundary
        # makes focal-cost logarithms and decoded box exponentials run in
        # FP16.  Saturated FP16 probabilities can then produce inf/NaN in the
        # Hungarian cost matrix.  Preserve AMP for the neural forward pass,
        # but match the original implementation by computing assignment and
        # losses from FP32 predictions.
        predictions_fp32 = {
            key: value.float()
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for key, value in predictions.items()
        }
        gt_bboxes_3d = gt_bboxes_3d.float()
        from .transfusion_head import TransFusionHead
        with torch.autocast(device_type='cuda', enabled=False):
            return TransFusionHead.loss(
                self, gt_bboxes_3d, gt_labels_3d, predictions_fp32, **kwargs
            )

    def encode_bbox(self, bboxes):
        """Encode OpenPCDet GT boxes in CCF head-output convention.

        OpenPCDet stores boxes as gravity-centred ``(x, y, z, l, w, h,
        yaw)``.  The released CCF head predicts the footprint and yaw in the
        MMDetection3D/CCF convention; final decoding swaps ``l/w`` and applies
        ``yaw = -yaw - pi/2``.  Regression targets must apply the inverse
        (the yaw transform is self-inverse), otherwise training supervises a
        different representation from the one used at inference.
        """
        targets = bboxes.new_zeros((bboxes.shape[0], self.code_size))
        targets[:, 0] = (
            (bboxes[:, 0] - self.point_cloud_range[0])
            / (self.feature_map_stride * self.voxel_size[0])
        )
        targets[:, 1] = (
            (bboxes[:, 1] - self.point_cloud_range[1])
            / (self.feature_map_stride * self.voxel_size[1])
        )
        targets[:, 2] = bboxes[:, 2]
        targets[:, 3] = bboxes[:, 4].log()
        targets[:, 4] = bboxes[:, 3].log()
        targets[:, 5] = bboxes[:, 5].log()
        ccf_yaw = -bboxes[:, 6] - (torch.pi / 2)
        targets[:, 6] = torch.sin(ccf_yaw)
        targets[:, 7] = torch.cos(ccf_yaw)
        if self.code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:9]
        return targets

    def format_heatmap_center(self, center_int):
        """Reproduce CCF's explicit ``center_int[[1, 0]]`` layout fix."""
        return center_int[[1, 0]]

    @staticmethod
    def _convert_ccf_box_to_openpcdet(boxes, bottom_center):
        """Convert decoded CCF boxes to OpenPCDet's box convention in-place."""
        if boxes.numel() == 0:
            return boxes
        if bottom_center:
            boxes[:, 2] -= boxes[:, 5] * 0.5
        boxes[:, [3, 4]] = boxes[:, [4, 3]]
        boxes[:, 6] = -boxes[:, 6] - (torch.pi / 2)
        boxes[:, 6] = (
            torch.remainder(boxes[:, 6] + torch.pi, 2 * torch.pi) - torch.pi
        )
        return boxes

    def decode_bbox_for_assignment(self, heatmap, rot, dim, center, height, vel):
        """Decode predictions for matching against gravity-centred GT boxes."""
        from .transfusion_head import TransFusionHead
        decoded = TransFusionHead.decode_bbox(
            self, heatmap, rot, dim, center, height, vel, filter=False
        )
        for prediction in decoded:
            self._convert_ccf_box_to_openpcdet(
                prediction['pred_boxes'], bottom_center=False
            )
        return decoded

    def decode_bbox(self, *args, **kwargs):
        from .transfusion_head import TransFusionHead
        decoded = TransFusionHead.decode_bbox(self, *args, **kwargs)

        # ISFusion's released CCF head uses a different box convention from
        # OpenPCDet's stock TransFusion implementation.  The checkpoint's
        # three ``dim`` channels are (length, width, height), CCF exports a
        # bottom-centred z coordinate, and its LiDAR yaw convention is the
        # reflected/-pi/2 form below.  Reusing the stock decoder unchanged
        # silently produced correct xy centres and scores but swapped box
        # footprint / wrong heading (large mASE and mAOE on nuScenes).
        #
        # This full conversion is only for final output.  Hungarian matching
        # uses decode_bbox_for_assignment(), which intentionally keeps the
        # gravity-centred z convention used by OpenPCDet GT boxes.
        for prediction in decoded:
            self._convert_ccf_box_to_openpcdet(
                prediction['pred_boxes'], bottom_center=True
            )
        return decoded

    def get_bboxes(self, *args, **kwargs):
        from .transfusion_head import TransFusionHead
        return TransFusionHead.get_bboxes(self, *args, **kwargs)
