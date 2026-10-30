"""Trainable, framework-native OpenPCDet implementation of ISFusion."""

from __future__ import annotations

from pathlib import Path

import torch

from .detector3d_template import Detector3DTemplate
from ..backbones_2d.isfusion_bev_backbone import (
    ISFusionSECOND, ISFusionSECONDFPN,
)
from ..backbones_3d.isfusion_sparse_encoder import ISFusionSparseEncoder
from ..backbones_3d.vfe.isfusion_dynamic_vfe import ISFusionDynamicVFE
from ..backbones_image.swin import SwinTransformer
from ..backbones_image.img_neck.generalized_lss import GeneralizedLSSFPN
from ..dense_heads.isfusion_transfusion_head import ISFusionTransFusionHead
from ..fusion_layers.isfusion_pillar_image_sampler import ISFusionPillarImageSampler
from ..fusion_layers.isfusion_sst import ISFusionFusionEncoder


class ISFusionNative(Detector3DTemplate):
    """Pure OpenPCDet ISFusion with native training and inference contracts."""

    CCF_PREFIX = 'pts_backbone.'

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        grid_size = dataset.grid_size
        point_range = dataset.point_cloud_range
        voxel_size = dataset.voxel_size

        # Keep CCF module names so the published 642-tensor checkpoint is a
        # strict numerical oracle for every trainable component.
        self.pts_voxel_encoder = ISFusionDynamicVFE(
            model_cfg.VFE, dataset.point_feature_encoder.num_point_features,
            voxel_size, grid_size, point_range,
        )
        self.pts_middle_encoder = ISFusionSparseEncoder(
            model_cfg.BACKBONE_3D,
            input_channels=self.pts_voxel_encoder.get_output_feature_dim(),
            grid_size=grid_size,
        )
        self.img_backbone = SwinTransformer(model_cfg.IMAGE_BACKBONE)
        self.img_backbone.init_weights()
        self.img_neck = GeneralizedLSSFPN(model_cfg.IMAGE_NECK)
        self.freeze_image_backbone = bool(
            model_cfg.get('FREEZE_IMAGE_BACKBONE', False)
        )
        if self.freeze_image_backbone:
            for parameter in self.img_backbone.parameters():
                parameter.requires_grad = False
        self.pillar_sampler = ISFusionPillarImageSampler(model_cfg.PILLAR_SAMPLER)
        self.fusion_encoder = ISFusionFusionEncoder(model_cfg.FUSION_ENCODER)
        self.pts_backbone = ISFusionSECOND(model_cfg.BACKBONE_2D, input_channels=128)
        self.pts_neck = ISFusionSECONDFPN(model_cfg.BEV_NECK)
        self.pts_bbox_head = ISFusionTransFusionHead(
            model_cfg=model_cfg.DENSE_HEAD,
            input_channels=512,
            num_class=num_class,
            class_names=dataset.class_names,
            grid_size=grid_size,
            point_cloud_range=point_range,
            voxel_size=voxel_size,
        )

    def forward(self, batch_dict):
        batch_dict = self.pts_voxel_encoder(batch_dict)
        batch_dict = self.pts_middle_encoder(batch_dict)
        if 'img_mask_idx' in batch_dict:
            # Original IS-Fusion applies ModalMask after normalization and
            # immediately before the image backbone.
            images = batch_dict['camera_imgs']
            for batch_index, indices in enumerate(batch_dict['img_mask_idx']):
                indices = indices[indices >= 0].long()
                if indices.numel():
                    images[batch_index, indices] = 0.0
        if self.freeze_image_backbone:
            # The Source-only Swin is a fixed image feature extractor.  Keep
            # the GeneralizedLSSFPN trainable because its structure does not
            # match the Faster R-CNN FPN checkpoint.
            with torch.no_grad():
                batch_dict = self.img_backbone(batch_dict)
        else:
            batch_dict = self.img_backbone(batch_dict)
        batch_dict = self.img_neck(batch_dict)
        batch_dict = self.pillar_sampler(batch_dict)
        batch_dict = self.fusion_encoder(batch_dict, self.pts_backbone)
        batch_dict = self.pts_neck(batch_dict)
        batch_dict = self.pts_bbox_head(batch_dict)
        if self.training:
            loss = batch_dict['loss']
            metrics = {'loss_trans': float(loss.detach()), **batch_dict['tb_dict']}
            return {'loss': loss}, metrics, {}
        return self.post_processing_native(batch_dict)

    def post_processing_native(self, batch_dict):
        predictions = batch_dict['final_box_dicts']
        recall = {}
        for index, prediction in enumerate(predictions):
            recall = self.generate_recall_record(
                box_preds=prediction['pred_boxes'], recall_dict=recall,
                batch_index=index, data_dict=batch_dict,
                thresh_list=self.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST,
            )
        return predictions, recall

    def load_params_from_file(self, filename, logger, to_cpu=False,
                              pre_trained_path=None):
        """Load either native OpenPCDet weights or the released CCF checkpoint."""
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict) or not state:
            return super().load_params_from_file(
                filename, logger, to_cpu=to_cpu,
                pre_trained_path=pre_trained_path,
            )
        if not all(key.startswith(self.CCF_PREFIX) for key in state):
            return super().load_params_from_file(
                filename, logger, to_cpu=to_cpu,
                pre_trained_path=pre_trained_path,
            )
        converted = {
            key[len(self.CCF_PREFIX):]: value for key, value in state.items()
        }
        expected = self.state_dict()
        model_keys = set(expected) - {'global_step'}
        missing = sorted(model_keys - set(converted))
        unexpected = sorted(set(converted) - model_keys)
        mismatched = sorted(
            key for key in model_keys & set(converted)
            if expected[key].shape != converted[key].shape
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                'ISFusionNative strict CCF load failed: '
                f'missing={len(missing)}, unexpected={len(unexpected)}, '
                f'shape_mismatch={len(mismatched)}'
            )
        self.load_state_dict(converted, strict=False)
        logger.info(
            '==> ISFusionNative strict CCF load OK: %d/%d tensors from %s',
            len(converted), len(model_keys), str(Path(filename).resolve()),
        )
