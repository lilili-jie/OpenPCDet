"""OpenPCDet detector adapter for CCF's released ISFusion model.

The CCF checkpoint stores a complete ISFusion detector below the
``pts_backbone.`` prefix.  This class keeps that detector's proven numerical
implementation intact while adapting OpenPCDet's dataset/checkpoint/output
contracts.  It is deliberately inference-only: the immediate goal is strict
cross-framework reproduction of the released Source-only checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import torch

from .detector3d_template import Detector3DTemplate


class ISFusion(Detector3DTemplate):
    """Run the released CCF ISFusion detector through OpenPCDet."""

    CHECKPOINT_PREFIX = "pts_backbone."

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if num_class != 10:
            raise ValueError(f"ISFusion expects 10 nuScenes classes, got {num_class}")

        self.ccf_root = Path(__file__).resolve().parents[5]
        if str(self.ccf_root) not in sys.path:
            sys.path.insert(0, str(self.ccf_root))

        # Importing the plugins registers all custom CCF modules required by
        # the nested ISFusion config.
        import projects.isfusion  # noqa: F401
        import projects.mmdet3d_plugin  # noqa: F401
        from mmcv import Config
        from mmdet3d.models import build_model

        config_path = self.ccf_root / model_cfg.get(
            "CCF_CONFIG", "projects/configs/ccf/baseline_source.py"
        )
        config = Config.fromfile(str(config_path))
        reference_cfg = copy.deepcopy(config.model.pts_backbone)
        reference_cfg.pop("init_cfg", None)
        reference_cfg["freeze"] = False
        reference_cfg["norm_eval"] = False
        self.reference_model = build_model(reference_cfg)

        self.loaded_checkpoint = None
        self.loaded_checkpoint_sha256 = None
        self.loaded_tensor_count = 0

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        """Strictly load either the original or prefix-stripped CCF checkpoint."""
        if pre_trained_path is not None:
            raise ValueError("ISFusion reference reproduction accepts one checkpoint only")

        checkpoint_path = Path(filename).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise TypeError("Expected a CCF checkpoint containing 'state_dict'")

        source = checkpoint["state_dict"]
        if not isinstance(source, dict):
            raise TypeError("checkpoint['state_dict'] must be a mapping")

        prefixed = [key.startswith(self.CHECKPOINT_PREFIX) for key in source]
        if all(prefixed):
            converted = {
                key[len(self.CHECKPOINT_PREFIX):]: value for key, value in source.items()
            }
        elif not any(prefixed):
            converted = dict(source)
        else:
            raise RuntimeError("Checkpoint mixes prefixed and unprefixed ISFusion keys")

        expected = self.reference_model.state_dict()
        missing = sorted(set(expected) - set(converted))
        unexpected = sorted(set(converted) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(converted)
            if tuple(expected[key].shape) != tuple(converted[key].shape)
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "ISFusion strict load failed: "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"shape_mismatch={len(mismatched)}"
            )

        self.reference_model.load_state_dict(converted, strict=True)
        self.loaded_checkpoint = str(checkpoint_path)
        self.loaded_checkpoint_sha256 = self._sha256(checkpoint_path)
        self.loaded_tensor_count = len(converted)
        logger.info(
            "==> ISFusion strict load OK: %d/%d tensors, SHA256=%s",
            len(converted), len(expected), self.loaded_checkpoint_sha256,
        )

    @staticmethod
    def _split_points(points: torch.Tensor, batch_size: int):
        return [
            points[points[:, 0].long() == batch_idx, 1:].contiguous()
            for batch_idx in range(batch_size)
        ]

    @staticmethod
    def _build_img_metas(camera_imgs: torch.Tensor, batch_size: int):
        from mmdet3d.core.bbox import LiDARInstance3DBoxes

        input_shape = tuple(int(value) for value in camera_imgs.shape[-2:])
        return [
            {
                "input_shape": input_shape,
                "img_shape": input_shape,
                "pad_shape": input_shape,
                "box_type_3d": LiDARInstance3DBoxes,
            }
            for _ in range(batch_size)
        ]

    @staticmethod
    def _compose_ccf_lidar2image(lidar2image: torch.Tensor,
                                 img_aug_matrix: torch.Tensor) -> torch.Tensor:
        """Fold OpenPCDet image augmentation into projection like CCF does.

        OpenPCDet applies resize/crop after perspective division, storing the
        2-D translation in ``img_aug_matrix[..., :3, 3]``.  CCF instead folds
        the same affine operation into the camera intrinsics before forming
        ``lidar2img``.  Converting to CCF's representation avoids a different
        floating-point operation order during strict checkpoint comparison.
        """
        if lidar2image.ndim != 4 or img_aug_matrix.ndim != 4:
            raise ValueError(
                "Expected batched [B, V, 4, 4] calibration matrices, got "
                f"{tuple(lidar2image.shape)} and {tuple(img_aug_matrix.shape)}"
            )

        image_homography = img_aug_matrix[..., :3, :3].clone()
        image_homography[..., :, 2] += img_aug_matrix[..., :3, 3]

        composed = lidar2image.clone()
        composed[..., :3, :] = torch.matmul(
            image_homography.to(lidar2image.dtype), lidar2image[..., :3, :]
        )
        return composed

    def forward(self, batch_dict):
        if self.training:
            raise NotImplementedError(
                "The strict reference port is inference-only; training is outside this reproduction."
            )
        if self.loaded_tensor_count == 0:
            raise RuntimeError("ISFusion checkpoint has not been loaded")

        required = (
            "points", "camera_imgs", "lidar2image", "img_aug_matrix",
            "lidar_aug_matrix", "batch_size",
        )
        missing = [key for key in required if key not in batch_dict]
        if missing:
            raise KeyError(f"ISFusion batch is missing fields: {missing}")

        batch_size = int(batch_dict["batch_size"])
        points = self._split_points(batch_dict["points"], batch_size)
        images = batch_dict["camera_imgs"]
        img_metas = self._build_img_metas(images, batch_size)
        lidar2image = self._compose_ccf_lidar2image(
            batch_dict["lidar2image"], batch_dict["img_aug_matrix"]
        )

        output = self.reference_model.simple_test(
            points=points,
            img_metas=img_metas,
            img=images,
            rescale=False,
            lidar2img=lidar2image,
            img_aug_matrix=None,
            lidar_aug_matrix=batch_dict["lidar_aug_matrix"],
        )
        if output is None or len(output.get("query_bboxes", [])) != batch_size:
            raise RuntimeError("ISFusion returned an invalid prediction batch")

        pred_dicts = []
        # nuScenes mAP/NDS evaluation only consumes ``pred_dicts``.  Do not
        # run OpenPCDet's optional CUDA recall helper here: the released CCF
        # boxes are wrapped by MMDetection3D and that auxiliary helper is not
        # part of the reference ISFusion metric path.
        recall_dict = {}
        for index, (boxes, scores, labels) in enumerate(output["query_bboxes"]):
            pred_boxes = boxes.tensor
            pred_dicts.append({
                "pred_boxes": pred_boxes,
                "pred_scores": scores,
                # OpenPCDet labels are one-based; MMDetection3D labels are zero-based.
                "pred_labels": labels.long() + 1,
            })

        return pred_dicts, recall_dict
