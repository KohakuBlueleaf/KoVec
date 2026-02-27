import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from kovec.config import SegmentationConfig
from kovec.segmentation.base import SegmentationResult, Segmentor
from kovec.segmentation.checkpoints import ensure_sam2_checkpoint


class SAM2Segmentor(Segmentor):
    """Segmentor backed by SAM2."""

    def __init__(self, config: SegmentationConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        checkpoint = config.checkpoint
        if checkpoint is None:
            checkpoint = ensure_sam2_checkpoint(config.model_type)

        sam2 = build_sam2(config.model_type, checkpoint, device=str(device))
        self.generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=config.points_per_side,
            pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            min_mask_region_area=config.min_mask_region_area,
            box_nms_thresh=config.box_nms_thresh,
        )

    def segment(self, image: np.ndarray) -> SegmentationResult:
        raw = self.generator.generate(image)

        masks: list[np.ndarray] = [np.full(image.shape[:2], 255, dtype=np.uint8)]
        scores: list[float] = [1.0]

        for entry in raw:
            m = np.where(entry["segmentation"], 255, 0).astype(np.uint8)
            masks.append(m)
            scores.append(float(entry.get("predicted_iou", 0.0)))

        return SegmentationResult(masks=masks, scores=scores)
