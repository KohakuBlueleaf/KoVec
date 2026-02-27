import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from kovec.config import SegmentationConfig
from kovec.segmentation.base import SegmentationResult, Segmentor
from kovec.segmentation.checkpoints import ensure_sam_checkpoint


class SAMSegmentor(Segmentor):
    """Segmentor backed by Meta's Segment Anything (SAM v1)."""

    def __init__(self, config: SegmentationConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        checkpoint = config.checkpoint
        if checkpoint is None:
            checkpoint = ensure_sam_checkpoint(config.model_type)

        sam = sam_model_registry[config.model_type](checkpoint=checkpoint)
        sam.to(device=device)

        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=config.points_per_side,
            pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            crop_n_layers=config.crop_n_layers,
            crop_n_points_downscale_factor=config.crop_n_points_downscale_factor,
            min_mask_region_area=config.min_mask_region_area,
            box_nms_thresh=config.box_nms_thresh,
        )

    def segment(self, image: np.ndarray) -> SegmentationResult:
        raw = self.generator.generate(image)

        # Start with a full-image background mask
        masks: list[np.ndarray] = [np.full(image.shape[:2], 255, dtype=np.uint8)]
        scores: list[float] = [1.0]

        for entry in raw:
            m = np.where(entry["segmentation"], 255, 0).astype(np.uint8)
            masks.append(m)
            scores.append(float(entry.get("predicted_iou", 0.0)))

        return SegmentationResult(masks=masks, scores=scores)
