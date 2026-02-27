from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict


class SegmentationResult(BaseModel):
    """Output of a segmentor: list of binary masks + optional confidence scores."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    masks: list[np.ndarray]  # each (H, W) uint8, 0/255
    scores: list[float] = []


class Segmentor(ABC):
    """Abstract interface for image segmentation backends."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment *image* (H, W, 3) uint8 RGB → SegmentationResult."""
        ...
