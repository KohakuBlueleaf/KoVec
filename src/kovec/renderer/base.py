from abc import ABC, abstractmethod

from torch import Tensor

from kovec.vector.types import VectorScene


class VectorRenderer(ABC):
    """Abstract renderer that can differentiably rasterize a VectorScene."""

    @abstractmethod
    def render(self, scene: VectorScene, width: int, height: int) -> Tensor:
        """Render *scene* → (H, W, 4) RGBA float tensor."""
        ...

    @abstractmethod
    def save_svg(self, scene: VectorScene, path: str, width: int, height: int) -> None:
        """Write *scene* to an SVG file at *path*."""
        ...
