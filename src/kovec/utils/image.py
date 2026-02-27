import numpy as np
import torch
from PIL import Image
from torch import Tensor


def load_image(path: str, resolution: int = 512, step: int = 64) -> np.ndarray:
    """Load image, scale longest side to *resolution*, snap dims to *step*.

    Returns (H, W, 3) uint8 RGB array where both H and W are multiples of
    *step*.  Aspect ratio is preserved (up to the snapping rounding).
    """
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size

    scale = resolution / max(orig_w, orig_h)
    new_w = max(step, round(orig_w * scale / step) * step)
    new_h = max(step, round(orig_h * scale / step) * step)

    return np.array(img.resize((new_w, new_h), Image.LANCZOS))


def rgba_to_rgb(
    img: Tensor,
    bg: Tensor | None = None,
) -> Tensor:
    """Composite RGBA (H, W, 4) over a solid background → (3, H, W).

    *bg* defaults to white ``[1, 1, 1]`` on the same device as *img*.
    """
    if bg is None:
        bg = torch.tensor([1.0, 1.0, 1.0], device=img.device)
    alpha = img[:, :, 3:4]
    rgb = alpha * img[:, :, :3] + bg * (1.0 - alpha)
    return rgb.permute(2, 0, 1)


def normalize(image: np.ndarray) -> Tensor:
    """uint8 (H, W, 3) → float (1, 3, H, W) in [-1, 1]."""
    t = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0)


def denormalize(latent_image: Tensor) -> np.ndarray:
    """float (1, 3, H, W) in [-1, 1] → uint8 (H, W, 3)."""
    img = (latent_image / 2.0 + 0.5).clamp(0, 1)
    img = img.cpu().permute(0, 2, 3, 1).numpy()
    return (img[0] * 255).astype(np.uint8)
