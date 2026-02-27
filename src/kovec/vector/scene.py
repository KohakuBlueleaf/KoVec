import numpy as np
import torch

from kovec.vector.contour import mask_to_path
from kovec.vector.types import VectorPathGroup, VectorScene


def _get_mean_color(image: np.ndarray, mask: np.ndarray) -> tuple[int, ...]:
    """Average RGB color of pixels under a binary mask."""
    pixels = image[mask > 0]
    if len(pixels) > 0:
        return tuple(np.mean(pixels, axis=0, dtype=int))
    return (0, 0, 0)


def build_scene_from_masks(
    layered_masks: list[list[np.ndarray]],
    target_img: np.ndarray,
    epsilon: float = 5.0,
) -> VectorScene:
    """Build a VectorScene from hierarchical layer masks + a target image.

    *layered_masks* is ``list[list[mask]]`` — outer list = layers (back→front),
    inner list = masks within that layer.
    """
    scene = VectorScene()

    for masks in layered_masks:
        for mask in masks:
            path = mask_to_path(mask, epsilon=epsilon)
            color = _get_mean_color(target_img, mask)
            group = VectorPathGroup(
                shape_idx=len(scene.paths),
                fill_color=torch.tensor(list(color) + [255], dtype=torch.float32)
                / 255.0,
                stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            )
            scene.paths.append(path)
            scene.groups.append(group)

    return scene
