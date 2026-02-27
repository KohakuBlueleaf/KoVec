from collections import Counter

import cv2
import numpy as np
import torch

from kovec.vector.types import VectorScene


def fit_colors(
    scene: VectorScene,
    target_img: np.ndarray,
    layered_masks: list[list[np.ndarray]],
    cluster: bool = True,
    k: int = 30,
) -> tuple[VectorScene, np.ndarray]:
    """Assign dominant colors from *target_img* to each path in *scene*.

    If *cluster* is True, k-means quantises the target image first.
    Returns ``(scene, clustered_image)``.
    """
    if cluster:
        data = target_img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        target_img = centers[labels.flatten()].reshape(target_img.shape)

    idx = 0
    for i, masks in enumerate(layered_masks):
        # Mask of the layer above (to subtract occlusion)
        if i <= len(layered_masks) - 2:
            above = layered_masks[i + 1][0].copy()
            for m in layered_masks[i + 1][1:]:
                above = cv2.add(above, m)
        else:
            above = np.zeros_like(masks[0])

        for mask in masks:
            visible = mask.astype(np.int16) - above.astype(np.int16)
            visible = np.clip(visible, 0, 255).astype(np.uint8)
            pixels = target_img[visible == 255]
            color_tuples = [tuple(c) for c in pixels]
            counts = Counter(color_tuples)
            if counts:
                dominant = list(counts.most_common(1)[0][0])
            else:
                dominant = [0, 0, 0]

            dev = scene.groups[idx].fill_color.device
            scene.groups[idx].fill_color = (
                torch.tensor(dominant + [255], device=dev, dtype=torch.float32) / 255.0
            )
            idx += 1

    return scene, target_img
