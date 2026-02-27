import cv2
import numpy as np


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask via flood-fill from (0,0)."""
    flood = mask.copy()
    h, w = mask.shape[:2]
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def preprocess_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Fill holes + split connected components into separate masks, sorted by area."""
    result: list[np.ndarray] = []
    for mask in masks:
        filled = fill_holes(mask)
        n_labels, labels = cv2.connectedComponents(filled)
        for i in range(1, n_labels):
            region = np.zeros_like(mask)
            region[labels == i] = 255
            result.append(region)
    return _sort_by_area(result)


def _sort_by_area(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Sort masks by pixel area, largest first."""
    return sorted(masks, key=lambda m: cv2.countNonZero(m), reverse=True)


def layer_masks(
    layered: list[list[np.ndarray]],
    unlayered: list[np.ndarray],
) -> list[list[np.ndarray]]:
    """Assign *unlayered* masks into hierarchical *layered* structure.

    Each unlayered mask is placed into the deepest layer where it overlaps
    significantly with an existing mask.  Logic mirrors the reference
    ``layer_segmented_masks`` implementation.
    """
    remaining = list(unlayered)

    while remaining:
        mask = remaining.pop(0)
        placed = False

        for layer_i, layer in enumerate(layered):
            if not layer:
                continue
            for mask_i, existing in enumerate(layer):
                area_existing = cv2.countNonZero(existing)
                area_new = cv2.countNonZero(mask)
                intersection = int(
                    np.sum(
                        (existing.astype(np.float32) + mask.astype(np.float32)) == 510
                    )
                )
                union = cv2.countNonZero(cv2.add(existing, mask))

                if union > 0 and intersection / union > 0.70:
                    placed = True
                    break

                if (
                    area_existing > 0
                    and intersection / area_existing >= 0.5
                    and area_new > area_existing
                ):
                    placed = True
                    break

                if (
                    area_new > 0
                    and intersection / area_new >= 0.5
                    and area_new < area_existing
                ):
                    if layer_i == len(layered) - 1:
                        layered.append([mask])
                        placed = True
                    break

                if mask_i == len(layer) - 1:
                    layer.append(mask)
                    placed = True
                    break

            if placed:
                break

    return layered


def filter_by_area(
    layered_masks: list[list[np.ndarray]], n: int
) -> list[list[np.ndarray]]:
    """Keep at most *n* masks total, selecting the largest by pixel area."""
    total = sum(len(layer) for layer in layered_masks)
    if total <= n:
        return layered_masks

    # Collect (area, layer_i, mask_i) and pick top-n
    elements: list[tuple[float, int, int]] = []
    for i, layer in enumerate(layered_masks):
        for j, mask in enumerate(layer):
            elements.append((np.sum(mask.astype(np.float32)), i, j))
    elements.sort(key=lambda x: -x[0])
    keep = {(e[1], e[2]) for e in elements[:n]}

    result: list[list[np.ndarray]] = []
    for i, layer in enumerate(layered_masks):
        filtered = [m for j, m in enumerate(layer) if (i, j) in keep]
        if filtered:
            result.append(filtered)
    return result


def segment_image_sequence(
    segmentor: "Segmentor",
    images: list[np.ndarray],
) -> list[np.ndarray]:
    """Run segmentor on each image in sequence, reverse, and flatten.

    Returns all masks ordered from simplest (last image) to most detailed (first).
    """
    all_masks: list[list[np.ndarray]] = []
    for img in images:
        result = segmentor.segment(img)
        processed = preprocess_masks(result.masks)
        all_masks.append(processed)

    all_masks.reverse()
    return [m for sublist in all_masks for m in sublist]
