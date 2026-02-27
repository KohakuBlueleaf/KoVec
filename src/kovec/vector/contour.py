import random

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist

from kovec.vector.types import VectorPath


def _interpolate_segments(points: np.ndarray, num_interp: int = 2) -> np.ndarray:
    """Insert *num_interp* linearly-spaced control points between each pair."""
    new_points: list[np.ndarray] = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        new_points.append(p1)
        for t in np.linspace(0, 1, num_interp + 2)[1:-1]:
            new_points.append(p1 + t * (p2 - p1))
    new_points.append(points[-1])
    return np.array(new_points)


def _find_closest_contours(
    contours: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest pair of points across separate contours."""
    flat = [c[:, 0, :] for c in contours]
    min_dist = float("inf")
    best_p1 = flat[0][0]
    best_p2 = flat[1][0] if len(flat) > 1 else flat[0][0]

    for i, c1 in enumerate(flat):
        for j, c2 in enumerate(flat):
            if i >= j:
                continue
            dists = cdist(c1, c2)
            idx = np.unravel_index(dists.argmin(), dists.shape)
            d = dists[idx]
            if d < min_dist:
                min_dist = d
                best_p1 = c1[idx[0]]
                best_p2 = c2[idx[1]]
    return best_p1, best_p2


def _connect_contours(mask: np.ndarray) -> np.ndarray:
    """Draw lines between separate contours until only one remains."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    while len(contours) > 1:
        p1, p2 = _find_closest_contours(contours)
        cv2.line(mask, tuple(p1), tuple(p2), 255, 3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return mask


def connect_interior_exterior(mask: np.ndarray) -> np.ndarray:
    """Bridge interior holes to the outer contour so the mask is simply-connected."""
    mask = np.where(mask > 1, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) <= 1:
        return mask

    areas = [(cv2.contourArea(c), c) for c in contours]
    areas.sort(key=lambda x: x[0], reverse=True)
    outer = areas[0][1]

    for _, inner in areas[1:]:
        min_dist = float("inf")
        best = (tuple(outer[0][0]), tuple(inner[0][0]))
        for op in outer:
            for ip in inner:
                d = np.linalg.norm(op[0].astype(float) - ip[0].astype(float))
                if d < min_dist:
                    min_dist = d
                    best = (tuple(op[0]), tuple(ip[0]))
        cv2.line(mask, best[0], best[1], 0, 3)
    return mask


def mask_to_path(mask: np.ndarray, epsilon: float = 5.0) -> VectorPath:
    """Convert a binary mask to a closed cubic-Bezier VectorPath.

    Steps: binarize -> connect disjoint contours -> Douglas-Peucker simplification
    -> insert cubic Bezier control points.
    """
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    mask = np.where(mask > 1, 255, 0).astype(np.uint8)

    mask = _connect_contours(mask.copy())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        pts = torch.zeros(3, 2)
        return VectorPath(
            points=pts,
            num_control_points=torch.tensor([2], dtype=torch.int64),
            stroke_width=torch.tensor(0.0),
        )

    contour = contours[0]
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

    # Ensure at least 3 anchor points
    if len(simplified) == 1:
        x, y = simplified[0, 0, 0], simplified[0, 0, 1]
        simplified = np.array([[[x, y]], [[x + 3, y + 3]], [[x - 3, y + 3]]])
    elif len(simplified) == 2:
        x, y = simplified[0, 0, 0], simplified[0, 0, 1]
        new_pt = np.array(
            [[[x + random.uniform(2, 5), y + random.uniform(2, 5)]]]
        ).astype(simplified.dtype)
        simplified = np.insert(simplified, 1, new_pt, axis=0)

    anchor_pts = simplified[:, 0, :]  # (M, 2)
    # Close the loop, then insert 2 control points per segment
    closed = np.vstack([anchor_pts, anchor_pts[0:1]])
    interp = _interpolate_segments(closed, num_interp=2)
    points = torch.tensor(
        interp[:-1], dtype=torch.float32
    )  # drop duplicated last point
    num_control_points = torch.tensor([2] * len(anchor_pts), dtype=torch.int64)

    return VectorPath(
        points=points,
        num_control_points=num_control_points,
        stroke_width=torch.tensor(0.0),
    )
