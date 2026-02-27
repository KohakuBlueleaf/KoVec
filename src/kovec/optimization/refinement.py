import os

import cv2
import numpy as np
import torch

from kovec.config import KoVecConfig
from kovec.optimization.visual import optimize_visual
from kovec.renderer.base import VectorRenderer
from kovec.utils.image import rgba_to_rgb
from kovec.vector.contour import connect_interior_exterior, mask_to_path
from kovec.vector.scene import _get_mean_color
from kovec.vector.types import VectorPathGroup, VectorScene


def _select_mask_by_conn_area(
    pred: np.ndarray,
    gt: np.ndarray,
    n: int = -1,
) -> list[np.ndarray]:
    """Find connected high-error regions between *pred* and *gt* (both 3,H,W float)."""
    error_map = ((pred - gt) ** 2).sum(0)

    nodiff_thres = 0.025
    error_map[error_map < nodiff_thres] = 0

    quantile_interval = np.linspace(0.0, 1.0, 139)
    quantized = np.unique(np.quantile(error_map, quantile_interval))
    quantized = sorted(quantized[1:-1])
    error_map = np.digitize(error_map, quantized, right=False)
    error_map = np.clip(error_map, 0, 255).astype(np.uint8)

    csize_list: list[list[int]] = []
    component_list: list[np.ndarray] = []

    for val in np.unique(error_map):
        if val == 0:
            continue
        _, comp, cstats, _ = cv2.connectedComponentsWithStats(
            (error_map == val).astype(np.uint8), connectivity=4
        )
        sizes = [c[-1] for c in cstats[1:]]
        csize_list.append(sizes)
        component_list.append(comp)

    if not csize_list:
        return []

    max_len = max(len(s) for s in csize_list)
    csize_arr = np.array([s + [0] * (max_len - len(s)) for s in csize_list])
    mask_ = csize_arr >= 1
    values = csize_arr[mask_]
    indices = np.argwhere(mask_)
    sorted_indices = indices[np.argsort(-values)]
    if n >= 0:
        sorted_indices = sorted_indices[:n]

    masks: list[np.ndarray] = []
    for idx in sorted_indices:
        m = (component_list[idx[0]] == idx[1] + 1).astype(np.uint8) * 255
        masks.append(m)
    return masks


def _insert_in_struct_layer(
    mask: np.ndarray,
    struct_masks: list[np.ndarray],
) -> tuple[bool, list[np.ndarray], int]:
    """Try to insert *mask* into the structural layer ordering."""
    for index, existing in enumerate(struct_masks):
        mask_area = np.sum(mask == 255)
        existing_area = np.sum(existing == 255)
        intersection = np.sum(
            (mask.astype(np.uint16) + existing.astype(np.uint16)) == 510
        )
        if (
            existing_area > 0
            and intersection / existing_area >= 0.7
            and mask_area > 1.1 * existing_area
        ):
            struct_masks.insert(
                index,
                np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.uint8),
            )
            return True, struct_masks, index
    return False, struct_masks, 0


def _add_visual_paths(
    scene: VectorScene,
    renderer: VectorRenderer,
    struct_path_num: int,
    target_img: np.ndarray,
    pseudo_struct_masks: list[np.ndarray],
    opt_mask: list[int],
    epsilon: float = 5.0,
    n: int = 50,
    device: torch.device | None = None,
) -> tuple[VectorScene, list[np.ndarray], list[int], int]:
    """Add new paths in high-error regions."""
    if device is None:
        device = torch.device("cpu")

    h, w = target_img.shape[:2]
    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)
    with torch.no_grad():
        raster = renderer.render(scene, w, h)
        raster_rgb = rgba_to_rgb(raster, white_bg).cpu().numpy()
    target_f = np.transpose((target_img / 255.0).astype(np.float32), (2, 0, 1))

    masks = _select_mask_by_conn_area(raster_rgb, target_f, n)
    if not masks:
        return scene, pseudo_struct_masks, opt_mask, -1

    for mask in masks:
        color = _get_mean_color(target_img, mask)
        group = VectorPathGroup(
            shape_idx=0,
            fill_color=torch.tensor(list(color) + [255], device=device) / 255.0,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
        )
        mask = connect_interior_exterior(mask)
        path = mask_to_path(mask, epsilon)
        path.to(device)

        is_struct, pseudo_struct_masks, insert_idx = _insert_in_struct_layer(
            mask, pseudo_struct_masks
        )
        if is_struct:
            scene.insert_at(insert_idx, path, group)
            opt_mask.insert(insert_idx, 1)
            struct_path_num += 1
        else:
            scene.append(path, group)
            opt_mask.append(1)

    scene.reindex_groups()
    return scene, pseudo_struct_masks, opt_mask, struct_path_num


def _save_stage(
    renderer: VectorRenderer,
    scene: VectorScene,
    output_dir: str | None,
    counter: int,
    name: str,
    w: int,
    h: int,
) -> int:
    if output_dir is None:
        return counter
    path = os.path.join(output_dir, f"{counter:02d}_{name}.svg")
    renderer.save_svg(scene, path, w, h)
    return counter + 1


def refine_visual(
    scene: VectorScene,
    renderer: VectorRenderer,
    target_img: np.ndarray,
    layered_masks: list[list[np.ndarray]],
    config: KoVecConfig,
    output_dir: str | None = None,
    stage_counter: int = 0,
    width: int = 0,
    height: int = 0,
) -> tuple[VectorScene, int]:
    """Visual refinement: repeatedly add K paths from high-error regions, then optimise."""
    device = torch.device(config.device)
    ref = config.refinement
    h, w = target_img.shape[:2]

    pseudo_struct_masks = [m for layer in layered_masks for m in layer]
    opt_mask: list[int] = [0] * len(scene)  # structural paths: frozen
    struct_path_num = len(scene)

    for i in range(ref.num_rounds):
        n_before = len(scene)
        scene, pseudo_struct_masks, opt_mask, struct_path_num = _add_visual_paths(
            scene,
            renderer,
            struct_path_num,
            target_img,
            pseudo_struct_masks,
            opt_mask,
            epsilon=ref.approxpolydp_epsilon,
            n=ref.paths_per_round,
            device=device,
        )
        print(f"  Round {i}: added {len(scene) - n_before} paths ({len(scene)} total)")
        if struct_path_num == -1:
            break

        stage_counter = _save_stage(
            renderer,
            scene,
            output_dir,
            stage_counter,
            f"refine_r{i}_add",
            width,
            height,
        )

        scene = optimize_visual(scene, renderer, target_img, config, opt_mask=opt_mask)
        stage_counter = _save_stage(
            renderer,
            scene,
            output_dir,
            stage_counter,
            f"refine_r{i}_opt",
            width,
            height,
        )

    return scene, stage_counter
