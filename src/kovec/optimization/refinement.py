import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

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
    target_clustered: np.ndarray,
    pseudo_struct_masks: list[np.ndarray],
    opt_mask: list[int],
    epsilon: float = 5.0,
    n: int = 50,
    device: torch.device | None = None,
) -> tuple[VectorScene, list[np.ndarray], list[int], int]:
    """Add new paths in high-error regions.

    Error detection uses *target_clustered* (k-means quantised) so the error
    map has large uniform regions instead of fine-grained noise.  Colors are
    also sampled from the clustered image (matching reference behaviour).
    """
    if device is None:
        device = torch.device("cpu")

    h, w = target_clustered.shape[:2]
    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)
    with torch.no_grad():
        raster = renderer.render(scene, w, h)
        raster_rgb = rgba_to_rgb(raster, white_bg).cpu().numpy()
    target_f = np.transpose((target_clustered / 255.0).astype(np.float32), (2, 0, 1))

    masks = _select_mask_by_conn_area(raster_rgb, target_f, n)
    if not masks:
        return scene, pseudo_struct_masks, opt_mask, -1

    for mask in masks:
        color = _get_mean_color(target_clustered, mask)
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


def _remove_low_quality_paths(
    scene: VectorScene,
    renderer: VectorRenderer,
    struct_path_num: int,
    opt_mask: list[int],
    w: int,
    h: int,
    threshold: float = 7.0,
) -> tuple[VectorScene, list[int]]:
    """Remove visual paths whose presence barely changes the rendered image.

    For each visual path, render the scene without it and measure L1 pixel
    difference.  If the difference is <= *threshold*, the path is negligible
    and gets removed.  Structural paths (index < *struct_path_num*) are
    always kept.
    """
    device = scene.paths[0].points.device if scene.paths else torch.device("cpu")
    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)

    with torch.no_grad():
        full_rgb = rgba_to_rgb(renderer.render(scene, w, h), white_bg)

    remove_indices: set[int] = set()
    for i in tqdm(range(struct_path_num, len(scene)), desc="Remove check"):
        sub_paths: list = []
        sub_groups: list = []
        for j in range(len(scene)):
            if j == i:
                continue
            sub_paths.append(scene.paths[j])
            sub_groups.append(
                VectorPathGroup(
                    shape_idx=len(sub_paths) - 1,
                    fill_color=scene.groups[j].fill_color,
                    stroke_color=scene.groups[j].stroke_color,
                )
            )
        sub = VectorScene(paths=sub_paths, groups=sub_groups)
        with torch.no_grad():
            sub_rgb = rgba_to_rgb(renderer.render(sub, w, h), white_bg)
        diff = torch.sum(torch.abs(sub_rgb - full_rgb)).item()
        if diff <= threshold:
            remove_indices.add(i)

    if remove_indices:
        print(f"    [remove] {len(remove_indices)} low-quality paths removed")
        scene.remove(remove_indices)
        opt_mask = [v for i, v in enumerate(opt_mask) if i not in remove_indices]
    else:
        print("    [remove] no paths removed")

    return scene, opt_mask


def _merge_paths(
    scene: VectorScene,
    renderer: VectorRenderer,
    struct_path_num: int,
    pseudo_struct_masks: list[np.ndarray],
    opt_mask: list[int],
    w: int,
    h: int,
    color_threshold: float = 0.1,
    overlap_threshold: int = 3,
    epsilon: float = 5.0,
    device: torch.device | None = None,
) -> tuple[VectorScene, list[np.ndarray], list[int], int]:
    """Merge overlapping visual paths with similar colors.

    Each visual path is rendered individually (white on black).  Pairs that
    overlap by >= *overlap_threshold* pixels and whose fill-color L1 distance
    is <= *color_threshold* are greedily merged into a single new path.
    """
    if device is None:
        device = torch.device("cpu")

    n_visual = len(scene) - struct_path_num
    if n_visual == 0:
        return scene, pseudo_struct_masks, opt_mask, struct_path_num

    black_bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Render each visual path individually (white fill on black bg)
    path_imgs: list[torch.Tensor] = []
    for idx in tqdm(range(struct_path_num, len(scene)), desc="Merge render"):
        white_group = VectorPathGroup(
            shape_idx=0,
            fill_color=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
            stroke_color=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
        )
        single = VectorScene(paths=[scene.paths[idx]], groups=[white_group])
        with torch.no_grad():
            rgb = rgba_to_rgb(renderer.render(single, w, h), black_bg)
        path_imgs.append(rgb[0])  # (H, W) coverage

    path_colors = [
        scene.groups[i].fill_color for i in range(struct_path_num, len(scene))
    ]

    # Greedy pairwise merge (matching reference algorithm)
    record_list: list[list[int]] = [[i] for i in range(len(path_imgs))]
    merged = False
    i = 0
    while i < len(path_imgs):
        j = i + 1
        while j < len(path_imgs):
            overlap = torch.sum((path_imgs[i] + path_imgs[j]) > 1.9).item()
            cdiff = torch.sum(torch.abs(path_colors[i] - path_colors[j])).item()
            if overlap >= overlap_threshold and cdiff <= color_threshold:
                path_imgs[i] = torch.clamp(path_imgs[i] + path_imgs[j], max=1)
                path_colors[i] = ((path_colors[i] + path_colors[j]) / 2).detach()
                record_list[i] = record_list[i] + record_list[j]
                path_imgs.pop(j)
                path_colors.pop(j)
                record_list.pop(j)
                merged = True
            else:
                j += 1
        if not merged:
            i += 1
        else:
            merged = False

    # Identify groups that were actually merged (len > 1)
    merged_imgs = [
        path_imgs[i] for i in range(len(record_list)) if len(record_list[i]) > 1
    ]
    merged_colors = [
        path_colors[i] for i in range(len(record_list)) if len(record_list[i]) > 1
    ]
    merged_records = [r for r in record_list if len(r) > 1]
    flat_merged = {item for sublist in merged_records for item in sublist}

    # Rebuild scene: struct + kept visual + new merged paths
    new_paths = list(scene.paths[:struct_path_num])
    new_groups = list(scene.groups[:struct_path_num])
    struct_opt = list(opt_mask[:struct_path_num])
    visual_opt = list(opt_mask[struct_path_num:])

    for i in range(n_visual):
        if i not in flat_merged:
            new_paths.append(scene.paths[struct_path_num + i])
            new_groups.append(scene.groups[struct_path_num + i])
    kept_visual_opt = [v for i, v in enumerate(visual_opt) if i not in flat_merged]

    for img_t, color in zip(merged_imgs, merged_colors):
        mask_np = (img_t.detach().cpu().numpy() * 255).astype(np.uint8)
        mask_np = connect_interior_exterior(mask_np)
        new_path = mask_to_path(mask_np, epsilon)
        new_path.to(device)
        new_group = VectorPathGroup(
            shape_idx=0,
            fill_color=color.clone(),
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
        )
        new_paths.append(new_path)
        new_groups.append(new_group)
        kept_visual_opt.append(1)

    scene = VectorScene(paths=new_paths, groups=new_groups)
    scene.reindex_groups()
    opt_mask = struct_opt + kept_visual_opt

    n_merged_paths = len(flat_merged)
    n_new = len(merged_records)
    if n_merged_paths > 0:
        print(
            f"    [merge] {n_merged_paths} paths merged into {n_new} "
            f"({len(scene)} total)"
        )
    else:
        print("    [merge] no paths merged")

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
    target_clustered: np.ndarray,
    layered_masks: list[list[np.ndarray]],
    config: KoVecConfig,
    output_dir: str | None = None,
    stage_counter: int = 0,
    width: int = 0,
    height: int = 0,
) -> tuple[VectorScene, int]:
    """Visual refinement loop: add paths -> optimise -> remove -> merge -> optimise.

    Matches reference: skip remove/merge on last round.  N paths per round
    is computed dynamically as 60% of remaining capacity (all remaining on
    last round).
    """
    device = torch.device(config.device)
    ref = config.refinement
    h, w = target_img.shape[:2]

    pseudo_struct_masks = [m for layer in layered_masks for m in layer]
    opt_mask: list[int] = [0] * len(scene)  # structural paths: frozen
    struct_path_num = len(scene)

    for i in range(ref.num_rounds):
        # Dynamic N: 60% of remaining, or all remaining on last round
        if i == ref.num_rounds - 1:
            n_paths = ref.max_path_limit - len(scene)
        else:
            n_paths = int((ref.max_path_limit - len(scene)) * 0.6)
        n_paths = max(n_paths, 0)

        n_before = len(scene)
        scene, pseudo_struct_masks, opt_mask, struct_path_num = _add_visual_paths(
            scene,
            renderer,
            struct_path_num,
            target_img,
            target_clustered,
            pseudo_struct_masks,
            opt_mask,
            epsilon=ref.approxpolydp_epsilon,
            n=n_paths,
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

        # First optimisation (full iterations)
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

        # Skip remove/merge on last round (matching reference)
        if i == ref.num_rounds - 1:
            break

        # Remove low-quality paths
        scene, opt_mask = _remove_low_quality_paths(
            scene,
            renderer,
            struct_path_num,
            opt_mask,
            w,
            h,
            threshold=ref.paths_remove_threshold,
        )
        stage_counter = _save_stage(
            renderer,
            scene,
            output_dir,
            stage_counter,
            f"refine_r{i}_remove",
            width,
            height,
        )

        # Merge similar paths
        scene, pseudo_struct_masks, opt_mask, struct_path_num = _merge_paths(
            scene,
            renderer,
            struct_path_num,
            pseudo_struct_masks,
            opt_mask,
            w,
            h,
            color_threshold=ref.paths_merge_color_threshold,
            overlap_threshold=ref.paths_merge_overlap_threshold,
            epsilon=ref.approxpolydp_epsilon,
            device=device,
        )
        stage_counter = _save_stage(
            renderer,
            scene,
            output_dir,
            stage_counter,
            f"refine_r{i}_merge",
            width,
            height,
        )

        # Second optimisation (shorter, post-merge phase)
        scene = optimize_visual(
            scene,
            renderer,
            target_img,
            config,
            opt_mask=opt_mask,
            num_iters=ref.merge_opt_num_iters,
        )
        stage_counter = _save_stage(
            renderer,
            scene,
            output_dir,
            stage_counter,
            f"refine_r{i}_opt2",
            width,
            height,
        )

    return scene, stage_counter
