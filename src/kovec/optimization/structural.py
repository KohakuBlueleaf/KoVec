import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

from kovec.config import KoVecConfig
from kovec.renderer.base import VectorRenderer
from kovec.utils.image import rgba_to_rgb
from kovec.vector.types import VectorPathGroup, VectorScene


def _build_struct_targets(
    layered_masks: list[list[np.ndarray]],
) -> tuple[list[Tensor], list[list[list[float]]]]:
    """Build random-color structure targets for each layer."""
    targets: list[Tensor] = []
    colors_per_layer: list[list[list[float]]] = []

    for masks in layered_masks:
        h, w = masks[0].shape[:2]
        seg_image = torch.zeros(3, h, w)
        layer_colors: list[list[float]] = []

        for mask in masks:
            color_channels: list[float] = []
            tensor = torch.zeros(3, h, w)
            for ch in range(3):
                val = 0.2 + 0.8 * torch.rand(1).item()
                tensor[ch, :, :] = val
                color_channels.append(val)
            layer_colors.append(color_channels)

            mask_t = transforms.ToTensor()(Image.fromarray(mask))
            seg_image = torch.clamp(seg_image + tensor * mask_t, max=1.0)

        targets.append(seg_image)
        colors_per_layer.append(layer_colors)

    return targets, colors_per_layer


def _exclude_loss(raster_img: Tensor, scale: float = 2e-7) -> Tensor:
    """Penalise dark (near-transparent) regions to encourage coverage."""
    img = F.relu(178.0 / 255.0 - raster_img)
    return torch.sum(img) * scale


def optimize_structural(
    scene: VectorScene,
    renderer: VectorRenderer,
    target_img: np.ndarray,
    layered_masks: list[list[np.ndarray]],
    config: KoVecConfig,
) -> VectorScene:
    """Layer-wise structural optimisation: per-layer MSE + transparency penalty + global MSE."""
    device = torch.device(config.device)

    struct_targets, struct_colors = _build_struct_targets(layered_masks)
    struct_targets = [t.to(device) for t in struct_targets]

    # Build per-layer ShapeGroups with matching random colors
    struct_groups_per_layer: list[list[VectorPathGroup]] = []
    for layer_colors in struct_colors:
        groups: list[VectorPathGroup] = []
        for i, color in enumerate(layer_colors):
            g = VectorPathGroup(
                shape_idx=i,
                fill_color=torch.tensor(color + [1.0], device=device),
                stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
            )
            groups.append(g)
        struct_groups_per_layer.append(groups)

    # Transparent groups for exclude-loss
    transparent_groups: list[VectorPathGroup] = []
    for i in range(len(scene)):
        g = VectorPathGroup(
            shape_idx=i,
            fill_color=torch.tensor([0.0, 0.0, 0.0, 0.3], device=device),
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 0.3], device=device),
        )
        transparent_groups.append(g)

    black_bg = torch.tensor([0.0, 0.0, 0.0], device=device)
    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)

    h, w = target_img.shape[:2]
    target_t = torch.tensor(target_img, device=device, dtype=torch.float32) / 255.0
    target_t = target_t.permute(2, 0, 1)

    # Enable gradients + optimizer
    params = scene.enable_gradients(
        points=True,
        colors=config.train.is_train_struct_color,
    )
    lr_groups = [
        {"params": params[k], "lr": getattr(config.lr, k), "_id": k}
        for k in sorted(params)
    ]
    optimizer = torch.optim.Adam(lr_groups, betas=(0.9, 0.9), eps=1e-6)

    shape_idx = 0
    layer_ranges: list[tuple[int, int]] = []
    for layer in layered_masks:
        start = shape_idx
        shape_idx += len(layer)
        layer_ranges.append((start, shape_idx))

    for _step in tqdm(range(config.train.struct_opt_num_iters), desc="Structural opt"):
        loss_struct = torch.tensor(0.0, device=device)
        loss_exclude = torch.tensor(0.0, device=device)

        for li, (start, end) in enumerate(layer_ranges):
            sub_scene = VectorScene(
                paths=scene.paths[start:end],
                groups=struct_groups_per_layer[li],
            )
            sub_scene.reindex_groups()
            struct_img = renderer.render(sub_scene, w, h)
            struct_rgb = rgba_to_rgb(struct_img, black_bg)
            loss_struct = loss_struct + F.mse_loss(struct_rgb, struct_targets[li])

            trans_scene = VectorScene(
                paths=scene.paths[start:end],
                groups=transparent_groups[: end - start],
            )
            trans_scene.reindex_groups()
            trans_img = renderer.render(trans_scene, w, h)
            trans_rgb = rgba_to_rgb(trans_img, white_bg)
            loss_exclude = loss_exclude + _exclude_loss(trans_rgb)

        full_img = renderer.render(scene, w, h)
        full_rgb = rgba_to_rgb(full_img, white_bg)
        loss_mse = F.mse_loss(full_rgb, target_t)

        loss = loss_mse * 0.02 + loss_exclude + loss_struct
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scene.disable_gradients()
    return scene
