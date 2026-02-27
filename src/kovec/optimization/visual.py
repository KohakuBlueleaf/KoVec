import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from kovec.config import KoVecConfig
from kovec.renderer.base import VectorRenderer
from kovec.utils.image import rgba_to_rgb
from kovec.vector.types import VectorScene


def optimize_visual(
    scene: VectorScene,
    renderer: VectorRenderer,
    target_img: np.ndarray,
    config: KoVecConfig,
    opt_mask: list[int] | None = None,
    num_iters: int | None = None,
) -> VectorScene:
    """MSE-only visual optimisation with selective path optimisation via *opt_mask*."""
    device = torch.device(config.device)

    h, w = target_img.shape[:2]
    target_t = torch.tensor(target_img, device=device, dtype=torch.float32) / 255.0
    target_t = target_t.permute(2, 0, 1)

    white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)

    params = scene.enable_gradients(
        points=True,
        colors=config.train.is_train_visual_color,
        opt_mask=opt_mask,
    )
    lr_groups = [
        {"params": params[k], "lr": getattr(config.lr, k), "_id": k}
        for k in sorted(params)
    ]
    optimizer = torch.optim.Adam(lr_groups, betas=(0.9, 0.9), eps=1e-6)

    iters = num_iters if num_iters is not None else config.train.visual_opt_num_iters

    for _step in tqdm(range(iters), desc="Visual opt"):
        img = renderer.render(scene, w, h)
        rgb = rgba_to_rgb(img, white_bg)
        loss = F.mse_loss(rgb, target_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scene.disable_gradients()
    return scene
