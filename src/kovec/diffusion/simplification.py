import numpy as np
import torch
from torch.optim.sgd import SGD
from tqdm import tqdm

from kovec.config import SDSConfig
from kovec.diffusion.base import DiffusionModel, SDSLoss
from kovec.utils.image import denormalize, normalize


def run_simplification(
    model: DiffusionModel,
    image: np.ndarray,
    config: SDSConfig,
) -> list[np.ndarray]:
    """SDS-based progressive image simplification.

    Returns a list of images: [original, simplified_at_idx_0, ...].
    The indices in ``config.simplification_indices`` select which iteration
    snapshots to keep (sorted descending, e.g. [80, 60, 40, 20, 0]).
    """
    device = next(
        p.device
        for p in (
            model.unet.parameters() if hasattr(model, "unet") else [torch.tensor(0)]
        )
    )

    image_t = normalize(image).to(device)
    z_source = model.encode_image(image_t)
    condition = model.encode_text(config.prompt)

    z_target = z_source.clone()
    z_target.requires_grad = True
    optimizer = SGD(params=[z_target], lr=config.lr)
    sds = SDSLoss(model, config)

    num_iters = config.simplification_indices[0]
    all_results: list[np.ndarray] = []

    for i in tqdm(range(num_iters), desc="SDS simplification"):
        loss, _ = sds.compute(z_target, condition, config.guidance_scale)
        optimizer.zero_grad()
        (config.loss_scale * loss).backward()
        optimizer.step()

        with torch.no_grad():
            decoded = model.decode_latent(z_target)
        out = denormalize(decoded)
        all_results.append(out)

    # Collect selected snapshots
    result = [image]
    for idx in config.simplification_indices:
        if 0 < idx <= len(all_results):
            result.append(all_results[idx - 1])
        elif idx == 0:
            result.append(image)

    return result
