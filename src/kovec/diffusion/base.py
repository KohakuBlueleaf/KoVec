from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from kovec.config import SDSConfig


class ModelCondition(BaseModel):
    """Stacked text embeddings: [uncond, cond]."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: Tensor  # (2, seq_len, dim) or model-specific shape
    pooled_embeds: Tensor | None = None  # SDXL: (2, dim) pooled text embeddings


class NoiseEstimate(BaseModel):
    """Predicted noise in epsilon-space, regardless of model parameterisation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    eps: Tensor
    pred_z0: Tensor | None = None


class DiffusionModel(ABC):
    """Abstract diffusion model interface."""

    @abstractmethod
    def encode_image(self, image: Tensor) -> Tensor:
        """VAE encode (1, 3, H, W) -> latent."""
        ...

    @abstractmethod
    def decode_latent(self, latent: Tensor) -> Tensor:
        """VAE decode latent -> (1, 3, H, W)."""
        ...

    @abstractmethod
    def encode_text(self, prompt: str) -> ModelCondition:
        """Encode text prompt -> stacked [uncond, cond] embeddings."""
        ...

    @abstractmethod
    def predict_noise(
        self,
        z_t: Tensor,
        timestep: Tensor,
        condition: ModelCondition,
        guidance_scale: float = 1.0,
    ) -> NoiseEstimate:
        """Predict noise — always returns eps-space regardless of parameterisation."""
        ...

    @abstractmethod
    def get_schedule_params(self, timestep: Tensor) -> tuple[Tensor, Tensor]:
        """Return (alpha_t, sigma_t) for the given timestep."""
        ...


class SDSLoss:
    """Model-agnostic Score Distillation Sampling loss."""

    def __init__(
        self,
        model: DiffusionModel,
        config: SDSConfig,
    ) -> None:
        self.model = model
        self.t_min = config.t_min
        self.t_max = config.t_max
        self.alpha_exp = config.alpha_exp
        self.sigma_exp = config.sigma_exp

    def _noise_input(
        self,
        z: Tensor,
        eps: Tensor | None = None,
        timestep: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if timestep is None:
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,
                size=(z.shape[0],),
                device=z.device,
                dtype=torch.long,
            )
        if eps is None:
            eps = torch.randn_like(z)

        alpha_t, sigma_t = self.model.get_schedule_params(timestep)
        alpha_t = alpha_t[:, None, None, None]
        sigma_t = sigma_t[:, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def compute(
        self,
        z: Tensor,
        condition: ModelCondition,
        guidance_scale: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Compute SDS loss. Returns (sds_loss, log_loss)."""
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self._noise_input(z)
            estimate = self.model.predict_noise(
                z_t, timestep, condition, guidance_scale
            )
            grad_z = (
                (alpha_t**self.alpha_exp)
                * (sigma_t**self.sigma_exp)
                * (estimate.eps - eps)
            )
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
            log_loss = (grad_z**2).mean()

        sds_loss = (grad_z.clone() * z).sum() / (z.shape[2] * z.shape[3])
        return sds_loss, log_loss
