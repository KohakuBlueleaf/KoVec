import torch
from diffusers import StableDiffusionPipeline
from torch import Tensor

from kovec.config import SDSConfig
from kovec.diffusion.base import DiffusionModel, ModelCondition, NoiseEstimate


class StableDiffusionModel(DiffusionModel):
    """Wrapper for Stable Diffusion 1.x / 2.x (eps and v_prediction)."""

    def __init__(self, config: SDSConfig, device: torch.device) -> None:
        self.device = device
        self.config = config

        dtype = torch.float32 if config.dtype == "float32" else torch.float16
        self.dtype = dtype

        pipe = StableDiffusionPipeline.from_pretrained(
            config.model_id, torch_dtype=dtype
        )

        self.scheduler = pipe.scheduler
        self.prediction_type = pipe.scheduler.config.prediction_type
        self.vae_scale = 0.18215

        # Precompute schedule
        with torch.inference_mode():
            self.alphas = torch.sqrt(self.scheduler.alphas_cumprod).to(
                device, dtype=dtype
            )
            self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(
                device, dtype=dtype
            )

        # Precompute text condition on CPU — only a single forward pass
        self._cached_condition = self._encode_text_cpu(
            pipe.tokenizer, pipe.text_encoder, config.prompt, device
        )

        # Move only VAE + UNet to GPU, delete text encoder from RAM
        self.vae = pipe.vae.to(device)
        self.unet = pipe.unet.to(device)
        del pipe

        self.vae.eval().requires_grad_(False)
        self.unet.eval().requires_grad_(False)

    @staticmethod
    @torch.no_grad()
    def _encode_text_cpu(
        tokenizer, text_encoder, prompt: str, device: torch.device
    ) -> ModelCondition:
        """Encode prompt on CPU, return condition tensors on *device*."""
        tokens_cond = tokenizer(
            [prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        tokens_uncond = tokenizer(
            [""],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        emb_cond = text_encoder(tokens_cond).last_hidden_state.detach()
        emb_uncond = text_encoder(tokens_uncond).last_hidden_state.detach()
        embeddings = torch.stack([emb_uncond, emb_cond], dim=1).to(device)
        return ModelCondition(embeddings=embeddings)

    @torch.no_grad()
    def encode_image(self, image: Tensor) -> Tensor:
        return (
            self.vae.encode(image.to(self.dtype))["latent_dist"].mean * self.vae_scale
        )

    @torch.no_grad()
    def decode_latent(self, latent: Tensor) -> Tensor:
        return self.vae.decode(
            (1 / self.vae_scale) * latent.to(self.dtype), return_dict=False
        )[0]

    def encode_text(self, prompt: str) -> ModelCondition:
        return self._cached_condition

    def get_schedule_params(self, timestep: Tensor) -> tuple[Tensor, Tensor]:
        return self.alphas[timestep], self.sigmas[timestep]

    def predict_noise(
        self,
        z_t: Tensor,
        timestep: Tensor,
        condition: ModelCondition,
        guidance_scale: float = 1.0,
    ) -> NoiseEstimate:
        latent_input = torch.cat([z_t] * 2)
        t_input = torch.cat([timestep] * 2)
        embeds = condition.embeddings.permute(1, 0, 2, 3).reshape(
            -1, *condition.embeddings.shape[2:]
        )

        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            raw = self.unet(latent_input, t_input, embeds).sample

            # Convert to eps-space
            match self.prediction_type:
                case "v_prediction":
                    alpha_2 = torch.cat([alpha_t] * 2)
                    sigma_2 = torch.cat([sigma_t] * 2)
                    raw = alpha_2 * raw + sigma_2 * latent_input
                case "sample":
                    alpha_2 = torch.cat([alpha_t] * 2)
                    sigma_2 = torch.cat([sigma_t] * 2)
                    raw = (latent_input - alpha_2 * raw) / sigma_2
                case "epsilon":
                    pass

            e_uncond, e_cond = raw.chunk(2)
            eps = e_uncond + guidance_scale * (e_cond - e_uncond)
            assert torch.isfinite(eps).all()

        pred_z0 = (z_t - sigma_t * eps) / alpha_t
        return NoiseEstimate(eps=eps, pred_z0=pred_z0)
