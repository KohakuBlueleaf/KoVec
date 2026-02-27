import torch
from diffusers import StableDiffusionXLPipeline
from torch import Tensor

from kovec.config import SDSConfig
from kovec.diffusion.base import DiffusionModel, ModelCondition, NoiseEstimate


class SDXLModel(DiffusionModel):
    """Wrapper for Stable Diffusion XL (dual CLIP, pooled embeddings, time_ids)."""

    def __init__(self, config: SDSConfig, device: torch.device) -> None:
        self.device = device
        self.config = config

        dtype = torch.float32 if config.dtype == "float32" else torch.float16
        self.dtype = dtype

        pipe = StableDiffusionXLPipeline.from_pretrained(
            config.model_id, torch_dtype=dtype
        )

        self.scheduler = pipe.scheduler
        self.prediction_type = pipe.scheduler.config.prediction_type
        self.vae_scale = 0.13025

        with torch.inference_mode():
            self.alphas = torch.sqrt(self.scheduler.alphas_cumprod).to(
                device, dtype=dtype
            )
            self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(
                device, dtype=dtype
            )

        # Precompute text condition on CPU — dual CLIP, single forward each
        self._cached_condition = self._encode_text_cpu(
            pipe.tokenizer,
            pipe.tokenizer_2,
            pipe.text_encoder,
            pipe.text_encoder_2,
            config.prompt,
            device,
        )

        # Move only VAE + UNet to GPU, delete text encoders from RAM
        self.vae = pipe.vae.to(device)
        self.unet = pipe.unet.to(device)
        del pipe

        self.vae.eval().requires_grad_(False)
        self.unet.eval().requires_grad_(False)

    @staticmethod
    @torch.no_grad()
    def _encode_single(
        tokenizer, text_encoder, prompt: str
    ) -> tuple[Tensor, Tensor | None]:
        tokens = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        output = text_encoder(tokens, output_hidden_states=True)
        hidden = output.hidden_states[-2].detach()
        pooled = output.text_embeds.detach() if hasattr(output, "text_embeds") else None
        return hidden, pooled

    @staticmethod
    @torch.no_grad()
    def _encode_text_cpu(
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        prompt: str,
        device: torch.device,
    ) -> ModelCondition:
        """Encode prompt on CPU with dual CLIP, return condition on *device*."""
        h1, _ = SDXLModel._encode_single(tokenizer, text_encoder, prompt)
        h2, pooled = SDXLModel._encode_single(tokenizer_2, text_encoder_2, prompt)
        prompt_embeds = torch.cat([h1, h2], dim=-1)

        h1_u, _ = SDXLModel._encode_single(tokenizer, text_encoder, "")
        h2_u, pooled_u = SDXLModel._encode_single(tokenizer_2, text_encoder_2, "")
        uncond_embeds = torch.cat([h1_u, h2_u], dim=-1)

        embeddings = torch.stack([uncond_embeds, prompt_embeds], dim=1).to(device)
        pooled_embeds = torch.cat([pooled_u, pooled], dim=0).to(device)
        return ModelCondition(embeddings=embeddings, pooled_embeds=pooled_embeds)

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

        # SDXL requires text_embeds (pooled) and time_ids in added_cond_kwargs
        # Derive pixel dims from latent shape: latent (B, C, H/8, W/8)
        px_h = float(z_t.shape[2] * 8)
        px_w = float(z_t.shape[3] * 8)
        add_time_ids = torch.tensor(
            [[px_h, px_w, 0.0, 0.0, px_h, px_w]], device=self.device, dtype=self.dtype
        ).repeat(2, 1)
        added_cond = {
            "text_embeds": condition.pooled_embeds,
            "time_ids": add_time_ids,
        }

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            raw = self.unet(
                latent_input,
                t_input,
                encoder_hidden_states=embeds,
                added_cond_kwargs=added_cond,
            ).sample

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
