import click

from kovec.config import KoVecConfig
from kovec.pipeline import KoVecPipeline


@click.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("-o", "--output", default="output.svg", help="Output SVG path.")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("--device", default="cuda", help="Torch device.")
@click.option("--resolution", default=512, type=int, help="Longest side resolution.")
@click.option(
    "--resolution-step",
    default=64,
    type=int,
    help="Snap both dims to this step (default 64, good for SD/SDXL latent alignment).",
)
@click.option(
    "--model-id",
    default=None,
    help="Diffusion model HF id (overrides config).",
)
@click.option(
    "--model-type",
    type=click.Choice(["sd", "sdxl"]),
    default=None,
    help="Model type (overrides config).",
)
@click.option(
    "--dtype",
    type=click.Choice(["float32", "float16"]),
    default=None,
    help="Diffusion model dtype (float16 saves ~50%% VRAM).",
)
@click.option(
    "--segmentation-backend",
    type=click.Choice(["sam", "sam2"]),
    default=None,
    help="Segmentation backend (overrides config).",
)
@click.option(
    "--segmentation-checkpoint",
    default=None,
    help="Path to SAM/SAM2 checkpoint file.",
)
@click.option(
    "--segmentation-model-type",
    default=None,
    help="SAM model_type (e.g. vit_h) or SAM2 config (e.g. sam2_hiera_large).",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Directory to save intermediate SVGs (one per pipeline stage).",
)
def main(
    image: str,
    output: str,
    config_path: str | None,
    device: str,
    resolution: int,
    resolution_step: int,
    model_id: str | None,
    model_type: str | None,
    dtype: str | None,
    segmentation_backend: str | None,
    segmentation_checkpoint: str | None,
    segmentation_model_type: str | None,
    output_dir: str | None,
) -> None:
    """Vectorise IMAGE into a layered SVG."""
    if config_path:
        config = KoVecConfig.from_yaml(config_path)
    else:
        config = KoVecConfig()

    config.device = device
    config.resolution = resolution
    config.resolution_step = resolution_step
    if model_id:
        config.sds.model_id = model_id
    if model_type:
        config.sds.model_type = model_type
    if dtype:
        config.sds.dtype = dtype
    if segmentation_backend:
        config.segmentation.backend = segmentation_backend
    if segmentation_checkpoint:
        config.segmentation.checkpoint = segmentation_checkpoint
    if segmentation_model_type:
        config.segmentation.model_type = segmentation_model_type

    pipeline = KoVecPipeline(config)
    pipeline.run(image, output, output_dir=output_dir)
    click.echo(f"Saved to {output}")
    if output_dir:
        click.echo(f"Intermediate SVGs in {output_dir}")


if __name__ == "__main__":
    main()
