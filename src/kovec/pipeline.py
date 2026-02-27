import os

import torch

from kovec.config import KoVecConfig
from kovec.diffusion.base import DiffusionModel
from kovec.diffusion.sd import StableDiffusionModel
from kovec.diffusion.sdxl import SDXLModel
from kovec.diffusion.simplification import run_simplification
from kovec.optimization.color import fit_colors
from kovec.optimization.refinement import refine_visual
from kovec.optimization.structural import optimize_structural
from kovec.renderer.base import VectorRenderer
from kovec.renderer.torch_renderer import TorchRenderer
from kovec.renderer.triton_renderer import TritonRenderer
from kovec.segmentation.base import Segmentor
from kovec.segmentation.masks import filter_by_area, layer_masks, segment_image_sequence
from kovec.segmentation.sam import SAMSegmentor
from kovec.segmentation.sam2 import SAM2Segmentor
from kovec.utils.image import load_image
from kovec.vector.scene import build_scene_from_masks
from kovec.vector.types import VectorScene


def _build_diffusion_model(config: KoVecConfig, device: torch.device) -> DiffusionModel:
    match config.sds.model_type:
        case "sd":
            return StableDiffusionModel(config.sds, device)
        case "sdxl":
            return SDXLModel(config.sds, device)
        case _:
            raise ValueError(f"Unknown model_type: {config.sds.model_type}")


def _build_segmentor(config: KoVecConfig, device: torch.device) -> Segmentor:
    match config.segmentation.backend:
        case "sam":
            return SAMSegmentor(config.segmentation, device)
        case "sam2":
            return SAM2Segmentor(config.segmentation, device)
        case _:
            raise ValueError(f"Unknown backend: {config.segmentation.backend}")


def _save_stage(
    renderer: VectorRenderer,
    scene: "VectorScene",
    output_dir: str | None,
    counter: int,
    name: str,
    w: int,
    h: int,
) -> int:
    """Save an intermediate SVG snapshot.  No-op when *output_dir* is ``None``."""
    if output_dir is None:
        return counter
    path = os.path.join(output_dir, f"{counter:02d}_{name}.svg")
    renderer.save_svg(scene, path, w, h)
    return counter + 1


def _build_renderer(config: KoVecConfig, device: torch.device) -> VectorRenderer:
    match config.renderer.backend:
        case "torch":
            return TorchRenderer(
                device,
                samples_per_seg=config.renderer.samples_per_seg,
                sigma=config.renderer.sigma,
            )
        case "triton":
            return TritonRenderer(
                device,
                samples_per_seg=config.renderer.samples_per_seg,
                sigma=config.renderer.sigma,
            )
        case _:
            raise ValueError(f"Unknown renderer: {config.renderer.backend}")


class KoVecPipeline:
    """End-to-end layered vectorisation pipeline."""

    def __init__(self, config: KoVecConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.renderer = _build_renderer(config, self.device)

    def run(
        self,
        image_path: str,
        output_path: str = "output.svg",
        output_dir: str | None = None,
    ) -> VectorScene:
        """Run the full pipeline: simplify -> segment -> vectorise -> optimise."""
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        config = self.config
        device = self.device

        # 1. Load image (aspect-ratio preserving, snapped to step)
        target_img = load_image(image_path, config.resolution, config.resolution_step)
        h, w = target_img.shape[:2]

        # 2. SDS simplification
        diff_model = _build_diffusion_model(config, device)
        simp_seq = run_simplification(diff_model, target_img, config.sds)
        del diff_model
        torch.cuda.empty_cache()

        # 3. Segmentation
        segmentor = _build_segmentor(config, device)
        all_masks = segment_image_sequence(segmentor, simp_seq)
        del segmentor
        torch.cuda.empty_cache()

        # 4. Layer masks
        layered = layer_masks([[all_masks[0]]], all_masks[1:])
        max_struct = config.refinement.max_struct_paths
        layered = filter_by_area(layered, max_struct)

        # 5. Build initial scene (CPU tensors from OpenCV, then move to device)
        scene = build_scene_from_masks(
            layered, target_img, config.refinement.approxpolydp_epsilon
        )
        scene.to(device)
        stage = _save_stage(self.renderer, scene, output_dir, 0, "initial", w, h)

        # 6. Structural optimisation
        scene = optimize_structural(scene, self.renderer, target_img, layered, config)
        torch.cuda.empty_cache()
        stage = _save_stage(self.renderer, scene, output_dir, stage, "structural", w, h)

        # 7. Color fitting
        match config.refinement.color_fitting_type:
            case "dominan":
                scene, target_clustered = fit_colors(
                    scene,
                    target_img,
                    layered,
                    cluster=config.refinement.is_cluster_target_img,
                    k=config.refinement.kmeans_k,
                )
            case "mse":
                target_clustered = target_img
            case _:
                raise ValueError(
                    f"Unknown color_fitting_type: {config.refinement.color_fitting_type}"
                )

        # 8. Visual refinement
        torch.cuda.empty_cache()
        stage = _save_stage(self.renderer, scene, output_dir, stage, "color", w, h)
        scene, stage = refine_visual(
            scene,
            self.renderer,
            target_img,
            layered,
            config,
            output_dir=output_dir,
            stage_counter=stage,
            width=w,
            height=h,
        )

        # 9. Save
        _save_stage(self.renderer, scene, output_dir, stage, "final", w, h)
        self.renderer.save_svg(scene, output_path, w, h)
        return scene
