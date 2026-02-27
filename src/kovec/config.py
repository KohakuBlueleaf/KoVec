from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SDSConfig(BaseModel):
    model_id: str = "runwayml/stable-diffusion-v1-5"
    model_type: Literal["sd", "sdxl"] = "sd"
    prompt: str = " "
    num_iters: int = 80
    lr: float = 0.1
    loss_scale: float = 2000.0
    guidance_scale: float = 0.0
    t_min: int = 50
    t_max: int = 950
    alpha_exp: float = 0.0
    sigma_exp: float = 0.0
    simplification_indices: list[int] = Field(
        default_factory=lambda: [80, 60, 40, 20, 0]
    )
    dtype: Literal["float32", "float16"] = "float32"


class SegmentationConfig(BaseModel):
    backend: Literal["sam", "sam2"] = "sam2"
    checkpoint: str | None = None
    model_type: str = "sam2_hiera_l.yaml"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 100
    box_nms_thresh: float = 0.7


class TrainConfig(BaseModel):
    struct_opt_num_iters: int = 200
    visual_opt_num_iters: int = 200
    is_train_stroke: bool = True
    is_train_struct_color: bool = True
    is_train_visual_color: bool = True


class LRConfig(BaseModel):
    point: float = 1.0
    color: float = 0.05
    stroke_width: float = 0.2
    stroke_color: float = 0.05


class RefinementConfig(BaseModel):
    num_rounds: int = 6
    paths_per_round: int = 64
    max_struct_paths: int = 200
    approxpolydp_epsilon: float = 5.0
    color_fitting_type: Literal["dominan", "mse"] = "dominan"
    is_cluster_target_img: bool = True
    kmeans_k: int = 32


class RendererConfig(BaseModel):
    backend: Literal["torch", "triton"] = "triton"
    samples_per_seg: int = 16
    sigma: float = 1.0


class KoVecConfig(BaseModel):
    resolution: int = 512
    resolution_step: int = 64
    device: str = "cuda"
    renderer: RendererConfig = Field(default_factory=RendererConfig)
    sds: SDSConfig = Field(default_factory=SDSConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    lr: LRConfig = Field(default_factory=LRConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "KoVecConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
