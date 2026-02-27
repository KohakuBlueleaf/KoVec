"""Basic usage example for KoVec."""

from kovec import KoVecConfig, KoVecPipeline

config = KoVecConfig(
    resolution=512,
    device="cuda",
    sds={
        "model_id": "runwayml/stable-diffusion-v1-5",
        "model_type": "sd",
        "simplification_indices": [80, 60, 40, 20, 0],
    },
    segmentation={"backend": "sam2"},
    refinement={"max_path_num_limit": 256},
)

pipeline = KoVecPipeline(config)
scene = pipeline.run("input.png", "output.svg")
print(f"Generated SVG with {len(scene)} paths")
