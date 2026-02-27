import tempfile

import gradio as gr
import numpy as np
from PIL import Image

from kovec.config import KoVecConfig
from kovec.pipeline import KoVecPipeline


def vectorize(
    image: np.ndarray,
    model_type: str,
    model_id: str,
    segmentation_backend: str,
    resolution: int,
    resolution_step: int,
    max_paths: int,
) -> str:
    """Run the KoVec pipeline on an uploaded image."""
    config = KoVecConfig(
        resolution=resolution,
        resolution_step=resolution_step,
        sds={"model_id": model_id, "model_type": model_type},
        segmentation={"backend": segmentation_backend},
        refinement={"max_path_num_limit": max_paths},
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        Image.fromarray(image).save(tmp_in.name)
        input_path = tmp_in.name

    output_path = input_path.replace(".png", ".svg")
    pipeline = KoVecPipeline(config)
    pipeline.run(input_path, output_path)
    return output_path


def create_app() -> gr.Blocks:
    with gr.Blocks(title="KoVec — Layered Image Vectorization") as app:
        gr.Markdown("# KoVec — Layered Image Vectorization")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Input Image", type="numpy")
                model_type = gr.Dropdown(["sd", "sdxl"], value="sd", label="Model Type")
                model_id = gr.Textbox(
                    value="runwayml/stable-diffusion-v1-5", label="Model ID"
                )
                seg_backend = gr.Dropdown(
                    ["sam", "sam2"], value="sam2", label="Segmentation Backend"
                )
                resolution = gr.Slider(
                    256, 1024, value=512, step=64, label="Resolution (longest side)"
                )
                resolution_step = gr.Slider(
                    8, 128, value=64, step=8, label="Resolution Step"
                )
                max_paths = gr.Slider(64, 512, value=256, step=32, label="Max Paths")
                run_btn = gr.Button("Vectorize", variant="primary")

            with gr.Column():
                output_file = gr.File(label="Output SVG")

        run_btn.click(
            fn=vectorize,
            inputs=[
                image_input,
                model_type,
                model_id,
                seg_backend,
                resolution,
                resolution_step,
                max_paths,
            ],
            outputs=[output_file],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
