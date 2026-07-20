"""
Gradio demo for the product background replacement pipeline.

Local:   python app.py
Spaces:  the same file works if the `spaces` package is installed; models are then
         kept resident and the GPU decorator is applied automatically.
"""
import gc
import os
import sys

import gradio as gr
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline import (CONTROLNET_SCALE, MASK_ERODE_PX, NEGATIVE_PROMPT,  # noqa: E402
                      RELIGHT_STRENGTH, SIZE, build_canny, build_masks,
                      load_iclight, load_models, relight)

# On Spaces (ZeroGPU) models stay resident and calls are wrapped in @spaces.GPU.
# Locally, VRAM is usually too tight for both pipelines at once, so stages are
# loaded and released sequentially.
try:
    import spaces

    ON_SPACES = True
except ImportError:
    ON_SPACES = False

LIGHT_DIRECTIONS = {
    "Left (warm)": "warm golden hour light from left side, soft shadows",
    "Right (warm)": "warm golden hour light from right side, soft shadows",
    "Top (soft)": "soft diffused light from above, gentle shadows",
    "Front (studio)": "bright even studio light from front, minimal shadows",
    "No relight": None,
}

_models = None
_iclight = None


def generate(photo, scene_prompt, light_direction, scale, seed, erode_px,
             light_strength, progress=gr.Progress()):
    global _models, _iclight

    if photo is None:
        raise gr.Error("Please upload a product photo first.")

    progress(0.05, desc="Loading models...")
    models = _models if _models is not None else load_models()
    if ON_SPACES:
        _models = models

    image = photo.convert("RGB").resize(SIZE)

    progress(0.25, desc="Segmenting product...")
    inpaint_mask, product_mask = build_masks(image, models["rembg"], erode_px=int(erode_px))
    control_image = build_canny(image, product_mask)

    progress(0.40, desc="Generating background...")
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    scene = models["pipe"](
        prompt=scene_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        mask_image=inpaint_mask,
        control_image=control_image,
        num_inference_steps=25,
        controlnet_conditioning_scale=float(scale),
        generator=generator,
        height=SIZE[1],
        width=SIZE[0],
    ).images[0]

    mask_image = Image.fromarray(product_mask)
    light_prompt = LIGHT_DIRECTIONS[light_direction]
    if light_prompt is None:
        return scene, scene, mask_image

    if not ON_SPACES:
        del models
        gc.collect()
        torch.cuda.empty_cache()

    progress(0.70, desc="Relighting with IC-Light...")
    iclight = _iclight if _iclight is not None else load_iclight()
    if ON_SPACES:
        _iclight = iclight

    result = relight(
        scene,
        mask_image,
        f"{scene_prompt}, {light_prompt}",
        iclight,
        seed=int(seed),
        strength=float(light_strength),
    )

    if not ON_SPACES:
        del iclight
        gc.collect()
        torch.cuda.empty_cache()

    return result, scene, mask_image


if ON_SPACES:
    generate = spaces.GPU(duration=120)(generate)


with gr.Blocks(title="Product Background Replacement") as demo:
    gr.Markdown("# Product Background Replacement Pipeline")
    gr.Markdown(
        "Segmentation (rembg) &rarr; ControlNet inpainting (SD 1.5) &rarr; IC-Light "
        "relighting. Product pixels are preserved; only the background is generated "
        "and the lighting is recomputed so both share one light source."
    )

    with gr.Row():
        with gr.Column(scale=1):
            photo = gr.Image(type="pil", label="Product photo")
            scene_prompt = gr.Textbox(
                label="Scene prompt",
                value="professional product photo on a marble kitchen counter, high quality",
                lines=2,
            )
            light_direction = gr.Radio(
                list(LIGHT_DIRECTIONS.keys()),
                value="Left (warm)",
                label="Light direction",
            )
            with gr.Accordion("Advanced", open=False):
                scale = gr.Slider(
                    0.0, 1.2, value=CONTROLNET_SCALE, step=0.1,
                    label="ControlNet scale (measured sweet spot: 0.6)",
                )
                erode = gr.Slider(
                    0, 21, value=MASK_ERODE_PX, step=2,
                    label="Mask erosion px (halo vs identity leak)",
                )
                light_strength = gr.Slider(
                    0.5, 1.0, value=RELIGHT_STRENGTH, step=0.05,
                    label="IC-Light strength (low preserves the scene)",
                )
                seed = gr.Number(value=42, label="Seed", precision=0)
            run = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Image(label="Result (relit)")
            with gr.Row():
                intermediate = gr.Image(label="Stage 2 (before relight)")
                mask_view = gr.Image(label="Product mask")

    run.click(
        generate,
        inputs=[photo, scene_prompt, light_direction, scale, seed, erode, light_strength],
        outputs=[output, intermediate, mask_view],
    )

    gr.Markdown(
        "**Known limitations:** mask quality degrades on low-contrast scenes "
        "(dark product on a dark surface); transparent products keep the old "
        "background inside their silhouette; IC-Light's light control partially "
        "reduces background texture fidelity."
    )

if __name__ == "__main__":
    demo.launch()
