"""
Product background replacement pipeline.

Three stages:
    1. Segmentation      - rembg (u2net) produces the product mask
    2. Inpainting        - SD 1.5 inpainting + canny ControlNet regenerates
                           only the background; product pixels are copied, not generated
    3. Relighting        - IC-Light (fc variant) recomputes lighting so that the
                           product and the new scene share a single light source

Recipe values below are the ones selected experimentally (see notebooks/ and README).
"""
import gc

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (ControlNetModel, StableDiffusionControlNetInpaintPipeline,
                       UniPCMultistepScheduler)
from rembg import new_session, remove

# --- Model ids -------------------------------------------------------------
SD_INPAINT_ID = "runwayml/stable-diffusion-inpainting"
SD_BASE_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_CANNY_ID = "lllyasviel/sd-controlnet-canny"
ICLIGHT_REPO = "lllyasviel/ic-light"
ICLIGHT_FILE = "iclight_sd15_fc.safetensors"

# --- Defaults (measured, see README) --------------------------------------
SIZE = (512, 640)  # (width, height); both axes must be divisible by 8 for the VAE
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, watermark, text, logo, signature"
)
CONTROLNET_SCALE = 0.6   # sweet spot from the scale sweep (edge-F1 plateau)
MASK_ERODE_PX = 7        # halo (too low) vs identity leak (too high)
MASK_FEATHER_PX = 21     # hides the seam between preserved and generated regions
CANNY_LOW, CANNY_HIGH = 20, 60
CANNY_BLUR = 5
CANNY_CROP_DILATE_PX = 15
INFERENCE_STEPS = 25
RELIGHT_STRENGTH = 0.7   # low keeps the stage-2 scene, high gives stronger light
RELIGHT_GUIDANCE = 2.0   # IC-Light works best at low CFG


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_models(device: str = "cuda") -> dict:
    """Load the segmentation session and the inpainting pipeline (stages 1-2).

    Expensive; call once and reuse the returned dict.
    """
    rembg_session = new_session("u2net")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_CANNY_ID, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        SD_INPAINT_ID,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return {"rembg": rembg_session, "pipe": pipe}


def load_iclight(weights_path: str | None = None, device: str = "cuda"):
    """Build an IC-Light pipeline on top of SD 1.5.

    IC-Light is architecturally an SD 1.5 UNet whose input convolution takes 8
    latent channels instead of 4: the extra 4 carry the clean foreground latent.
    The released file is an *offset* that is added to the SD 1.5 weights, not a
    full checkpoint. A forward hook injects the foreground latent so the stock
    img2img pipeline can be reused unchanged.
    """
    import safetensors.torch as sf
    from diffusers import (DDIMScheduler, StableDiffusionImg2ImgPipeline,
                           UNet2DConditionModel)

    unet = UNet2DConditionModel.from_pretrained(SD_BASE_ID, subfolder="unet")

    # 4 -> 8 input channels; new channels start at zero so the model is still
    # exactly SD 1.5 until the offset is merged.
    with torch.no_grad():
        conv = torch.nn.Conv2d(
            8,
            unet.conv_in.out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        conv.weight.zero_()
        conv.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        conv.bias = unet.conv_in.bias
        unet.conv_in = conv

    if weights_path is None:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(ICLIGHT_REPO, ICLIGHT_FILE)

    offset = sf.load_file(weights_path)
    origin = unet.state_dict()
    unet.load_state_dict({k: origin[k] + offset[k] for k in origin}, strict=True)
    del offset, origin
    unet = unet.half().to(device)

    original_forward = unet.forward

    def hooked_forward(sample, timestep, encoder_hidden_states, **kwargs):
        cond = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
        # classifier-free guidance doubles the batch; replicate the foreground
        cond = torch.cat([cond] * (sample.shape[0] // cond.shape[0]), dim=0)
        kwargs["cross_attention_kwargs"] = {}
        return original_forward(
            torch.cat([sample, cond], dim=1), timestep, encoder_hidden_states, **kwargs
        )

    unet.forward = hooked_forward

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_BASE_ID, unet=unet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


# --------------------------------------------------------------------------
# Stage 1 - masks and control image
# --------------------------------------------------------------------------
def build_masks(image, rembg_session, erode_px=MASK_ERODE_PX, feather=MASK_FEATHER_PX):
    """Return (inpaint_mask, product_mask).

    The inpaint mask is inverted (white = regenerate = background), eroded so the
    slightly dilated rembg silhouette does not drag old background pixels into the
    preserved region, and feathered to hide the seam.
    """
    product = np.array(remove(image, session=rembg_session, only_mask=True))
    eroded = (
        cv2.erode(product, np.ones((erode_px, erode_px), np.uint8))
        if erode_px > 0
        else product
    )
    background = cv2.GaussianBlur(255 - eroded, (feather, feather), 0)
    return Image.fromarray(background), product


def build_canny(
    image,
    product_mask,
    low=CANNY_LOW,
    high=CANNY_HIGH,
    dilate_px=CANNY_CROP_DILATE_PX,
):
    """Blurred canny edges, cropped to the product region, as a 3-channel image.

    Blur + low thresholds keep the silhouette closed where the product edge is in
    shadow. Cropping removes old-background edges so the model is free to invent a
    new scene; the mask is dilated first so the silhouette line itself survives.
    """
    array = np.array(image)
    edges = cv2.Canny(cv2.GaussianBlur(array, (CANNY_BLUR, CANNY_BLUR), 0), low, high)
    widened = cv2.dilate(product_mask, np.ones((dilate_px, dilate_px), np.uint8))
    edges = np.where(widened > 127, edges, 0).astype(np.uint8)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


# --------------------------------------------------------------------------
# Stage 2 - background inpainting
# --------------------------------------------------------------------------
def replace_background(
    image,
    prompt,
    models,
    negative_prompt=NEGATIVE_PROMPT,
    scale=CONTROLNET_SCALE,
    steps=INFERENCE_STEPS,
    seed=42,
    erode_px=MASK_ERODE_PX,
    feather=MASK_FEATHER_PX,
    device="cuda",
):
    """Generate a new background around a preserved product.

    `image` may be a PIL image or a path.
    Returns (result, product_mask) so the mask can be reused by stage 3.
    """
    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    image = image.convert("RGB").resize(SIZE)

    inpaint_mask, product_mask = build_masks(image, models["rembg"], erode_px, feather)
    control_image = build_canny(image, product_mask)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = models["pipe"](
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=inpaint_mask,
        control_image=control_image,
        num_inference_steps=steps,
        controlnet_conditioning_scale=float(scale),
        generator=generator,
        height=SIZE[1],
        width=SIZE[0],
    ).images[0]
    return result, product_mask


# --------------------------------------------------------------------------
# Stage 3 - relighting
# --------------------------------------------------------------------------
def relight(
    scene,
    product_mask,
    light_prompt,
    iclight_pipe,
    seed=42,
    strength=RELIGHT_STRENGTH,
    guidance=RELIGHT_GUIDANCE,
    steps=INFERENCE_STEPS,
    device="cuda",
):
    """Hybrid IC-Light conditioning.

    The concat latent is the product on a neutral grey field (identity, free of
    lighting), the img2img input is the stage-2 scene (composition), and the light
    itself comes from the prompt. Feeding the full lit scene as the concat latent
    makes the model copy the existing lighting instead of following the prompt.
    """
    if not isinstance(product_mask, Image.Image):
        product_mask = Image.fromarray(product_mask)

    array = np.array(scene).astype(np.float32)
    mask = (np.array(product_mask.convert("L")).astype(np.float32) / 255.0)[..., None]
    grey_field = (array * mask + 127.0 * (1 - mask)).astype(np.uint8)

    tensor = torch.from_numpy(grey_field).float().to(device) / 127.5 - 1.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).half()
    with torch.no_grad():
        concat = (
            iclight_pipe.vae.encode(tensor).latent_dist.mode()
            * iclight_pipe.vae.config.scaling_factor
        )

    generator = torch.Generator(device=device).manual_seed(int(seed))
    return iclight_pipe(
        prompt=light_prompt,
        negative_prompt="bad quality, blurry",
        image=scene,
        strength=float(strength),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        cross_attention_kwargs={"concat_conds": concat},
    ).images[0]


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def run_pipeline(image, scene_prompt, light_prompt, seed=42, sequential=True):
    """Full three-stage run. Returns (relit_result, stage2_scene).

    `sequential=True` unloads stage 1-2 before loading IC-Light, which is required
    on 12 GB GPUs. On hardware with enough VRAM, load both once and call the stage
    functions directly instead.
    """
    models = load_models()
    scene, product_mask = replace_background(image, scene_prompt, models, seed=seed)

    if sequential:
        del models
        gc.collect()
        torch.cuda.empty_cache()

    iclight_pipe = load_iclight()
    result = relight(scene, product_mask, light_prompt, iclight_pipe, seed=seed)

    if sequential:
        del iclight_pipe
        gc.collect()
        torch.cuda.empty_cache()

    return result, scene
