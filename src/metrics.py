"""
Evaluation layer.

edge_f1    - structural fidelity: do the product's edges survive the pipeline?
clip_score - prompt adherence: does the output match the scene description?

A third metric (lighting consistency) was attempted and dropped. Two formulations
were tried - product/background histogram agreement, and gradient-direction
agreement - and both ended up measuring silhouette contrast rather than lighting.
See the Limitations section of the README.
"""
import cv2
import numpy as np
import torch

_clip_model = None
_clip_processor = None

CLIP_ID = "openai/clip-vit-base-patch32"


def canny_of(image, low=20, high=60, blur=5):
    """Same detector settings the pipeline uses, so the ruler matches the tool."""
    array = np.array(image.convert("RGB"))
    return cv2.Canny(cv2.GaussianBlur(array, (blur, blur), 0), low, high) > 0


def edge_f1(source, output, product_mask, tolerance=2):
    """Edge agreement inside the product region.

    A tolerance is needed because even a VAE round-trip shifts edges by a pixel or
    two; reference edges are dilated so a near miss still counts.

    Low recall    -> original edges disappeared (lost detail).
    Low precision -> edges appeared that were not there (invented detail).
    """
    mask = np.array(product_mask.convert("L")) > 127
    ref = canny_of(source) & mask
    out = canny_of(output) & mask

    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    ref_thick = cv2.dilate(ref.astype(np.uint8), kernel) > 0
    out_thick = cv2.dilate(out.astype(np.uint8), kernel) > 0

    precision = (out & ref_thick).sum() / max(out.sum(), 1)
    recall = (ref & out_thick).sum() / max(ref.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def clip_score(image, text, device="cuda"):
    """Cosine similarity between CLIP image and text embeddings.

    Typical values sit between 0.20 and 0.35; only relative comparisons against a
    fixed prompt are meaningful. The model is loaded lazily on first call.
    """
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        _clip_model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)

    inputs = _clip_processor(
        text=[text], images=image, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        out = _clip_model(**inputs)
        image_embed = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_embed = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float(image_embed @ text_embed.T)


def mask_iou(mask_a, mask_b, threshold=127):
    """Intersection over union between two masks.

    Measures agreement, not correctness: two methods can agree perfectly and both
    be wrong (see the transparent-glass case in the segmentation ablation).
    """
    a = np.array(mask_a.convert("L")) > threshold
    b = np.array(mask_b.convert("L")) > threshold
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


def mask_diff_map(mask_a, mask_b, threshold=127):
    """Visual diff: red = only B, blue = only A, white = both."""
    from PIL import Image

    a = np.array(mask_a.convert("L")) > threshold
    b = np.array(mask_b.convert("L")) > threshold
    out = np.zeros((*a.shape, 3), dtype=np.uint8)
    out[np.logical_and(b, ~a)] = [255, 60, 60]
    out[np.logical_and(a, ~b)] = [60, 60, 255]
    out[np.logical_and(a, b)] = [255, 255, 255]
    return Image.fromarray(out)
