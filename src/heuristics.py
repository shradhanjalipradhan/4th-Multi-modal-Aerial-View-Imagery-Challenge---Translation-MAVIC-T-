"""
MAVIC-T 2026 - Heuristic Methods
Task-specific baselines for SAR->RGB, SAR->IR, and RGB->IR
"""

import os
import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None
    print("WARNING: rasterio not installed. TIFF loading will fail.")


def hist_match(source, reference):
    """
    CDF-based histogram matching.
    Transfers the intensity distribution of source to match reference.
    """
    s = source.flatten().astype(np.uint8)
    r = reference.flatten().astype(np.uint8)
    s_vals, s_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
    r_vals, r_counts = np.unique(r, return_counts=True)
    s_cdf = np.cumsum(s_counts).astype(np.float64) / s.size
    r_cdf = np.cumsum(r_counts).astype(np.float64) / r.size
    mapped = np.interp(s_cdf, r_cdf, r_vals.astype(np.float64))
    return np.clip(mapped[s_idx].reshape(source.shape), 0, 255).astype(np.uint8)


def build_reference_distributions(uc_davis_path):
    """
    Build reference RGB and IR intensity distributions from UC Davis MAGIC stacks.

    Returns:
        ref_rgb: list of 3 arrays (R, G, B channel pixel values)
        ref_ir: array of IR pixel values, or None
    """
    all_rgb = [[], [], []]
    all_ir = []

    for loc in sorted(os.listdir(uc_davis_path)):
        loc_path = os.path.join(uc_davis_path, loc)
        if not os.path.isdir(loc_path):
            continue
        for f in os.listdir(loc_path):
            fp = os.path.join(loc_path, f)
            try:
                if "rgb" in f.lower() and f.endswith((".tiff", ".tif")):
                    with rasterio.open(fp) as s:
                        data = s.read()
                    if data.shape[0] >= 3:
                        for c in range(3):
                            v = data[c].flatten()
                            v = v[v > 0]
                            if len(v) > 0:
                                all_rgb[c].append(
                                    v[np.random.choice(len(v), min(50000, len(v)), replace=False)]
                                )
                elif (
                    "ir" in f.lower()
                    and "rgb" not in f.lower()
                    and f.endswith((".tiff", ".tif"))
                ):
                    with rasterio.open(fp) as s:
                        data = s.read()
                    v = data[0].flatten()
                    v = v[v > 0]
                    if len(v) > 0:
                        all_ir.append(
                            v[np.random.choice(len(v), min(50000, len(v)), replace=False)]
                        )
            except Exception:
                continue

    ref_rgb = [np.concatenate(ch).astype(np.uint8) for ch in all_rgb]
    ref_ir = np.concatenate(all_ir).astype(np.uint8) if all_ir else None

    print(f"Reference RGB: {[len(c) for c in ref_rgb]} pixels/channel")
    print(f"Reference IR:  {len(ref_ir) if ref_ir is not None else 0} pixels")

    return ref_rgb, ref_ir


def rgb_to_ir(img_path, output_path, output_size=256):
    """
    RGB -> IR conversion using physics-informed grayscale with water darkening.

    Formula: I_IR = 0.85 * (0.299R + 0.587G + 0.114B) + 0.15 * mean(RGB)
    Water heuristic: pixels with blue_ratio > 0.4 and intensity < 100 scaled by 0.5
    """
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float64)

    valid = (img[0] > 0) | (img[1] > 0) | (img[2] > 0) if img.shape[0] >= 3 else img[0] > 0

    if img.shape[0] >= 3:
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        gray = gray * 0.85 + np.mean(img[:3], axis=0) * 0.15
        # Water body darkening
        blue_ratio = img[2] / (img[0] + img[1] + img[2] + 1e-8)
        gray[(blue_ratio > 0.4) & (gray < 100) & valid] *= 0.5
    else:
        gray = img[0].copy()

    gray[~valid] = 0
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(gray).convert("RGB").resize((output_size, output_size), Image.LANCZOS)
    out_img.save(output_path)


def sar_to_rgb(sar_path, output_path, ref_rgb, output_size=256):
    """SAR -> RGB via per-channel histogram matching."""
    with rasterio.open(sar_path) as s:
        sar = s.read()
    sg = (sar[0] if sar.shape[0] == 1 else np.mean(sar[:3], axis=0)).astype(np.uint8)
    rgb = np.stack([hist_match(sg, ref_rgb[c]) for c in range(3)], axis=-1)
    Image.fromarray(rgb).resize((output_size, output_size), Image.LANCZOS).save(output_path)


def sar_to_ir(sar_path, output_path, ref_ir, output_size=256):
    """SAR -> IR via histogram matching."""
    with rasterio.open(sar_path) as s:
        sar = s.read()
    sg = (sar[0] if sar.shape[0] == 1 else np.mean(sar[:3], axis=0)).astype(np.uint8)
    ir = hist_match(sg, ref_ir) if ref_ir is not None else sg
    Image.fromarray(ir).convert("RGB").resize((output_size, output_size), Image.LANCZOS).save(
        output_path
    )
