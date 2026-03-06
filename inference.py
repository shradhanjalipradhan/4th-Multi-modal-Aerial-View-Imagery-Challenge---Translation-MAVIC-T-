"""
MAVIC-T 2026 - Inference Script
Team: pshradha | Rank: #2

Generates outputs for all 4 translation tasks:
  1. SAR -> EO  (U-Net GAN)
  2. RGB -> IR  (grayscale conversion)
  3. SAR -> RGB (histogram matching)
  4. SAR -> IR  (histogram matching)

Usage:
    python inference.py --data_base /path/to/mavic-t-design-data --checkpoint weights/sar2eo_final.pth
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from src.model import UNetGenerator
from src.heuristics import build_reference_distributions, rgb_to_ir, sar_to_rgb, sar_to_ir


def parse_args():
    parser = argparse.ArgumentParser(description="MAVIC-T inference (all 4 tasks)")
    parser.add_argument(
        "--data_base",
        type=str,
        default="/kaggle/input/datasets/shradhanjali15/mavic-t-design-data",
    )
    parser.add_argument("--checkpoint", type=str, default="weights/sar2eo_final.pth")
    parser.add_argument("--output_dir", type=str, default="submission")
    parser.add_argument("--output_size", type=int, default=256, help="Output resolution (256 to avoid scorer OOM)")
    return parser.parse_args()


def infer_sar2eo(gen, device, test_dir, out_dir):
    """Task 1: SAR -> EO using trained U-Net GAN."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".png"):
            continue
        img = Image.open(os.path.join(test_dir, fname)).convert("L").resize((256, 256), Image.LANCZOS)
        inp = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            out = gen(inp.to(device)).cpu()
        out_np = ((out[0, 0].numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(out_np).convert("RGB").save(os.path.join(out_dir, fname))
        count += 1
        if count % 500 == 0:
            print(f"  SAR->EO: {count} done")
    print(f"SAR->EO complete: {count} files")


def infer_rgb2ir(test_dir, out_dir, output_size=256):
    """Task 2: RGB -> IR using grayscale conversion."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(test_dir)):
        if not (fname.endswith(".tiff") or fname.endswith(".tif")):
            continue
        rgb_to_ir(os.path.join(test_dir, fname), os.path.join(out_dir, fname), output_size)
        count += 1
    print(f"RGB->IR complete: {count} files")


def infer_sar2rgb(test_dir, out_dir, ref_rgb, output_size=256):
    """Task 3: SAR -> RGB using histogram matching."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(test_dir)):
        if not (fname.endswith(".tiff") or fname.endswith(".tif")):
            continue
        sar_to_rgb(os.path.join(test_dir, fname), os.path.join(out_dir, fname), ref_rgb, output_size)
        count += 1
    print(f"SAR->RGB complete: {count} files")


def infer_sar2ir(test_dir, out_dir, ref_ir, output_size=256):
    """Task 4: SAR -> IR using histogram matching."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(test_dir)):
        if not (fname.endswith(".tiff") or fname.endswith(".tif")):
            continue
        sar_to_ir(os.path.join(test_dir, fname), os.path.join(out_dir, fname), ref_ir, output_size)
        count += 1
    print(f"SAR->IR complete: {count} files")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find test path
    for test_root in ["mavic_t_2025_test/mavic_t_2025_test", "mavic_t_2025_test"]:
        test_path = os.path.join(args.data_base, test_root)
        if os.path.exists(test_path):
            break
    print(f"Test path: {test_path}")

    start = time.time()

    # --- Task 1: SAR -> EO ---
    print("\n[1/4] SAR -> EO (U-Net GAN)")
    gen = UNetGenerator(1, 1).to(device)
    gen.load_state_dict(torch.load(args.checkpoint, map_location=device))
    gen.eval()
    infer_sar2eo(gen, device, os.path.join(test_path, "sar2eo"), os.path.join(args.output_dir, "sar2eo"))
    del gen
    torch.cuda.empty_cache()

    # --- Task 2: RGB -> IR ---
    print("\n[2/4] RGB -> IR (grayscale conversion)")
    infer_rgb2ir(os.path.join(test_path, "rgb2ir"), os.path.join(args.output_dir, "rgb2ir"), args.output_size)

    # --- Build reference distributions for histogram matching ---
    print("\n[3/4] Building reference distributions...")
    uc_base = os.path.join(args.data_base, "uc_davis_merged_chips_stacks/uc_davis_merged_chips_stacks")
    if not os.path.exists(uc_base):
        uc_base = os.path.join(args.data_base, "uc_davis_merged_chips_stacks")
    ref_rgb, ref_ir = build_reference_distributions(uc_base)

    # --- Task 3: SAR -> RGB ---
    print("\n[3/4] SAR -> RGB (histogram matching)")
    infer_sar2rgb(os.path.join(test_path, "sar2rgb"), os.path.join(args.output_dir, "sar2rgb"), ref_rgb, args.output_size)

    # --- Task 4: SAR -> IR ---
    print("\n[4/4] SAR -> IR (histogram matching)")
    infer_sar2ir(os.path.join(test_path, "sar2ir"), os.path.join(args.output_dir, "sar2ir"), ref_ir, args.output_size)

    elapsed = time.time() - start
    print(f"\nAll tasks complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
