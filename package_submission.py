"""
MAVIC-T 2026 - Package Submission
Creates submission.zip with all outputs and readme.txt

Usage:
    python package_submission.py --submission_dir submission
"""

import argparse
import os
import zipfile
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", type=str, default="submission")
    parser.add_argument("--output_zip", type=str, default="submission.zip")
    args = parser.parse_args()

    sub_dir = args.submission_dir
    expected = {
        "sar2eo": 3586,
        "sar2rgb": 60,
        "sar2ir": 60,
        "rgb2ir": 60,
    }

    # Verify
    print("=== VERIFICATION ===")
    all_ok = True
    for folder, count in expected.items():
        path = os.path.join(sub_dir, folder)
        if not os.path.exists(path):
            print(f"  {folder}: MISSING")
            all_ok = False
            continue
        files = sorted(os.listdir(path))
        sample = Image.open(os.path.join(path, files[0]))
        ok = len(files) == count
        status = "OK" if ok else "FAIL"
        print(f"  {folder}: {len(files)} files | size={sample.size} | mode={sample.mode} [{status}]")
        if not ok:
            all_ok = False

    # Create readme
    readme_path = os.path.join(sub_dir, "readme.txt")
    readme = """runtime per image [s] : 0.05
CPU[1] / GPU[0] : 0
Extra Data [1] / No Extra Data [0] : 0
Other description : U-Net GAN with skip connections for SAR2EO (LSGAN+L1 loss, 5 epochs on 68K pairs). Grayscale conversion for RGB2IR. Histogram matching for SAR2RGB and SAR2IR using UC Davis reference data.
"""
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"\n  readme.txt created")

    # Create ZIP
    if os.path.exists(args.output_zip):
        os.remove(args.output_zip)

    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme_path, "readme.txt")
        for folder in expected:
            folder_path = os.path.join(sub_dir, folder)
            for fname in sorted(os.listdir(folder_path)):
                zf.write(os.path.join(folder_path, fname), f"{folder}/{fname}")

    size_mb = os.path.getsize(args.output_zip) / 1024 / 1024
    print(f"\nZIP: {args.output_zip} ({size_mb:.1f} MB)")
    print("READY TO SUBMIT!" if all_ok else "FIX ISSUES ABOVE BEFORE SUBMITTING")


if __name__ == "__main__":
    main()
