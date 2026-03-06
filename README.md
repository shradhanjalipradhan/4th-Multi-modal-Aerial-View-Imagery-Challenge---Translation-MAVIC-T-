# MAVIC-T 2026: Multi-Modal Aerial View Image Translation

**4th Multi-modal Aerial View Imagery Challenge — Translation Track**  
PBVS Workshop @ CVPR 2026 | Denver, CO, USA

**Team:** pshradha | **Final Ranking:** 🥈 #2 (Combined Score: 0.51)

---

## Overview

This repository contains the code for our solution to the MAVIC-T 2026 challenge, which requires translating aerial images across four sensor modality pairs:

| Task | Method | Parameters |
|------|--------|------------|
| SAR → EO | U-Net cGAN | 54.4M (generator) + 2.8M (discriminator) |
| SAR → RGB | Histogram matching | — |
| SAR → IR | Histogram matching | — |
| RGB → IR | Luminance + water heuristic | — |

## Results

| Rank | Team | Combined ↓ | SAR→EO | SAR→RGB | RGB→IR | SAR→IR |
|------|------|-----------|--------|---------|--------|--------|
| **2** | **pshradha** | **0.51** | 0.51 | 0.56 | 0.42 | 0.55 |

## Installation

```bash
git clone https://github.com/shradhanjalipradhan/4th-Multi-modal-Aerial-View-Imagery-Challenge---Translation-MAVIC-T-.git
cd 4th-Multi-modal-Aerial-View-Imagery-Challenge---Translation-MAVIC-T-
pip install -r requirements.txt
```

## Dataset

Download the MAVIC-T dataset from the [Codabench competition page](https://www.codabench.org/competitions/12566/) and place it at your preferred location. Update `--data_base` accordingly.

Expected structure:
```
mavic-t-design-data/
├── design_data/design_data/
│   ├── SAR/train/          # 68,151 PNG (256×256)
│   └── EO/train/           # 68,151 PNG (256×256)
├── uc_davis_merged_chips_stacks/
│   └── <locations>/        # Multi-modal TIFF stacks
└── mavic_t_2025_test/
    ├── sar2eo/             # 3,586 PNG
    ├── sar2rgb/            # 60 TIFF
    ├── sar2ir/             # 60 TIFF
    └── rgb2ir/             # 60 TIFF
```

## Training

Train the U-Net GAN for SAR → EO (requires ~5-6 hours on T4):

```bash
python train.py --data_base /path/to/mavic-t-design-data --epochs 5 --batch_size 16
```

Checkpoints are saved after every epoch to `weights/`.

## Inference

Generate outputs for all 4 tasks:

```bash
python inference.py --data_base /path/to/mavic-t-design-data --checkpoint weights/sar2eo_final.pth
```

## Package Submission

Create the submission ZIP for Codabench:

```bash
python package_submission.py --submission_dir submission
```

## Hardware

- **GPU:** NVIDIA Tesla T4 (16 GB VRAM)
- **CPU:** Intel Xeon (Kaggle, 4 cores)
- **RAM:** 13 GB
- **Training time:** ~5-6 hours
- **Inference time:** ~15 minutes (all 4 tasks)

## Repository Structure

```
├── configs/
│   └── config.yaml          # Training/inference configuration
├── figures/                  # Sample outputs
├── src/
│   ├── __init__.py
│   ├── model.py             # U-Net Generator + PatchGAN Discriminator
│   ├── dataset.py           # SAR-EO paired dataset
│   └── heuristics.py        # Histogram matching + grayscale conversion
├── weights/                  # Model checkpoints (after training)
├── train.py                  # Training script
├── inference.py              # Inference for all 4 tasks
├── package_submission.py     # Create submission ZIP
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you find this work useful, please cite the MAVIC-T challenge:

```bibtex
@inproceedings{mavict2024,
  title={Multi-modal Aerial View Image Challenge: Sensor Domain Translation},
  author={Low, Spencer and Nina, Oliver and Bowald, Dylan and Sappa, Angel D and Inkawhich, Nathan and Bruns, Peter},
  booktitle={CVPR Workshops},
  year={2024}
}
```

## License

MIT License
