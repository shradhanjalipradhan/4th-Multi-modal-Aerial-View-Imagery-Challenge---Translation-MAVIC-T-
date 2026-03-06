"""
MAVIC-T 2026 - Dataset
Paired SAR-EO dataset from UNICORN (68,151 aligned 256x256 pairs)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SAREODataset(Dataset):
    """
    Paired SAR-EO dataset for training the U-Net GAN.

    Args:
        sar_dir: Path to SAR training images
        eo_dir: Path to EO training images
        size: Image size (default 256)
        augment: Whether to apply random horizontal flips
    """

    def __init__(self, sar_dir, eo_dir, size=256, augment=True):
        self.sar_dir = sar_dir
        self.eo_dir = eo_dir
        self.size = size
        self.augment = augment
        self.files = sorted(list(set(os.listdir(sar_dir)) & set(os.listdir(eo_dir))))
        print(f"Dataset: {len(self.files)} paired images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        sar = Image.open(os.path.join(self.sar_dir, fname)).convert("L").resize(
            (self.size, self.size)
        )
        eo = Image.open(os.path.join(self.eo_dir, fname)).convert("L").resize(
            (self.size, self.size)
        )

        # Random horizontal flip
        if self.augment and np.random.random() > 0.5:
            sar = sar.transpose(Image.FLIP_LEFT_RIGHT)
            eo = eo.transpose(Image.FLIP_LEFT_RIGHT)

        # Normalize to [-1, 1]
        sar_t = torch.from_numpy(np.array(sar)).float().unsqueeze(0) / 127.5 - 1.0
        eo_t = torch.from_numpy(np.array(eo)).float().unsqueeze(0) / 127.5 - 1.0

        return sar_t, eo_t
