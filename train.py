"""
MAVIC-T 2026 - Training Script
Team: pshradha | Rank: #2

Trains U-Net GAN for SAR -> EO translation on UNICORN dataset.

Usage:
    python train.py --data_base /path/to/mavic-t-design-data --epochs 5 --batch_size 16
"""

import argparse
import os
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import UNetGenerator, PatchGANDiscriminator, init_weights
from src.dataset import SAREODataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net GAN for SAR->EO")
    parser.add_argument(
        "--data_base",
        type=str,
        default="/kaggle/input/datasets/shradhanjali15/mavic-t-design-data",
        help="Base path to MAVIC-T dataset",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0001)
    parser.add_argument("--l1_weight", type=float, default=100.0)
    parser.add_argument("--output_dir", type=str, default="weights")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset
    sar_dir = os.path.join(args.data_base, "design_data/design_data/SAR/train")
    eo_dir = os.path.join(args.data_base, "design_data/design_data/EO/train")
    dataset = SAREODataset(sar_dir, eo_dir, size=256, augment=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # Models
    gen = UNetGenerator(1, 1).to(device)
    disc = PatchGANDiscriminator(2).to(device)
    gen.apply(init_weights)
    disc.apply(init_weights)

    print(f"Generator:     {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M params")
    print(f"Discriminator: {sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M params")

    # Optimizers
    opt_G = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batches/epoch: {len(loader)}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()

        for i, (sar, eo) in enumerate(loader):
            sar, eo = sar.to(device), eo.to(device)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            fake_eo = gen(sar).detach()

            real_pair = torch.cat([sar, eo], dim=1)
            pred_real = disc(real_pair)
            loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real) * 0.9)

            fake_pair = torch.cat([sar, fake_eo], dim=1)
            pred_fake = disc(fake_pair)
            loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            fake_eo = gen(sar)

            fake_pair = torch.cat([sar, fake_eo], dim=1)
            pred_fake = disc(fake_pair)
            loss_gan = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
            loss_l1 = F.l1_loss(fake_eo, eo) * args.l1_weight

            loss_G = loss_gan + loss_l1
            loss_G.backward()
            opt_G.step()

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Ep {epoch} [{i + 1}/{len(loader)}] "
                    f"G: {loss_G.item():.4f} D: {loss_D.item():.4f} "
                    f"({elapsed / 60:.1f}min)"
                )

        # LR decay after epoch 3
        if epoch >= 3:
            for pg in opt_G.param_groups:
                pg["lr"] *= 0.9
            for pg in opt_D.param_groups:
                pg["lr"] *= 0.9

        # Save checkpoint
        ckpt = os.path.join(args.output_dir, f"sar2eo_ep{epoch}.pth")
        torch.save(gen.state_dict(), ckpt)
        print(f">>> Epoch {epoch} saved to {ckpt}")

    # Save final model
    final = os.path.join(args.output_dir, "sar2eo_final.pth")
    torch.save(gen.state_dict(), final)
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining complete! {total_time:.1f} minutes")
    print(f"Final model: {final}")


if __name__ == "__main__":
    main()
