"""
MAVIC-T 2026 - Model Architecture
Team: pshradha | Rank: #2

U-Net Generator (~54.4M params) with skip connections
PatchGAN Discriminator (~2.8M params, 70x70 receptive field)
"""

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """Encoder block: Conv(4x4, stride=2) -> InstanceNorm -> LeakyReLU(0.2)"""

    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose(4x4, stride=2) -> InstanceNorm -> ReLU -> cat(skip)"""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net Generator with skip connections for SAR -> EO translation.

    Architecture:
        Encoder: 1 -> 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512 -> 512
        Decoder: 512 -> 512 -> 512 -> 512 -> 256 -> 128 -> 64 -> 1
        (with concatenated skip connections doubling decoder input channels)

    Input:  1 x 256 x 256 (grayscale SAR)
    Output: 1 x 256 x 256 (grayscale EO)
    Parameters: ~54.4M
    """

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        # Encoder
        self.down1 = UNetDown(in_ch, 64, normalize=False)  # -> 64, 128x128
        self.down2 = UNetDown(64, 128)                      # -> 128, 64x64
        self.down3 = UNetDown(128, 256)                     # -> 256, 32x32
        self.down4 = UNetDown(256, 512)                     # -> 512, 16x16
        self.down5 = UNetDown(512, 512)                     # -> 512, 8x8
        self.down6 = UNetDown(512, 512)                     # -> 512, 4x4
        self.down7 = UNetDown(512, 512)                     # -> 512, 2x2
        self.down8 = UNetDown(512, 512, normalize=False)    # -> 512, 1x1

        # Decoder (in_ch accounts for concatenated skip connections)
        self.up1 = UNetUp(512, 512, dropout=0.5)    # cat d7 -> 1024
        self.up2 = UNetUp(1024, 512, dropout=0.5)   # cat d6 -> 1024
        self.up3 = UNetUp(1024, 512, dropout=0.5)   # cat d5 -> 1024
        self.up4 = UNetUp(1024, 512)                 # cat d4 -> 1024
        self.up5 = UNetUp(1024, 256)                 # cat d3 -> 512
        self.up6 = UNetUp(512, 128)                  # cat d2 -> 256
        self.up7 = UNetUp(256, 64)                   # cat d1 -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator.

    Input:  concat(SAR, EO) = 2 x 256 x 256
    Output: 1 x 30 x 30 (patch predictions)
    Parameters: ~2.8M
    """

    def __init__(self, in_ch=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def init_weights(m):
    """Initialize weights from N(0, 0.02) as per Pix2Pix paper."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = UNetGenerator(1, 1).to(device)
    disc = PatchGANDiscriminator(2).to(device)

    gen.apply(init_weights)
    disc.apply(init_weights)

    x = torch.randn(1, 1, 256, 256).to(device)
    with torch.no_grad():
        fake = gen(x)
        pred = disc(torch.cat([x, fake], dim=1))

    print(f"Generator:     {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M params")
    print(f"Discriminator: {sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M params")
    print(f"Input:  {x.shape}")
    print(f"Output: {fake.shape}")
    print(f"Disc:   {pred.shape}")
    print("Architecture OK!")
