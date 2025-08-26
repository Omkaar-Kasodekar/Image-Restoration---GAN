import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

# ============================
# Generator (U-Net)
# ============================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.down1 = self._conv_block(3, 64)
        self.down2 = self._conv_block(64, 128)
        self.down3 = self._conv_block(128, 256)
        self.down4 = self._conv_block(256, 512)
        self.bottleneck = self._conv_block(512, 512)

        # Decoder
        self.up1 = self._up_conv_block(512 + 512, 256)
        self.up2 = self._up_conv_block(256 + 256, 128)
        self.up3 = self._up_conv_block(128 + 128, 64)
        self.up4 = self._up_conv_block(64 + 64, 64)

        # Final conv to get RGB output
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        
        # MaxPool for encoder
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        # Bottleneck
        b = self.bottleneck(self.pool(d4))

        # Decoder with skip connections (interpolating to match encoder sizes)
        u1 = self.up1(torch.cat([F.interpolate(b, size=d4.shape[2:], mode='nearest'), d4], dim=1))
        u2 = self.up2(torch.cat([F.interpolate(u1, size=d3.shape[2:], mode='nearest'), d3], dim=1))
        u3 = self.up3(torch.cat([F.interpolate(u2, size=d2.shape[2:], mode='nearest'), d2], dim=1))
        u4 = self.up4(torch.cat([F.interpolate(u3, size=d1.shape[2:], mode='nearest'), d1], dim=1))

        # Ensure final output matches input size exactly
        out = self.final_conv(u4)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return torch.tanh(out)


# ============================
# Discriminator (PatchGAN)
# ============================
class Discriminator(nn.Module):
    def __init__(self, in_channels=6):  # concat(night, fake_or_day)
        super().__init__()

        def block(ic, oc, use_bn=True):
            layers = [nn.Conv2d(ic, oc, kernel_size=4, stride=2, padding=1, bias=False)]
            if use_bn:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.main = nn.Sequential(
            block(in_channels, 64, use_bn=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # logits
        )

    def forward(self, x):
        return self.main(x)  # no sigmoid, we’ll use BCEWithLogitsLoss


# ============================
# Dataset Loader
# ============================
class PairedDayNightDataset(Dataset):
    def __init__(self, night_dir, day_dir, image_size=256):
        self.night_paths = sorted(glob.glob(os.path.join(night_dir, "*")))
        self.day_dir = day_dir
        self.image_size = image_size
        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.night_paths)

    def __getitem__(self, idx):
        night_path = self.night_paths[idx]
        fname = os.path.basename(night_path)
        day_path = os.path.join(self.day_dir, fname)

        night = Image.open(night_path).convert("RGB")
        day = Image.open(day_path).convert("RGB")

        return self.tf(night), self.tf(day), fname


# ============================
# Metrics
# ============================
@torch.no_grad()
def denorm(x): return (x + 1) * 0.5

@torch.no_grad()
def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2) + 1e-8
    return 10 * torch.log10((max_val**2) / mse)

@torch.no_grad()
def ssim(pred, target):
    # simplified mean SSIM placeholder
    mu_x, mu_y = pred.mean(), target.mean()
    sigma_x, sigma_y = pred.var(), target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    C1, C2 = 0.01**2, 0.03**2
    return ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))


# ============================
# Training Loop
# ============================
def train(night_dir, day_dir, out_dir="./runs", 
          epochs=20, batch_size=8, lr=2e-4, device=None,
          debug=True, debug_subset_size=250):

    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = PairedDayNightDataset(night_dir, day_dir)

    # Use subset if in debug mode
    if debug:
        dataset = Subset(dataset, range(min(debug_subset_size, len(dataset))))
        print(f"[DEBUG MODE] Using {len(dataset)} images for quick testing")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        g_loss_total, d_loss_total = 0, 0
        print(f"\nStarting Epoch {epoch}/{epochs}")

        for i, (night, day, names) in enumerate(loader, 1):
            night, day = night.to(device), day.to(device)

            # --- Train Discriminator ---
            fake_day = G(night).detach()
            real_pair = torch.cat([night, day], dim=1)
            fake_pair = torch.cat([night, fake_day], dim=1)

            opt_D.zero_grad()
            d_real = D(real_pair)
            d_fake = D(fake_pair)
            loss_D = 0.5 * (bce(d_real, torch.ones_like(d_real)) +
                            bce(d_fake, torch.zeros_like(d_fake)))
            loss_D.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            fake_day = G(night)
            fake_pair = torch.cat([night, fake_day], dim=1)
            d_fake = D(fake_pair)
            loss_G = bce(d_fake, torch.ones_like(d_fake))
            loss_G.backward()
            opt_G.step()

            g_loss_total += loss_G.item()
            d_loss_total += loss_D.item()

            # Print progress every 5 batches
            if i % 5 == 0 or i == len(loader):
                print(f"Epoch {epoch} | Batch {i}/{len(loader)} | "
                      f"D_loss={loss_D.item():.4f} | G_loss={loss_G.item():.4f}")

        # --- Evaluation ---
        G.eval()
        psnr_vals, ssim_vals, l1_vals = [], [], []
        with torch.no_grad():
            for night, day, names in loader:
                night, day = night.to(device), day.to(device)
                fake = G(night)
                fake_dn, day_dn = denorm(fake), denorm(day)
                psnr_vals.append(psnr(fake_dn, day_dn).item())
                ssim_vals.append(ssim(fake_dn, day_dn).item())
                l1_vals.append(nn.functional.l1_loss(fake_dn, day_dn).item())

        print(f"[Epoch {epoch}/{epochs}] "
              f"Avg D_loss={d_loss_total/len(loader):.4f} | Avg G_loss={g_loss_total/len(loader):.4f} | "
              f"PSNR={sum(psnr_vals)/len(psnr_vals):.2f} | SSIM={sum(ssim_vals)/len(ssim_vals):.4f} | "
              f"L1={sum(l1_vals)/len(l1_vals):.4f}")

        # Save sample images
        save_image(torch.cat([denorm(night[:4]), denorm(fake[:4]), denorm(day[:4])], dim=0),
                   os.path.join(out_dir, f"epoch_{epoch}.png"), nrow=4)

    return G, D





# ============================
# Example Run
# ============================
if __name__ == "__main__":
    NIGHT_DIR = "night2day/night"
    DAY_DIR   = "night2day/day"
    train(NIGHT_DIR, DAY_DIR, out_dir="./pix2pix_unsupervised", epochs=20)
