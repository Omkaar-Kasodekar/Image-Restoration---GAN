# ==============================================================================
# Pix2Pix for Night-to-Day Image Translation with Enhanced Loss & Scheduling
#
# Description:
# This script implements a Pix2Pix GAN with an enhanced loss function to
# translate night-time images into day-time images.
#
# Evaluation Metrics:
# 1. Fréchet Inception Distance (FID): Lower is better.
# 2. Learned Perceptual Image Patch Similarity (LPIPS): Lower is better.
#
# Dependencies:
# - PyTorch, Torchvision, Pillow
# - torch-fidelity (for FID calculation)
# - lpips (for LPIPS calculation)
# - piq (for SSIM loss calculation)
#
# To install dependencies:
# pip install torch torchvision Pillow
# pip install torch-fidelity lpips piq
# ==============================================================================

import os
import glob
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image
import lpips 
import piq # For SSIM Loss

# ============================
# Generator (U-Net)
# ============================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder (Down-sampling path)
        self.down1 = self._conv_block(3, 64, first_block=True)
        self.down2 = self._conv_block(64, 128)
        self.down3 = self._conv_block(128, 256)
        self.down4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 512)

        # Decoder (Up-sampling path)
        self.up_conv1 = self._up_conv(512, 512)
        self.up_block1 = self._conv_block(512 + 512, 256) # Cat(up_conv1, d4)
        
        self.up_conv2 = self._up_conv(256, 256)
        self.up_block2 = self._conv_block(256 + 256, 128) # Cat(up_conv2, d3)
        
        self.up_conv3 = self._up_conv(128, 128)
        self.up_block3 = self._conv_block(128 + 128, 64)  # Cat(up_conv3, d2)

        self.up_conv4 = self._up_conv(64, 64)
        self.up_block4 = self._conv_block(64 + 64, 64)   # Cat(up_conv4, d1)

        # Final convolution to produce the 3-channel RGB image
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        
        # MaxPool for down-sampling in the encoder
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_channels, out_channels, first_block=False):
        # A standard convolutional block with two conv layers
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        return nn.Sequential(*layers)

    def _up_conv(self, in_channels, out_channels):
        # A single transpose convolution to up-sample feature maps
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # --- Encoder Path ---
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        # --- Bottleneck ---
        b = self.bottleneck(self.pool(d4))

        # --- Decoder Path with Skip Connections ---
        u1 = self.up_conv1(b)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.up_block1(u1)

        u2 = self.up_conv2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.up_block2(u2)

        u3 = self.up_conv3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.up_block3(u3)

        u4 = self.up_conv4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.up_block4(u4)

        # Final output layer
        out = self.final_conv(u4)
        
        # Use tanh activation to scale output to [-1, 1]
        return torch.tanh(out)


# ============================
# Discriminator (PatchGAN)
# ============================
class Discriminator(nn.Module):
    def __init__(self, in_channels=6):  # Input: concat(night_image, day_image)
        super(Discriminator, self).__init__()

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
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # Output logits
        )

    def forward(self, x):
        return self.main(x)


# ============================
# Dataset Loader
# ============================
class PairedDayNightDataset(Dataset):
    def __init__(self, night_dir, day_dir, image_size=256):
        self.night_paths = sorted(glob.glob(os.path.join(night_dir, "*")))
        self.day_dir = day_dir
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.night_paths)

    def __getitem__(self, idx):
        night_path = self.night_paths[idx]
        fname = os.path.basename(night_path)
        day_path = os.path.join(self.day_dir, fname)

        try:
            night_img = Image.open(night_path).convert("RGB")
            day_img = Image.open(day_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Corresponding day image not found for {fname}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        return self.transform(night_img), self.transform(day_img)


# ============================
# Utilities & Evaluation
# Utilities & Evaluation
# ============================
@torch.no_grad()
def denorm(x):
    # Denormalize tensor from [-1, 1] to [0, 1] for saving/viewing
    return (x + 1) / 2.0

@torch.no_grad()
def calculate_lpips(generator, val_loader, loss_fn_lpips, device, epoch):
    print(f"--- Calculating LPIPS for epoch {epoch} ---")
    generator.eval()
    
    lpips_scores = []
    for night, day in val_loader:
        night, day = night.to(device), day.to(device)
        fake_day = generator(night)
        dist = loss_fn_lpips(fake_day, day)
        for score in dist:
            lpips_scores.append(score.item())
    
    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    print(f"LPIPS Score: {avg_lpips:.4f}")
    
    generator.train()
    return avg_lpips

@torch.no_grad()
def calculate_fid(generator, val_loader, device, epoch, out_dir):
    print(f"--- Calculating FID for epoch {epoch} ---")
    generator.eval()

    gen_dir = os.path.join(out_dir, "fid_generated")
    real_dir = os.path.join(out_dir, "fid_real")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    for i, (night, day) in enumerate(val_loader):
        night = night.to(device)
        fake_day = generator(night)
        
        for j in range(fake_day.size(0)):
            img_num = i * val_loader.batch_size + j
            save_image(denorm(fake_day[j]), os.path.join(gen_dir, f"{img_num:04d}.png"))
            save_image(denorm(day[j]), os.path.join(real_dir, f"{img_num:04d}.png"))

    print("Running FID calculation... This may take a moment.")
    try:
        cmd = ["fidelity", "--gpu", "0", "--fid", "--input1", real_dir, "--input2", gen_dir]
        cmd = ["fidelity", "--gpu", "0", "--fid", "--input1", real_dir, "--input2", gen_dir]
        result = subprocess.check_output(cmd, universal_newlines=True)
        fid_score = float(result.strip().split()[-1])
        print(f"FID Score: {fid_score:.4f}")
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid_score = float('inf')
        fid_score = float('inf')
    finally:
        shutil.rmtree(gen_dir)
        shutil.rmtree(real_dir)
    
    generator.train()
    generator.train()
    return fid_score

# ============================
# Training Loop
# ============================
def train(night_dir, day_dir, out_dir="./runs", 
          epochs=200, batch_size=8, lr=2e-4, device=None,
          debug=True, debug_subset_size=4000, l1_lambda=100.0,
          ssim_lambda=5.0, lpips_lambda=10.0, eval_every=5):

    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PairedDayNightDataset(night_dir, day_dir)

    if debug:
        dataset = Subset(dataset, range(min(debug_subset_size, len(dataset))))
        print(f"[DEBUG MODE] Using {len(dataset)} images for quick testing.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # --- NEW: Learning Rate Scheduler ---
    def lambda_rule(epoch):
        # Start decaying after half the epochs are done
        lr_l = 1.0 - max(0, epoch + 1 - epochs // 2) / float(epochs // 2 + 1)
        return lr_l

    scheduler_G = LambdaLR(opt_G, lr_lambda=lambda_rule)
    scheduler_D = LambdaLR(opt_D, lr_lambda=lambda_rule)
    # ------------------------------------

    # --- Losses ---
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    ssim_loss = piq.SSIMLoss(data_range=1.).to(device) # SSIM works on images in [0, 1] range
    # --------------

    best_fid = float('inf')
    best_lpips = float('inf')

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        g_loss_total, d_loss_total = 0.0, 0.0
        print(f"\n--- Starting Epoch {epoch}/{epochs} ---")

        for i, (night, day) in enumerate(loader, 1):
            night, day = night.to(device), day.to(device)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            with torch.no_grad():
                fake_day = G(night)
            real_pair = torch.cat([night, day], dim=1)
            d_real_logits = D(real_pair)
            loss_D_real = bce_loss(d_real_logits, torch.ones_like(d_real_logits))
            fake_pair = torch.cat([night, fake_day], dim=1)
            d_fake_logits = D(fake_pair)
            loss_D_fake = bce_loss(d_fake_logits, torch.zeros_like(d_fake_logits))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            fake_day_for_g = G(night)
            fake_pair_for_g = torch.cat([night, fake_day_for_g], dim=1)
            d_fake_logits_for_g = D(fake_pair_for_g)
            
            # Adversarial Loss
            loss_G_gan = bce_loss(d_fake_logits_for_g, torch.ones_like(d_fake_logits_for_g))
            
            # L1 Loss (Pixel-wise)
            loss_G_l1 = l1_loss(fake_day_for_g, day) * l1_lambda
            
            # --- NEW LOSSES ---
            # SSIM Loss (Structural) - Note: SSIM is a similarity, so loss is (1 - ssim)
            # We use denorm to move images from [-1, 1] to [0, 1] for the SSIM calculation
            loss_G_ssim = (1 - ssim_loss(denorm(fake_day_for_g), denorm(day))) * ssim_lambda
            
            # LPIPS Loss (Perceptual)
            loss_G_lpips = loss_fn_lpips(fake_day_for_g, day).mean() * lpips_lambda
            
            # --- COMBINED LOSS ---
            loss_G = loss_G_gan + loss_G_l1 + loss_G_ssim + loss_G_lpips
            # --------------------
            
            loss_G.backward()
            opt_G.step()

            g_loss_total += loss_G.item()
            d_loss_total += loss_D.item()

            if i % 50 == 0 or i == len(loader):
                print(f"Epoch {epoch} | Batch {i}/{len(loader)} | "
                      f"D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f} | "
                      f"G_gan: {loss_G_gan.item():.4f} | G_l1: {loss_G_l1.item():.4f} | "
                      f"G_ssim: {loss_G_ssim.item():.4f} | G_lpips: {loss_G_lpips.item():.4f}")

        print(f"[Epoch {epoch} Summary] "
              f"Avg D_loss: {d_loss_total/len(loader):.4f} | Avg G_loss: {g_loss_total/len(loader):.4f}")

        eval_night, eval_day = next(iter(val_loader))
        eval_night, eval_day = eval_night.to(device), eval_day.to(device)
        with torch.no_grad():
            fake_eval = G(eval_night)
        save_image(torch.cat([denorm(eval_night[:8]), denorm(fake_eval[:8]), denorm(eval_day[:8])], dim=0),
                   os.path.join(out_dir, f"epoch_{epoch:03d}_samples.png"), nrow=8)

        if epoch % eval_every == 0 or epoch == epochs:
            current_fid = calculate_fid(G, val_loader, device, epoch, out_dir)
            current_lpips = calculate_lpips(G, val_loader, loss_fn_lpips, device, epoch)
            
            current_lpips = calculate_lpips(G, val_loader, loss_fn_lpips, device, epoch)
            
            if current_fid < best_fid:
                best_fid = current_fid
                print(f"✨ New best FID: {best_fid:.4f}. Saving FID-best model. ✨")
                torch.save(G.state_dict(), os.path.join(out_dir, "generator_best_fid.pth"))
            
            if current_lpips < best_lpips:
                best_lpips = current_lpips
                print(f"✨ New best LPIPS: {best_lpips:.4f}. Saving LPIPS-best model. ✨")
                torch.save(G.state_dict(), os.path.join(out_dir, "generator_best_lpips.pth"))
        
        # Step the schedulers at the end of the epoch
        scheduler_G.step()
        scheduler_D.step()

    print("\n--- Training finished ---")
    print(f"Best FID score achieved: {best_fid:.4f}")
    print(f"Best LPIPS score achieved: {best_lpips:.4f}")
    print(f"Best LPIPS score achieved: {best_lpips:.4f}")
    return G, D

# ============================
# Example Run
# ============================
if __name__ == "__main__":
    # Define paths for your dataset
    NIGHT_DIR = "night2day/night"
    DAY_DIR   = "night2day/day"
    
    # Create dummy data if the dataset directories don't exist
    if not os.path.exists(NIGHT_DIR) or not os.path.exists(DAY_DIR):
        print("Creating dummy data for testing purposes...")
        os.makedirs(NIGHT_DIR, exist_ok=True)
        os.makedirs(DAY_DIR, exist_ok=True)
        for i in range(50):
        for i in range(50):
            dummy_night = Image.new('RGB', (256, 256), color = (10+i, 20+i, 40+i%20))
            dummy_day = Image.new('RGB', (256, 256), color = (150-i, 180-i, 200-i%30))
            dummy_night.save(os.path.join(NIGHT_DIR, f"{i:03d}.png"))
            dummy_day.save(os.path.join(DAY_DIR, f"{i:03d}.png"))

    # Start the training process
    train(NIGHT_DIR, DAY_DIR, out_dir="./pix2pix_enhanced_runs", epochs=200, 
          batch_size=8, eval_every=5, debug_subset_size=4000)
