import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import os

# ==============================================================================
# IMPORTANT: You must include the Generator's class definition in this script
# so that PyTorch knows the architecture of the model it's loading.
# You can copy it directly from your training script.
# ==============================================================================
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

# =================================
# Main Inference Function
# =================================
def translate_image(model_path, image_path, output_path, device="cuda"):
    """
    Loads a trained generator, processes an input image, and saves the translated output.
    """
    print(f"Loading model from: {model_path}")
    
    # 1. Initialize and load the trained model
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set the model to evaluation mode

    print(f"Processing image: {image_path}")

    # 2. Define the same image transformations used during training
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])
    
    # 3. Load and preprocess the input image
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(device) # Add batch dimension

    # 4. Run inference
    with torch.no_grad(): # No need to calculate gradients
        generated_tensor = model(input_tensor)
        
    # 5. Denormalize and save the output image
    # The denorm function from your script: (x + 1) / 2.0
    # torchvision.utils.save_image handles this with normalize=True
    save_image(generated_tensor, output_path, normalize=True)
    
    print(f"✨ Success! Translated image saved to: {output_path}")


if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    
    # Path to the trained generator weights.
    # Your training script saves 'generator_best_fid.pth' and 'generator_best_lpips.pth'
    # in the 'pix2pix_enhanced_runs' directory.
    MODEL_PATH = "pix2pix_enhanced_runs/generator_best_fid.pth"
    
    # Path to the night-time image you took with your camera.
    INPUT_IMAGE_PATH = "image.png" 
    
    # Path where the translated day-time image will be saved.
    OUTPUT_IMAGE_PATH = "my_day_photo_generated.png"

    # --- END OF CONFIGURATION ---
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
    elif not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image not found at '{INPUT_IMAGE_PATH}'")
    else:
        # Determine the device and run translation
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        translate_image(MODEL_PATH, INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH, device=DEVICE)