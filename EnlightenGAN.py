import torch 
import torch.nn as nn 
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__() 
        # Initial convolution block
        self.down1 = self._conv_block(3, 64) # 1st down-sampling convolution block
        self.down2 = self._conv_block(64, 128) # 2nd down-sampling convolution block
        self.down3 = self._conv_block(128, 256) # 3rd down-sampling convolution block
        self.down4 = self._conv_block(256, 512) # 4th down-sampling convolution block
        
        self.bottleneck = self._conv_block(512, 512) # Defines the central bottleneck of the U-Net
        
        # Upsampling blocks with skip connections
        self.up1 = self._up_conv_block(512 + 512, 256) # 1st up-sampling block
        self.up2 = self._up_conv_block(256 + 256, 128) # 2nd up-sampling block
        self.up3 = self._up_conv_block(128 + 128, 64) # 3rd up-sampling block
        self.final_conv = nn.Conv2d(64 + 64, 3, kernel_size=1) # final convolution layer for a RGB image

    def _conv_block(self, in_channels, out_channels): 
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # 2D convolution layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels) # Batch normalization 
        )
    
    def _up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2), # transposed convolution to up-sample the feature map
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels) # Batch normalization 
        )

    def forward(self, x): 
        d1 = self.down1(x) # First convolution block.
        d2 = self.down2(nn.MaxPool2d(2)(d1)) # Down-sample with max pooling and apply convolution.
        d3 = self.down3(nn.MaxPool2d(2)(d2)) 
        d4 = self.down4(nn.MaxPool2d(2)(d3)) 
        
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(d4)) # Apply final pooling and convolution for the bottleneck.
        
        u1 = self.up1(torch.cat([bottleneck, d4], dim=1)) # Up-sample and concatenate with d4 (skip connection).
        u2 = self.up2(torch.cat([u1, d3], dim=1)) # Up-sample and concatenate with d3.
        u3 = self.up3(torch.cat([u2, d2], dim=1)) # Up-sample and concatenate with d2.
        
        final_output = self.final_conv(torch.cat([u3, d1], dim=1)) # Concatenate with d1 and apply the final convolution.
        return torch.tanh(final_output) # Use Tanh activation to scale the output to [-1, 1].