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
