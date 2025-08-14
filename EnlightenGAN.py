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
    
#discriminator is the critic of the output we are getting out of the generator
#how does it judge the output of the generator
#1. look for basic patterns like edges, textures, and shapes
#2. It then combines the edges and colors to recognize textures like roughness of a tree barks for example or smoothness of the sky
#3. i tthen combiines the textures to recognize parts of the image 
#4. After all these, it makes the judgement and scores betweenn 0 and 1
# where a score of 0 means the image is fake and a score of 1 means the image is real
# our discriminator is a convolutional neural network (CNN) that takes an image as input and outputs a single score indicating whether the image is real or fake

class Discriminator(nn.Module):
    def __init__(self, in_channels=3): # the init method is the blueprint, it defines the architecture of the discriminator
        super(Discriminator, self).__init__() #this line initializzes the underlying "nn.Module" functionality

       
        def _discriminator_block(in_channels, out_channels, use_batch_norm=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
                #nn.Conv2d is a 2D convolutional layer, which is basically basically like a slide that moves over the image to extract features
                #in_channel is the number of channels in the input image, for the first layer this is 3 RGB
                #out_channel: The number of patterns the layer should learn to detect
                #kernel_size it the size of the window that slides over the image, here it is 4x4
                #stride=2, which means the window moves 2 pixels at a time, effectively down-sampling the image
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
                # Batch normalization helps stabilize the learning process (#not very clear to me)

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            #this is an activation functions, after the convolution finds a pattern,our activation function decides if that pattern is strong or important enough to be pased on
            return nn.Sequential(*layers)

        self.main = nn.Sequential(
            _discriminator_block(in_channels, 64, use_batch_norm=False), # Output: 64 x 128 x 128
            _discriminator_block(64, 128),                              # Output: 128 x 64 x 64
            _discriminator_block(128, 256),                             # Output: 256 x 32 x 32
            _discriminator_block(256, 512),                             # Output: 512 x 16 x 16
            
            # Final layer to produce a single output value (the "score")
            # The output here is a feature map of "scores" for different patches
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)      # Output: 1 x 15 x 15
        )

    def forward(self, x):
        # Pass the image 'x' through the layers
        x = self.main(x)
        
        # Apply the sigmoid function to get a final probability score between 0 and 1
        return torch.sigmoid(x)


# workflow
#Input: A 3 x 256 x 256 image enters the forward method.

#Block 1: It passes through _discriminator_block(3, 64). The Conv2d (with stride=2) transforms it into a 64 x 128 x 128 feature map. We now have 64 maps of basic patterns, and the image size is halved.

#Block 2: The 64 x 128 x 128 map goes into _discriminator_block(64, 128). It's transformed into a 128 x 64 x 64 map. The patterns are becoming more complex, and the image is smaller.

#Block 3 & 4: This continues, shrinking the spatial size and increasing the feature depth. The network is moving from "what" the patterns are to "where" they are in a more abstract sense. The output is 512 x 16 x 16.

#Final Conv Layer: The nn.Conv2d(512, 1, ...) layer takes the 512 complex feature maps and condenses all that information into a single channel, giving a 1 x 15 x 15 map.

#Sigmoid & Return: This 1 x 15 x 15 map is a grid of scores. It means the discriminator isn't giving just one score for the whole image, but a score for every overlapping patch. This powerful technique (called PatchGAN) forces the generator to make every single part of the image look real. The sigmoid function then converts all these raw scores into probabilities.