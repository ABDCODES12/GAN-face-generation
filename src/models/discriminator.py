import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        
        def conv_block(in_channels, out_channels, use_spectral_norm):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            return conv
        
        layers = [
            # Input: 3 x 64 x 64
            conv_block(3, 64, use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv_block(64, 128, use_spectral_norm),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv_block(128, 256, use_spectral_norm),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv_block(256, 512, use_spectral_norm),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 1 x 1 x 1
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)