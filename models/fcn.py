import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCN(nn.Module):

    def __init__(self):
        super().__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # CLASSIFIER
        self.classifier = nn.Conv2d(in_channels=64, 
                                    out_channels=1,
                                    kernel_size=1)
        
    def forward(self, x):

        # encoder
        x = self.encoder(x)

        # pixel-wise prediction
        x = self.classifier(x)

        # upsampling
        x = F.interpolate(x,
                          scale_factor=4,
                          mode='bilinear',
                          align_corners=False)
        
        return x
