import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=atrous_rates[0],
                dilation=atrous_rates[0],
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=atrous_rates[1],
                dilation=atrous_rates[1],
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=atrous_rates[2],
                dilation=atrous_rates[2],
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * 5,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        global_feature = self.global_pool(x)

        global_feature = F.interpolate(global_feature,
                                       size=x.shape[2:],
                                       mode='bilinear',
                                       align_corners=False)
        
        features = torch.cat(
            [branch1, branch2, branch3, branch4, global_feature], dim=1
        )

        output = self.project(features)

        return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.block1(x)
        x = self.pool(x1)

        x2 = self.block2(x)
        x = self.pool(x2)

        x3 = self.block3(x)
        x = self.pool(x3)

        x4 = self.block4(x)

        return x1, x2, x3, x4

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.low_level_projection = nn.Sequential(
            nn.Conv2d(
                128,
                48,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                256 + 48,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, low_level):
        x = F.interpolate(
            x,
            size=low_level.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        low_level = self.low_level_projection(low_level)

        x = torch.cat(
            [x, low_level], dim=1
        )

        x = self.conv(x)

        return x
    
class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.aspp = ASPP(
            in_channels=512,
            out_channels=256,
            atrous_rates=(6, 12, 18)
        )

        self.decoder = Decoder()

        self.classifier = nn.Conv2d(
            256,
            1,
            kernel_size=1
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)

        x = self.aspp(x4)

        x = self.decoder(x, x2)

        x = F.interpolate(
            x,
            size=(512,512),
            mode='bilinear',
            align_corners=False
        )

        x = self.classifier(x)

        return x
        
