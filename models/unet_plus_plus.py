import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class UNetPlusPlus(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # Encoder
        # ====================================================

        self.conv00 = DoubleConv(1, 32)

        self.conv10 = DoubleConv(32, 64)

        self.conv20 = DoubleConv(64, 128)

        self.conv30 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)


        # ====================================================
        # Decoder / Nested blocks
        # ====================================================

        # X1,1
        self.up21 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv11 = DoubleConv(
            128,
            64
        )


        # X0,1
        self.up10 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.conv01 = DoubleConv(
            64,
            32
        )


        # X2,1
        self.up32 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.conv21 = DoubleConv(
            256,
            128
        )


        # X1,2
        self.up12 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv12 = DoubleConv(
            64 + 64 + 64,
            64
        )


        # X0,2
        self.up01 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.conv02 = DoubleConv(
            32 + 32 + 32,
            32
        )


        # ====================================================
        # Output
        # ====================================================

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )


    def forward(self, x):

        # ====================================================
        # Encoder
        # ====================================================

        x00 = self.conv00(x)

        x10 = self.conv10(
            self.pool(x00)
        )

        x20 = self.conv20(
            self.pool(x10)
        )

        x30 = self.conv30(
            self.pool(x20)
        )


        # ====================================================
        # First nested level
        # ====================================================

        x21 = self.conv21(
            torch.cat(
                [
                    x20,
                    self.up32(x30)
                ],
                dim=1
            )
        )


        x11 = self.conv11(
            torch.cat(
                [
                    x10,
                    self.up21(x20)
                ],
                dim=1
            )
        )


        x01 = self.conv01(
            torch.cat(
                [
                    x00,
                    self.up10(x10)
                ],
                dim=1
            )
        )


        # ====================================================
        # Second nested level
        # ====================================================

        x12 = self.conv12(
            torch.cat(
                [
                    x10,
                    x11,
                    self.up12(x21)
                ],
                dim=1
            )
        )


        x02 = self.conv02(
            torch.cat(
                [
                    x00,
                    x01,
                    self.up01(x11)
                ],
                dim=1
            )
        )


        # ====================================================
        # Output
        # ====================================================

        return self.final(x02)