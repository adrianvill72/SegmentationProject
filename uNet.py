import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.down1 = CBR(1, 64)
        self.down2 = CBR(64, 128)
        self.down3 = CBR(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = CBR(256 + 128, 128)
        self.up2 = CBR(128 + 64, 64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        print("x1 shape:", x1.shape)
        p1 = self.pool(x1)
        print("p1 shape:", p1.shape)
        x2 = self.down2(p1)
        print("x2 shape:", x2.shape)
        p2 = self.pool(x2)
        print("p2 shape:", p2.shape)
        x3 = self.down3(p2)
        print("x3 shape:", x3.shape)
        up3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        print("up3 shape:", up3.shape)
        concat3 = torch.cat([up3, x2], dim=1)
        print("concat3 shape:", concat3.shape)
        x4 = self.up3(concat3)
        print("x4 shape:", x4.shape)
        up2 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        print("up2 shape:", up2.shape)
        concat2 = torch.cat([up2, x1], dim=1)
        print("concat2 shape:", concat2.shape)
        x5 = self.up2(concat2)
        print("x5 shape:", x5.shape)
        output = self.final_conv(x5)
        print("output shape:", output.shape)
        return output

