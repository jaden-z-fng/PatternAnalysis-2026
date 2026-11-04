import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample (maxpool) then DoubleConv."""
 
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
 
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """Upsample, concatenate the skip connection, then DoubleConv."""
 
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)


    def forward(self, x, skip):
        x = self.up(x)
        # pad if sizes differ by a pixel (odd input sizes)
        dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))

class UNet2D(nn.Module):
    """Standard 4-level U-Net. Outputs raw logits of shape (B, num_classes, H, W)."""

    def __init__(self, n_channels, num_classes=6, base=64):
        super().__init__()
        b = base
        self.inc = DoubleConv(n_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)
        self.up1 = Up(b * 16, b * 8)
        self.up2 = Up(b * 8, b * 4)
        self.up3 = Up(b * 4, b * 2)
        self.up4 = Up(b * 2, b)
        self.out = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

class ContextModule(nn.Module):
    def __init__(self, ch, p=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(ch), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.Dropout2d(p),
            nn.InstanceNorm2d(ch), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


def conv_in_act(in_ch, out_ch, k=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.01, inplace=True),
    )


class ImprovedDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = conv_in_act(in_ch, out_ch, 3, stride=2)
        self.ctx = ContextModule(out_ch)

    def forward(self, x):
        return self.ctx(self.down(x))


class ImprovedUp(nn.Module):
    def __init__(self, in_ch, out_ch, final=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv_in_act(in_ch, out_ch, 3),
        )
        if final: 
            self.loc = conv_in_act(out_ch * 2, out_ch * 2, 3)
        else:
            self.loc = nn.Sequential(
                conv_in_act(out_ch * 2, out_ch * 2, 3),
                conv_in_act(out_ch * 2, out_ch, 1),
            )

    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.loc(torch.cat([skip, x], dim=1))


class ImprovedUNet2D(nn.Module):
    def __init__(self, n_channels, num_classes=6, base=16):
        super().__init__()
        b = base
        self.inc = nn.Sequential(conv_in_act(n_channels, b, 3), ContextModule(b))
        self.down1 = ImprovedDown(b, b * 2)
        self.down2 = ImprovedDown(b * 2, b * 4)
        self.down3 = ImprovedDown(b * 4, b * 8)
        self.down4 = ImprovedDown(b * 8, b * 16)

        self.up1 = ImprovedUp(b * 16, b * 8)
        self.up2 = ImprovedUp(b * 8, b * 4)
        self.up3 = ImprovedUp(b * 4, b * 2)
        self.up4 = ImprovedUp(b * 2, b, final=True)


        self.seg2 = nn.Conv2d(b * 4, num_classes, 1)  # from up2
        self.seg3 = nn.Conv2d(b * 2, num_classes, 1)  # from up3
        self.seg4 = nn.Conv2d(b * 2, num_classes, 1)  # from up4

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d = self.up1(x5, x4)
        d2 = self.up2(d, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        s = self.seg2(d2)
        s = F.interpolate(s, size=d3.shape[2:], mode="nearest") + self.seg3(d3)
        s = F.interpolate(s, size=d4.shape[2:], mode="nearest") + self.seg4(d4)
        return s