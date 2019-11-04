import torch.nn.functional as F

from .unet_parts import *
import torch.nn as nn
class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet1, self).__init__()
        self.inc = inconv(n_channels, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.outc = outconv(256, n_classes)
        self.LS = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x1 = self.inc(x)
        # print('0', x1.shape)
        x2 = self.down4(x1)
        # print('1', x2.shape)
        x3 = self.up1(x2, x1)
        # print('x3', x3.shape)
        x = self.outc(x3)
        x = self.LS(x)
        # print('x', x.shape)
        return x

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.outc1 = outconv(256, n_classes)
        self.outc2 = outconv(128, n_classes)
        self.LS = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x1 = self.inc(x)#64
        x2 = self.down3(x1)#32
        x3 = self.down4(x2)#16
        x4 = self.up1(x3, x2)
        x5 = self.up2(x4, x1)
        x4 = self.outc1(x4)
        x5 = self.outc2(x5)
		
        x4 = nn.functional.interpolate(x4, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = x4 + x5
        x = self.LS(x)
        # print('x', x.shape)
        return x

class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3, self).__init__()
        self.inc = inconv(n_channels, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.outc1 = outconv(256, n_classes)
        self.outc2 = outconv(128, n_classes)
        self.outc3 = outconv(64, n_classes)
        self.LS = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x1 = self.inc(x)#128
        x2 = self.down2(x1)#64
        x3 = self.down3(x2)#32
        x4 = self.down4(x3)#16
        x5 = self.up1(x4, x3)#32+32
        x6 = self.up2(x5, x2) 
        x7 = self.up3(x6, x1)
        x5 = self.outc1(x5)
        x6 = self.outc2(x6)
        x7 = self.outc3(x7)
		
        # x5 = nn.functional.interpolate(x5, scale_factor=(4, 4), mode='bilinear', align_corners=True)
        x6 = nn.functional.interpolate(x6, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        # x = x5 + x6 + x7
        x = x6 + x7
        x = self.LS(x)
        # print('x', x.shape)
        return x

class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet4, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc1 = outconv(256, n_classes)
        self.outc2 = outconv(128, n_classes)
        self.outc3 = outconv(64, n_classes)
        self.outc4 = outconv(64, n_classes)
        self.LS = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x1 = self.inc(x)#256
        x2 = self.down1(x1)#128
        x3 = self.down2(x2)#64
        x4 = self.down3(x3)#32
        x5 = self.down4(x4)#16
        x6 = self.up1(x5, x4)#32+32
        
        x7 = self.up2(x6, x3)
        
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        x6 = self.outc1(x6)
        x7 = self.outc2(x7)
        x8 = self.outc3(x8)
        x9 = self.outc4(x9)
        # x6 = nn.functional.interpolate(x6, scale_factor=(8, 8), mode='bilinear', align_corners=True)
        # x7 = nn.functional.interpolate(x7, scale_factor=(4, 4), mode='bilinear', align_corners=True)
        x8 = nn.functional.interpolate(x8, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        # x = x6 + x7 + x8 + x9
        x = x8 + x9
        x = self.LS(x)
        # print('x', x.shape)
        return x