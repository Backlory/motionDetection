from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import sys

class ResUnit_n(nn.Module):
    def __init__(self, c1, c2, n):
        super(ResUnit_n, self).__init__()
        self.shortcut = c1 == c2
        res_unit = nn.Sequential(
            CBL(c1, c1, k=1, s=1),
            CBL(c1, c2, k=3, s=1))
        self.res_unit_n = nn.Sequential(*[res_unit for _ in range(n)])
    def forward(self, x):
        out = self.res_unit_n(x)
        if self.shortcut:
            out.add_(x)
        return out

class CBL(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out

class CSP1_n(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, n_resblock=1):
        super(CSP1_n, self).__init__()

        c_ = c2 // 2
        self.up = nn.Sequential(
            CBL(c1, c_, k, s),
            ResUnit_n(c_, c_, n_resblock),
        )
        self.bottom = nn.Conv2d(c1, c_, 1, 1, 0)
        self.tie = nn.Sequential(
            nn.BatchNorm2d(c_ * 2),
            nn.LeakyReLU(),
            nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        total = torch.cat([self.up(x), self.bottom(x)], dim=1)
        out = self.tie(total)
        return out



class CALayer(nn.Module):  # Channel Attention (CA) Layer
    def __init__(self, in_channels, reduction=16, pool_types=['avg', 'max']):
        super().__init__()
        self.pool_list = ['avg', 'max']
        self.pool_types = pool_types
        self.in_channels = in_channels
        self.Pool = [nn.AdaptiveAvgPool2d(
            1), nn.AdaptiveMaxPool2d(1, return_indices=False)]
        self.conv_ca = nn.Sequential(
            nn.Conv2d(in_channels, in_channels //
                      reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction,
                      in_channels, 1, padding=0, bias=True)
        )

    def forward(self, x):
        for (i, pool_type) in enumerate(self.pool_types):
            pool = self.Pool[self.pool_list.index(pool_type)](x)
            channel_att_raw = self.conv_ca(pool)
            if i == 0:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw
        scale = F.sigmoid(channel_att_sum)
        return x * scale


class SALayer(nn.Module):  # Spatial Attention Layer
    def __init__(self):
        super().__init__()
        self.conv_sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid()
        )
    def forward(self, x):
        max1 = torch.max(x, 1, keepdim=True)[0]
        mean1 = torch.mean(x, dim=1, keepdim=True)
        x_compress = torch.cat( (max1, mean1), dim=1)
        scale = self.conv_sa(x_compress)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=2, pool_types=['avg', 'max']):
        super().__init__()
        self.CALayer = CALayer(
            in_channels, reduction, pool_types)
        self.SALayer = SALayer()

    def forward(self, x):
        x_out = self.CALayer(x)
        x_out = self.SALayer(x_out)
        return x_out
if __name__ == "__main__":
    model = CBAM(128, reduction=2).cuda()
    input = torch.rand([8,128,40,40]).cuda()
    output = model(input)
    summary(model, (128,40,40))
    print(output.shape)