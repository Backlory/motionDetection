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



