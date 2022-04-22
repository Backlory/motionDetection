from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import sys

from model.otherblocks.blocks import CSP1_n, CBL

class PANBlock(nn.Module):
    def __init__(self, c80, c40, c20):
        super().__init__()
        self.dn80_40 = CBL(c80, c80, 3, 2)
        self.out40 = CSP1_n(c80 + c40, c40, n_resblock=3)
        self.dn40_20 = CBL(c40, c40, 3, 2)
        self.out20 = CSP1_n(c40 + c20, c20, n_resblock=3)
    def forward(self, f80, f40, f20):
        dn80 = self.dn80_40(f80)
        out40 = torch.cat([dn80, f40], dim=1)
        out40 = self.out40(out40)

        dn40 = self.dn40_20(f40)
        out20 = torch.cat([dn40, f20], dim=1)
        out20 = self.out20(out20)

        return f80, out40, out20