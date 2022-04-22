from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import sys

from model.otherblocks.blocks import CSP1_n, CBL

class FPNBlock(nn.Module):
    def __init__(self, c80, c40, c20):
        super().__init__()
        self.up_20_40 = nn.Upsample(scale_factor=2, mode='nearest')
        self.outBlock40 = nn.Sequential(  
            CSP1_n(c20 + c40, c40, n_resblock=3),
            CBL(c40, c40, 1, 1)
        )
        self.up_40_80 = nn.Upsample(scale_factor=2, mode='nearest')
        self.outBlock80 = nn.Sequential(            
            CSP1_n(c40 + c80, c80, n_resblock=3)
        )

    def forward(self, f80, f40, f20):
        up_20 = self.up_20_40(f20)
        out40 = torch.cat([f40, up_20], dim=1)
        out40 = self.outBlock40(out40)

        up_40 = self.up_40_80(f40)
        out80 = torch.cat([f80, up_40], dim=1)
        out80 = self.outBlock80(out80)
        
        return out80, out40, f20


if __name__ == "__main__":
    model = FPNBlock(128, 280, 500)
    input1 = torch.rand([8,128,80,80])
    input2 = torch.rand([8,280,40,40])
    input3 = torch.rand([8,500,20,20])
    out = model(input1, input2, input3)
    print(out.shape)