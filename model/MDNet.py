import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

from model.backbone.shufflenetv2 import ShuffleNetV2
from model.otherblocks.spp import SPPBlock
from model.Neck.FPN import FPNBlock
from model.Neck.PAN import PANBlock

class MDNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.adaptresize = nn.AdaptiveAvgPool2d((512, 512))
        self.channels = [116>>1, 232>>1, 464>>1]
        # backbone
        self.backbone = ShuffleNetV2()
        self.sppBlock = SPPBlock(464>>1, 464>>1)
        # neck
        self.fpnBlock = FPNBlock(*self.channels)
        self.panBlock = PANBlock(*self.channels)
        # head
        self.out80 = nn.Conv2d(self.channels[0], 2, 1, 1, 0)
        self.out40 = nn.Conv2d(self.channels[1], 2, 1, 1, 0)
        self.out20 = nn.Conv2d(self.channels[2], 2, 1, 1, 0)
        self.softmax = nn.Softmax(1)

    

    
    def forward(self, img_t1, img_t2):
        _, img_t1_out2, img_t1_out3, img_t1_out4 = self.backbone(img_t1)
        _, img_t2_out2, img_t2_out3, img_t2_out4 = self.backbone(img_t2)
        
        img_t1_out2 = img_t1_out2[:,:116>>1,:,:]
        img_t1_out3 = img_t1_out3[:,:232>>1,:,:]
        img_t1_out4 = img_t1_out4[:,:464>>1,:,:]

        img_t2_out2 = img_t2_out2[:,:116>>1,:,:]
        img_t2_out3 = img_t2_out3[:,:232>>1,:,:]
        img_t2_out4 = img_t2_out4[:,:464>>1,:,:]

        img_t1_out4 = self.sppBlock(img_t1_out4)
        img_t2_out4 = self.sppBlock(img_t2_out4)

        # FPN
        feas_80, feas_40, feas_20 = self.fpnBlock(img_t1_out2, img_t1_out3, img_t1_out4) # 
        # PAN
        feas_80, feas_40, feas_20 = self.panBlock(feas_80, feas_40, feas_20)
        # Head
        out80 = self.out80(feas_80)
        out40 = self.out40(feas_40)
        out20 = self.out20(feas_20)
        return out80, out40, out20

        
    def _initialize_weights(self):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)