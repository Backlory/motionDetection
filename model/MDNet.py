import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import os
import sys
sys.path.append("model")
try:
    from model.backbone.shufflenetv2 import ShuffleNetV2
    from model.otherblocks.spp import SPPBlock
    from model.otherblocks.blocks import CSP1_n, CBL
    from model.otherblocks.attention import CoordAttention
except:
    from backbone.shufflenetv2 import ShuffleNetV2
    from otherblocks.spp import SPPBlock
    from otherblocks.blocks import CSP1_n, CBL
    from otherblocks.attention import CoordAttention

class MDNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.adaptresize = nn.AdaptiveAvgPool2d((512, 512))
        self.channels = [116, 232, 464]
        # backbone
        self.backbone = ShuffleNetV2()
        self.sppBlock = SPPBlock(464, 464)
        self.CA_80 = CoordAttention(self.channels[0],self.channels[0],16)
        self.CA_40 = CoordAttention(self.channels[1],self.channels[1],16)
        self.CA_20 = CoordAttention(self.channels[2],self.channels[2],16)
        # neck
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.FPN40out = nn.Sequential(
            CSP1_n(self.channels[2]+self.channels[1], self.channels[1], n_resblock=3),
            CBL(self.channels[1], self.channels[1], 1, 1)
        )
        self.FPN80out = nn.Sequential(
            CSP1_n(self.channels[1]+self.channels[0], self.channels[0], n_resblock=3),
            CBL(self.channels[0], self.channels[0], 1, 1)
        )
        # head
        self.out80 = nn.Conv2d(self.channels[0], 2, 1, 1, 0)
        self.out40 = nn.Conv2d(self.channels[1], 2, 1, 1, 0)
        self.out20 = nn.Conv2d(self.channels[2], 2, 1, 1, 0)
        self.softmax = nn.Softmax(1)

    

    
    def forward(self, img):
        _, fea_out2, fea_out3, fea_out4 = self.backbone(img)
        fea_out2 = fea_out2 + self.CA_80(fea_out2)
        fea_out3 = fea_out3 + self.CA_40(fea_out3)
        fea_out4 = fea_out4 + self.CA_20(fea_out4)
        #fea_out4 = self.sppBlock(fea_out4)
        # FPN
        feas_20 = fea_out4
        fea_out4_up = self.upsample(fea_out4)
        feas_40 = torch.cat([fea_out3, fea_out4_up], dim=1)
        feas_40 = self.FPN40out(feas_40)
        
        fea_out3_up = self.upsample(fea_out3)
        feas_80 = torch.cat([fea_out2, fea_out3_up], dim=1)
        feas_80 = self.FPN80out(feas_80)
        # PAN
        
        #
        #feas_80 = torch.cat([], dim=1)
        #feas_40 = torch.cat([], dim=1)
        #feas_20 = torch.cat([], dim=1)
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

if __name__ == "__main__":

    model = MDNet().cuda()
    input = torch.rand([8,3,640,640]).cuda()
    output = model(input)
    summary(model, (3,640,640))
    print(output.shape)