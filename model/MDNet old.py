#这里带了一部分运动感知代码。
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
#from model.Corr.Corr import CorrBlock
from model.Neck.FPN import FPNBlock
from model.Neck.PAN import PANBlock

class FastGridPreDetector(nn.Module):
    def __init__(self):
        super().__init__()
        #self.adaptresize = nn.AdaptiveAvgPool2d((512, 512))
        self.channels = [116>>1, 232>>1, 464>>1]
        # backbone
        self.backbone = ShuffleNetV2()
        self.sppBlock = SPPBlock(464>>1, 464>>1)

        #self.corrBlock80 = CorrBlock(radius=9)
        #self.corrBlock40 = CorrBlock(radius=7)
        #self.corrBlock20 = CorrBlock(radius=5)
        # neck
        self.fpnBlock = FPNBlock(*self.channels)
        self.panBlock = PANBlock(*self.channels)
        # head
        self.out80 = nn.Conv2d(self.channels[0], 2, 1, 1, 0)
        self.out40 = nn.Conv2d(self.channels[1], 2, 1, 1, 0)
        self.out20 = nn.Conv2d(self.channels[2], 2, 1, 1, 0)
        self.softmax = nn.Softmax(1)

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

        '''
        in1 = img_t1_out4
        in2 = img_t2_out4
        radius = 7
        out_cos, out_diss = self.corrBlock(in1, in2, radius)
        n,c,h,w = out_cos.shape
        k = int(img_t1.shape[2] / h)
        temp1 = out_cos.view(n, radius, radius, h, w)
        temp2 = out_diss.view(n, radius, radius, h, w)
        temp1 = (temp1-temp1.min())/(temp1.max()-temp1.min()) * 255
        temp2 = (temp2-temp2.min())/(temp2.max()-temp2.min()) * 255
        cv2.namedWindow("1", cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("1", 7*30*5, 7*30)
        for i in range(int(h*0.3), int(h*0.9),int(max(h*0.01, 1))):
            for j in range(int(w*0.3), int(w*0.9),int(max(w*0.01, 1))):
                watcher = []
                print(i, j)
                watcher.append(img_t1[0, :, k*(i-radius>>1):k*(i+radius>>1+1), k*(j-radius>>1):k*(j+radius>>1+1)].clone())
                watcher.append(img_t2[0, :, k*(i-radius>>1):k*(i+radius>>1+1), k*(j-radius>>1):k*(j+radius>>1+1)].clone())
                watcher.append(temp1[0:1,:,:,i,j].clone())
                watcher.append(None)
                watcher.append(temp2[0:1,:,:,i,j].clone())
                temp = img_square(watcher, 1, 5)
                temp = cv2.resize(temp, (7*30*5,7*30), interpolation=cv2.INTER_AREA)
                cv2.imshow("1", temp)
                cv2.waitKey(1)'''
        '''if self.MovDetection:
            out_cos2, out_diss2 = self.corrBlock80(img_t1_out2, img_t2_out2)    #[n, 81, 80, 80]
            out_cos3, out_diss3 = self.corrBlock40(img_t1_out3, img_t2_out3)    #[n, 49, 40, 40]
            out_cos4, out_diss4 = self.corrBlock20(img_t1_out4, img_t2_out4)    #[n, 25, 40, 40]
            feas_80 = torch.cat([img_t1_out2, out_cos2, out_diss2], dim=1)  #81*2 + 116 = 278
            feas_40 = torch.cat([img_t1_out3, out_cos3, out_diss3], dim=1)  #49*2 + 232 = 330
            feas_20 = torch.cat([img_t1_out4, out_cos4, out_diss4], dim=1)  #25*2 + 464 = 514
        else:
            feas_80 = img_t1_out2
            feas_40 = img_t1_out3
            feas_20 = img_t1_out4'''
        # FPN
        feas_80, feas_40, feas_20 = self.fpnBlock(img_t1_out2, img_t1_out3, img_t1_out4) # 
        # PAN
        feas_80, feas_40, feas_20 = self.panBlock(feas_80, feas_40, feas_20)
        # Head
        out80 = self.out80(feas_80)
        out40 = self.out40(feas_40)
        out20 = self.out20(feas_20)
        return out80, out40, out20