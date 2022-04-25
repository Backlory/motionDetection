import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from thop import profile
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

class FastGridPreDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_fea80 = None
        self.hist_fea40 = None
        self.hist_fea20 = None
        self.channels = [116, 232, 464]
        # backbone
        self.backbone = ShuffleNetV2(pretrain=False)
        #self.sppBlock = SPPBlock(self.channels[2], self.channels[2])
        # neck
        self.CA_80 = CoordAttention(self.channels[0],self.channels[0],16)
        self.CA_40 = CoordAttention(self.channels[1],self.channels[1],16)
        self.CA_20 = CoordAttention(self.channels[2],self.channels[2],16)
        
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
        
        # fea process
        self.softmax = nn.Softmax(1)
        self._initialize_weights()
    
    def forward(self, img):
        # 特征提取
        feas = self.backbone(img)
        
        fea_out2 = feas[1]
        fea_out3 = feas[2]
        fea_out4 = feas[3]

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
        
        # 记录
        self.hist_fea80 = fea_out2
        self.hist_fea40 = fea_out3
        self.hist_fea20 = fea_out4

        # Head
        out80 = self.out80(feas_80)
        out40 = self.out40(feas_40)
        out20 = self.out20(feas_20)

        self.hist_out80_sfmx = self.softmax(out80)
        self.hist_out40_sfmx = self.softmax(out40)
        self.hist_out20_sfmx = self.softmax(out20)

        return out80, out40, out20
        
    def get_grid_feature(self, hist_grid):
        '''
        根据历史网格，获取当前网格，输出对应的特征向量组。用于后续检测用。
        执行匈牙利匹配获取网格区域匹配，然后计算并集的最小外接矩形，作为运动检测区域？
        '''
        #获取当前关注网格区域
        out_grid = self.hist_out40_sfmx + self.upsample(self.hist_out20_sfmx)
        out_grid = self.hist_out80_sfmx + self.upsample(out_grid)
        out_grid = out_grid / 3
        
        #获取当前区域对应的特征组金字塔组
        self.hist_fea80
        self.hist_fea40
        self.hist_fea20
        return out_grid
        
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
        url = r"temp/model_Train_FastGridPreDetector_and_save_bs4_92.pkl"
        pretrained_state_dict = torch.load(url)["state_dict"]
        if True:
            for k,v in pretrained_state_dict.items():
                if (k in self.state_dict().keys()):
                    pass
                    print((k in self.state_dict().keys()), "==", k,"=>",v.shape)
                else:
                    print((k in self.state_dict().keys()), "==", k,"=>",v.shape)

        self.load_state_dict(pretrained_state_dict, strict=False)

if __name__ == "__main__":

    model = FastGridPreDetector().cuda()
    temp = torch.load("temp/model_Train_FastGridPreDetector_and_save_bs4_92.pkl")
    model.load_state_dict(temp["state_dict"], strict=False)
    
    model.eval()
    summary(model, (3,640,640))

    input = torch.rand([1,3,640,640]).cuda()
    #flops, params = profile(model, (input,))
    #print("flops = ", flops)
    #print("params = ", params)
    output = model(input)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)