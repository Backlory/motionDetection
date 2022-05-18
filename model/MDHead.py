import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import kornia

from model.unet import UNet
from utils.timers import tic, toc

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        
        if norm_fn == 'group':
            num_groups = planes // 8
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class MDHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.resblock1 = ResidualBlock(2,8,"batch",2)
        self.resblock2 = ResidualBlock(8,16,"batch",2)
        self.resblock3 = ResidualBlock(16,32,"batch",2)

        self.pool1 = self.generate_avgpool(3)
        self.pool2 = self.generate_avgpool(7)
        self.pool3 = self.generate_avgpool(15)
        self.pool4 = self.generate_avgpool(31)
        self.abs = torch.abs
        
        self.flo_conv = nn.Conv2d(32*4, 32*4, 1, 1, 0)
        self.flo_norm = nn.InstanceNorm2d(32*4)
        self.flo_relu = nn.ReLU()

        self.mix_norm_flo = nn.InstanceNorm2d(32*4)
        self.mix_norm_fea = nn.InstanceNorm2d(256)
        self.mix_conv1 = nn.Conv2d(128+256, 64, 3, 1, 1)
        
        self.outlayer = nn.Conv2d(32, 2, 3, 1, 1)
        
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0)
            )
        self.upsample = nn.Upsample(scale_factor=8, mode='nearest')

        #
        self.unet = UNet(3, in_channels=2, depth=3,merge_mode='concat')

    def generate_avgpool(self, kernel_size):
        return nn.AvgPool2d(kernel_size, 1, (kernel_size-1)//2)
        #return nn.AvgPool2d(kernel_size, 1, (kernel_size-1)//2)
    
    def upsample_conv(self, out, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = out.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W) #每个像素点及其周围一圈共计9个像素点，扩展成8*8个像素点，故权重共有9*8*8*n*c*h*w个
        mask = torch.softmax(mask, dim=2)

        up_out = F.unfold(8 * out, [3,3], padding=1)  #乘以上采样倍数
        up_out = up_out.view(N, 2, 9, 1, 1, H, W)

        up_out = torch.sum(mask * up_out, dim=2) 
        up_out = up_out.permute(0, 1, 4, 2, 5, 3)
        return up_out.reshape(N, 2, 8*H, 8*W)

    def forward(self, flo, fea):
        # 光流转换
        flo_u = flo[:,0:1,:,:]
        flo_v = flo[:,1:2,:,:] #y方向
        flo_r = torch.square(flo_u) + torch.square(flo_v)
        temp_max = flo_r.max()
        flo_r = flo_r.div(temp_max)
        flo_a = torch.arctan(flo_v.div(flo_u+1e-5)) / 3.14159 + 0.5
        flo_ra = torch.cat([flo_r, flo_a], dim=1).detach()

        out = self.unet(flo_ra, fea)
        # #flo下采样
        # flo_dn = self.resblock1(flo_ra)
        # flo_dn = self.resblock2(flo_dn)
        # flo_dn = self.resblock3(flo_dn)

        # # 均值采样
        # flo_dn_p1 = self.abs(flo_dn - self.pool1(flo_dn))
        # flo_dn_p2 = self.abs(flo_dn - self.pool2(flo_dn))
        # flo_dn_p3 = self.abs(flo_dn - self.pool3(flo_dn))
        # flo_dn_p4 = self.abs(flo_dn - self.pool4(flo_dn))
        # flo_dns = torch.cat([flo_dn_p1, flo_dn_p2, flo_dn_p3, flo_dn_p4], dim=1)
        # #flo_dns = self.norm_in(flo_dns)
        # flo_dn = self.flo_conv(flo_dns)
        # flo_dn = self.flo_norm(flo_dn)
        # flo_dn = self.flo_relu(flo_dn)

        # # 特征融合
        # flo_dn = self.mix_norm_flo(flo_dn)
        # fea = self.mix_norm_fea(fea)
        # feas = torch.cat([flo_dn, fea], dim=1)
        # feas = self.mix_conv1(feas)
        
        # #二分类
        # out = self.outlayer(feas)
        # out = self.outlayer(flo_dn)
        
        # #上采样
        # upmask = .25 * self.mask(flo_dn)
        # out = self.upsample_conv(out, upmask)
        #out = self.upsample(out)
        return out

if __name__ == "__main__":
    input_flo = torch.randn([8, 2, 640, 640])
    input_fea = torch.randn([8, 256, 80, 80])

    from model.MDHead import MDHead
    model = MDHead()
    t = tic()
    for i in range(100):
        out = model(input_flo, input_fea)
    toc(t, "1", 100, False)
    print(out.shape)

