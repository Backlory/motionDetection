from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import sys


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


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return x * self.relu(x + 3) / 6

class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h

if __name__ == "__main__":
    model = CoordAttention(128, 128, reduction=2).cuda()
    input = torch.rand([8,128,40,40]).cuda()
    output = model(input)
    summary(model, (128,40,40))
    print(output.shape)