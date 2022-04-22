from turtle import forward
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.img_display import img_square

class CorrBlock(nn.Module):
    def __init__(self, radius=9):
        super().__init__()
        self.radius = radius
        self.padding_size = (self.radius, self.radius, self.radius, self.radius)
    def forward(self, img, img_map):
        with torch.no_grad():
            device = img.device
            n,c,h,w = img.shape
            out_cos = torch.ones([n, self.radius**2, h, w ]).to(device)
            out_diss = torch.zeros([n, self.radius**2, h, w ]).to(device)
            
            # 夹角余弦
            norm_img =     torch.norm(img, p=2, dim=1)  # 归一化
            norm_img_map = torch.norm(img_map, p=2, dim=1)
            scaled_img =     img.div_(   norm_img.view(n, 1, h, w).add(torch.tensor(0.00001)))
            scaled_img_map = img_map.div(norm_img_map.view(n, 1, h, w).add(torch.tensor(0.00001)))
            scaled_img =     torch.clamp(scaled_img, -100, 100)
            scaled_img_map = torch.clamp(scaled_img_map, -100, 100)
            
            scaled_img_map = F.pad(scaled_img_map, self.padding_size, "replicate")
            for idx in range(self.radius ** 2):
                i = idx // self.radius
                j = idx % self.radius
                scaled_img_submap = scaled_img_map[:, :, i:i+h, j:j+w].clone()
                scaled_img_submap.mul_(scaled_img)
                out_cos[:, idx, :, :].sub_(scaled_img_submap.sum(1))   # 1 - cos
            # 点乘
            img_map = F.pad(img_map, self.padding_size, "replicate")
            for idx in range(self.radius ** 2):
                i = idx // self.radius
                j = idx % self.radius
                img_submap = img_map[:, :, i:i+h, j:j+w].clone()
                img_submap.mul_(img)
                out_diss[:, idx, :, :] = img_submap.sum(1)   # 点乘
                del img_submap
        out_cos = out_cos.detach()
        out_diss = out_diss.detach()
        return out_cos, out_diss
