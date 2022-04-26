from turtle import forward
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
#from utils.img_display import img_square

class CorrBlock():
    def __init__(self, radius=4):
        super().__init__()
        self.radius = radius
        self.d = (2*self.radius+1)
        #self.padding_size = (self.radius, self.radius, self.radius, self.radius)
    def __call__(self, img, img_map):
        with torch.no_grad():
            device = img.device
            n,c,h,w = img.shape
            _,_,h2,w2 = img_map.shape
            assert(h2 == h+2*self.radius)
            assert(w2 == w+2*self.radius)
            out_cos = torch.zeros([n, self.d**2, h, w ]).to(device)
            out_diss = torch.zeros([n, self.d**2, h, w ]).to(device)
            
            # 夹角余弦
            norm_img =     torch.norm(img, p=2, dim=1)  # 归一化
            norm_img_map = torch.norm(img_map, p=2, dim=1)
            scaled_img =     img.div(   norm_img.view(n, 1, h, w).add(torch.tensor(0.00001)))
            scaled_img_map = img_map.div(norm_img_map.view(n, 1, h2, w2).add(torch.tensor(0.00001)))
            scaled_img =     torch.clamp(scaled_img, -100, 100)
            scaled_img_map = torch.clamp(scaled_img_map, -100, 100)
            
            #scaled_img_map = F.pad(scaled_img_map, self.padding_size, "replicate")
            for idx in range(self.d ** 2):
                i = idx // self.d
                j = idx % self.d
                scaled_img_submap = scaled_img_map[:, :, i:i+h, j:j+w].clone()
                scaled_img_submap.mul_(scaled_img)
                out_cos[:, idx, :, :] = scaled_img_submap.sum(1)   # 1 - cos
                del scaled_img_submap
            # 点乘
            #img_map = F.pad(img_map, self.padding_size, "replicate")
            for idx in range(self.d ** 2):
                i = idx // self.d
                j = idx % self.d
                img_submap = img_map[:, :, i:i+h, j:j+w].clone()
                img_submap.mul_(img)
                out_diss[:, idx, :, :] = img_submap.sum(1)   # 点乘
                del img_submap
        out_cos = out_cos.detach().view(n, self.d, self.d, h, w)
        out_diss = out_diss.detach().view(n, self.d, self.d, h, w)
        return out_cos, out_diss

if __name__ == "__main__":
    model = CorrBlock(4)
    all = torch.randn([1, 40, 64, 64])
    img_t0 = all[:,:, 7:7+43, 7:7+43]
    img_t1 = all[:,:, 10-4:10+43+4, 10-4:10+43+4]
    out = model(img_t0, img_t1)
    
    print(out)

    _,_,h,w = img_t0.shape
    for i in range(h):
        for j in range(w):
            temp = (((out[0][0,:,:,i,j] + 1)/2)**2 *255).numpy().astype(np.uint8)
            temp = cv2.resize(temp, (300,300), interpolation=cv2.INTER_NEAREST)
            #cv2.imshow(f"i={i},j={j}", temp)
            cv2.imshow(f"i", temp)
            print(f"i={i},j={j}")
            cv2.waitKey(1)