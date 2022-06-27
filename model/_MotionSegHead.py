import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def corr_dis(fmap_obj, fmap_map):
    n, c, ht, wd = fmap_obj.shape
    _, _, ht2, wd2 = fmap_map.shape
    fmap_obj = fmap_obj.view(n, c, ht*wd).float()
    fmap_map = fmap_map.view(n, c, -1).float()
    norm_fmap_obj = torch.norm(fmap_obj, p=2, dim=1)    #[b, 9]
    norm_fmap_map = torch.norm(fmap_map, p=2, dim=1)    #[b, 25]

    # 欧式距离
    a = norm_fmap_obj.view(n, 1, ht, wd)**2
    a2 = a.repeat(1, wd2*ht2, 1, 1)
    b = norm_fmap_map.view(n, -1, 1, 1)**2
    b2 = b.repeat(1, 1, wd, ht)
    ab = torch.matmul(fmap_obj.transpose(1,2), fmap_map)
    ab = ab.view(n, ht, wd, -1)
    ab = ab.permute([0,3,1,2])
    corr = a2 + b2 - 2 * ab

    # 夹角余弦
    scaled_fmap_obj = fmap_obj.div(norm_fmap_obj.view(n, 1, -1).add(torch.tensor(1)))
    scaled_fmap_map = fmap_map.div(norm_fmap_map.view(n, 1, -1).add(torch.tensor(1)))
    cos = torch.matmul(scaled_fmap_obj.transpose(1,2), scaled_fmap_map)
    cos = cos.view(n, ht, wd, -1)
    cos = cos.permute([0,3,1,2])
    cos = cos.div((1 + torch.sqrt(torch.tensor(c).float())))   #统计学根号n

    # 通道标准化
    corr_norm = corr.view(n, ht2*wd2, -1).clone()
    mean = torch.mean(corr_norm, 2).unsqueeze(2)
    std = torch.std(corr_norm, 2).unsqueeze(2)
    corr_norm.sub_(mean).div_(std+1)
    corr_norm = corr_norm.sum(1)

    # 夹角标准化
    cos_norm = cos.view(n, ht2*wd2, -1).clone()
    mean = torch.mean(cos_norm, 2).unsqueeze(2)
    std = torch.std(cos_norm, 2).unsqueeze(2)
    cos_norm.sub_(mean).div_(std+1)
    cos_norm = cos_norm.sum(1)
    return corr, cos, corr_norm, cos_norm

class MothinSegHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, area1, area2):
        pass
    def export(self):
        pass


#获取模型
from Homo import Homo_cnn
model_cnn = Homo_cnn()
path = r"E:\codes\220311-motionDetection\weights\model_Train_Homo_and_save_bs32_96.pkl"
from utils.toCPP import updata_adaptive
temp_states = torch.load(path)['state_dict']
model_cnn = updata_adaptive(model_cnn, temp_states)

img_t0 = cv2.imread("model/067.png")
img_t1 = cv2.imread("model/068.png")
gt = cv2.imread("model/gt067.png")
b1,g1,r1 = cv2.split(img_t0)
b2,g2,r2 = cv2.split(img_t1)

def process(img_t0, img_t1):
    img_t0 = torch.tensor(img_t0)[None, None]
    img_t1 = torch.tensor(img_t1)[None, None]
    img_t0 = (img_t0 / 127.5) - 1
    img_t1 = (img_t1 / 127.5) - 1
    return img_t0, img_t1

b1, b2 = process(b1, b2)
g1, g2 = process(g1, g2)
r1, r2 = process(r1, r2)


b1_feas1, b1_feas2, b1_feas3, b1_feas4, b1_feas5 = model_cnn(b1, b1)
for i in range(b1_feas1.shape[1]):
    cv2.imwrite("model/"+str(i)+".png", (b1_feas1[0,i].cpu().detach().numpy()*127.5 + 127.5).astype(np.uint8))
temp1 = corr_dis(img_t0, img_t1)
