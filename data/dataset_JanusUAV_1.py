import os
import numpy as np
import cv2
import torch
import copy
import time
import random
from torchvision.transforms import functional as T_F
import torch.nn.functional as F 

from _base_dataset_generater import _Dataset_Generater_Base
from _tools import janus_uav_tools as tools

class Dataset_JanusUAV(_Dataset_Generater_Base):
    def __init__(self, dataset_path='',args={'ifDataAugment':True, 'img_size_h':640, 'img_size_w':640}) -> None:
        print('initializating Dataset_JanusUAV...')
        super().__init__(dataset_path, args)
        random.seed(int( time.time()*1000000))
        #
    
    def get_alldata_from_dataset_path(self):
        #
        data, data_piece = tools.getall_data_train(self.dataset_path)
        label, _ = tools.getall_label_train(self.dataset_path)
        tri = list(zip(data, data_piece, label ))

        data, data_piece = tools.getall_data_valid(self.dataset_path)
        label, _ = tools.getall_label_valid(self.dataset_path)
        tes = list(zip(data, data_piece, label ))
        
        split_ = int(len(tri) * 0.7)
        data_list_tri = tri[:split_]
        data_list_val = tri[split_:]
        data_list_test = tes
        #
        return data_list_tri, data_list_val, data_list_test
        
    def __getitem__(self, index):
        try:
            if self.args['trick_dataset_allissame']: 
                index = 0
        except:
            pass
        a = time.time()
        paths, _, path_gt = self.data_list[index]
        img = cv2.imread(paths, cv2.IMREAD_COLOR)
        img = torch.tensor(img/255).float().permute(2,0,1)    #hwc->chw
        target = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        ret, target = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY_INV)
        target = torch.tensor(target[None]/255).float()
        #print("读取原图", time.time()- a,"s")
        #
        if self.args['ifDataAugment']:
            img, target = self.transform(img, target)
        #
        #print("transform", time.time()- a,"s")
        img, target = self.preprocess(img, target)
        #print("preprocess", time.time()- a,"s")
        #
        return img, target

    def transform(self, img, target):
        if True:                        #随机裁剪至原始尺寸的70%~100%
            _, h, w = img.shape
            th, tw = int(h*(random.random()*0.3+0.7)), int(w*(random.random()*0.3+0.7))
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            img= img[:, y1: y1 + th,x1: x1 + tw]
            target = target[:, y1: y1 + th,x1: x1 + tw]
        if random.random() >0.5:    #高斯噪声
            temp_k = torch.rand(1)*90 + 20
            img = img + torch.randn(img.size())/temp_k
            img = torch.clip(img, 0, 1)
        if random.random() >0.5:    #水平翻转
            img = T_F.hflip(img)
            target = T_F.hflip(target)
        if random.random() >0.5:    #垂直翻转
            img = T_F.vflip(img)
            target = T_F.vflip(target)
        if random.random() >0.5:    #旋转
            temp_r = random.randrange(-30, 30)
            img = T_F.rotate(img, temp_r)
            target = T_F.rotate(target, temp_r)
        return img, target

    def preprocess(self, img, target):
        img = F.interpolate(img[None], (640, 640))[0]
        target = F.interpolate(target[None], (640, 640))[0]
        return img, target

if __name__=="__main__":
    from mypath import Path
    Dataset_generater = Dataset_JanusUAV(Path.db_root_dir('janus_uav'))
    Dataset_train = Dataset_generater.generate('train')
    Dataset_valid = Dataset_generater.generate('valid')
    Dataset_test = Dataset_generater.generate('test')
    img_t0_t4, gt_t0_t4 = Dataset_train[0]
    print(img_t0_t4.shape)
    print(gt_t0_t4.shape)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(Dataset_train, 8)
    for i, item in enumerate(dataloader):
        img, target = item

        img = (img[0] * 255).numpy().transpose(1,2,0).astype(np.uint8)
        print(img.shape)
        cv2.imwrite(f"{i}.png", img)

        target = (target[0] * 255).numpy().transpose(1,2,0).astype(np.uint8)
        cv2.imwrite(f"gt_{i}.png", target)