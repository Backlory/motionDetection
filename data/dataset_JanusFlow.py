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

class Dataset_JanusFlow(_Dataset_Generater_Base):
    def __init__(self, dataset_path='',args={'img_size_h':640, 'img_size_w':640}) -> None:
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
        paths, _, path_gt = self.data_list[index]
        img = cv2.imread(paths, cv2.IMREAD_COLOR)
        
        # 从掩膜中获取对象
        target = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        ret, target = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY_INV)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(target)
        stats = np.array(stats)
        stats[i,4]
        for i in range(1, nlabels):
            regions_size = stats[i,4]
        


        img = torch.tensor(img/255).float().permute(2,0,1)    #hwc->chw
        img = img*2 - 1   # [-1, +1]
        
        target = torch.tensor(target/255).float().permute(2,0,1)
        #
        return img, img_warpped, target


if __name__=="__main__":
    from mypath import Path

    Dataset_generater = Dataset_JanusFlow(Path.db_root_dir('janus_uav'))
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

        print(img.shape)
        img = (img[0] * 255).numpy().transpose(1,2,0).astype(np.uint8)
        cv2.imwrite(f"1.png", img)

        target = (target[0] * 255).numpy().transpose(1,2,0).astype(np.uint8)
        cv2.imwrite(f"1.png", target)