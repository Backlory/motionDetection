from pyexpat import model
import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

from utils.img_display import img_square
from model.FastGridPreDetector import FastGridPreDetector
from model.backbone.shufflenetv2 import ShuffleNetV2 
from model.Corr.Corr import CorrBlock
# 模型
# https://jishuin.proginn.com/p/763bfbd59ac5

# 数据
def loadimg(filedir):
    img = cv2.imread(filedir)
    #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

    #img = cv2.medianBlur(img, 3)
    #img = cv2.GaussianBlur(img, (7,7), 1)

    #temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #temp = cv2.split(temp)
    #temp[0] = cv2.medianBlur(temp[0], 3)
    #img = img // 16 * 16
    #temp = cv2.merge(temp)
    #img = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
    #cv2.imwrite("1.png", img)
    #img = cv2.GaussianBlur(img, (3,3), 0)
    input = torch.Tensor(img.transpose(2,0,1) / 255).float()[None].contiguous()
    return input

def ten2bboxs(ten):
    assert{False}
    assert(ten.shape[1] == 2)   #[n, 2, h, w]
    bboxes = ten.detach().cpu().numpy()
    return bboxes

def vis_bbox(img, bboxes):
    h_img,w_img,_ = img.shape
    for bbox in bboxes:
        x,y,w,h = bbox
        w = w * w_img
        h = h * h_img
        x = x * w_img - w/2
        y = y * h_img - h/2
        
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.imshow("1", img)
    cv2.waitKey(0)
    return img

#======================================================================
def main():
    device = torch.device("cuda:0")
    print("model...")
    #model = FastGridPreDetector().to(device)
    model = ShuffleNetV2(pretrain=True).to(device)
    mycorr = CorrBlock(radius=4)
    
    print("criterion...")
    criterion = nn.MSELoss()

    print("optimizer...")
    optimizer = optim.Adam(model.parameters(), 1e-4)

    try:
        assert(device.type == "cuda")
        from torch.cuda.amp import GradScaler, autocast
    except:
        print("no cuda device available, so amp will be forbidden.")
        from utils.cuda import GradScaler, autocast
    scaler = GradScaler(enabled=True)

    print("train...")
    if True:
        
        len_all = len(os.listdir("E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_1/video/"))
        for i in range(len_all * 1000-1000):
            i = i % (len_all-1)
            img_t0 = loadimg(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_1/video/{str(i).zfill(3)}.png").to(device)
            img_t1 = loadimg(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_1/video/{str(i+1).zfill(3)}.png").to(device)
            
            gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_1/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
            gt = 255 - gt
            _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
            #temp = nn.AdaptiveMaxPool2d(80)(torch.Tensor(gt[None,None])).numpy()[0,0];cv2.imwrite("1.png", temp);print(temp.shape)
            target = cv2.merge([gt, 255-gt])
            target = torch.Tensor(target.transpose(2,0,1) / 255).float()[None].contiguous().to(device)
            
            outputs_0 = model(img_t0)
            outputs_1 = model(img_t1)
            '''
            outputs[0].shape
            torch.Size([1, 24, 160, 160])
            outputs[1].shape
            torch.Size([1, 116, 80, 80])
            outputs[2].shape
            torch.Size([1, 232, 40, 40])
            outputs[3].shape
            torch.Size([1, 464, 20, 20])
            '''
            #temp = mycorr()
            
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
            #print(f"sample {i}", ", loss={:.5f}, loss_dice={:.5f}, loss_focal={:.5f}".format(loss, loss_dice, loss_focal))
        
            watcher =  [img_t0[0], gt, outputs_0[0][0,0], outputs_0[1][0,0], outputs_0[2][0,0]]
            watcher += [img_t1[0], gt, outputs_1[0][0,0], outputs_1[1][0,0], outputs_1[2][0,0]]
            cv2.imwrite("1.png", img_square(watcher, 2, 5))
        print("task has been finished.")

    return

if __name__ == "__main__":
    main()
