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
from utils.flow_viz import flow_to_image

from algorithm.infer_Homo_switcher import Inference_Homo_switcher
from algorithm.infer_Region_Proposal import Inference_Region_Proposal

#======================================================================
def main(video_idx):
    print("model...")
    infer_align = Inference_Homo_switcher()
    infer_RP = Inference_Region_Proposal()

    print("Processing...")
    if True:
        video_idx=video_idx
        step_frame = 1
        repeatTimes = 500   # 重复多少次
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        #len_all = len(os.listdir(r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video3"))
        
        #his_diffWarp_thres = None
        temp_rate = []
        for i in range(len_all * repeatTimes):
            i = i % (len_all-step_frame*2)
            img_t0 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i).zfill(3)}.png")
            img_t1 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i+step_frame).zfill(3)}.png")
            #img_t0 = cv2.imread(f"E:\\dataset\\dataset-fg-det\\UAC_IN_CITY\\video3\\{str(i).zfill(5)}.jpg")
            #img_t1 = cv2.imread(f"E:\\dataset\\dataset-fg-det\\UAC_IN_CITY\\video3\\{str(i+step_frame).zfill(5)}.jpg")
            #img_t0 = cv2.resize(img_t0, (640, 640))
            #img_t1 = cv2.resize(img_t1, (640, 640))
            img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
            img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
            
            gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
            _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
            #img_t0 = img_t0[:,:,(200-37):(213+37*2), ( 200-50):(201+50*2)]
            #img_t1 = img_t1[:,:,(200-37):(213+37*2), ( 200-50):(201+50*2)]
            #gt =             gt[(200-37):(213+37*2), ( 200-50):(201+50*2)]
            
            #################################
            hsv = np.zeros((640,640,3)) # 遍历每一行的第1列
            hsv[..., 1] = 255

            flow = cv2.calcOpticalFlowFarneback(img_t0, img_t1, None, 0.25, 1, 15, 3, 5, 1.2, 0)  
            print(flow.shape)

            # 笛卡尔坐标转换为极坐标，获得极轴和极角  
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
            hsv[..., 0] = ang * 180 / np.pi / 2  
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            ##################################
            
            
            watcher = [img_t0, rgb]
            cv2.imwrite(f"1.png", img_square(watcher, 1, 2))
            pass
            #cv2.waitKey(100)
            
        print(f"len_all = {len_all}, piexl = {len_all*640*640}, 屏蔽 = {(len_all*640*640) * (1 - np.mean(temp_rate))}")
        print(f"temp_rate = {1 - np.mean(temp_rate):.5f}")
        print(len_all, (len_all*640*640), int((len_all*640*640) * (1 - np.mean(temp_rate))), 1 - np.mean(temp_rate) )
        print("task has been finished.")

    return len_all, (len_all*640*640), int((len_all*640*640) * (1 - np.mean(temp_rate))), 1 - np.mean(temp_rate)




if __name__ == "__main__":
    #f = open("temp.txt", 'w')
    #sys.stdout = f
    main(1)
    #f.close()
