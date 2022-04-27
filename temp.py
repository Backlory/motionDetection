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

from algorithm.infer_Homo_switcher import Inference_Homo_switcher

#======================================================================
def main():
    print("model...")
    infer_align = Inference_Homo_switcher()

    print("Processing...")
    if True:    
        video_idx=1
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        for i in range(len_all * 1000-5000):
            i = i % (len_all-5)
            img_t0 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i).zfill(3)}.png")
            img_t1 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i+5).zfill(3)}.png")
            gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_1/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
            _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
            #img_t0 = img_t0[:,:,(200-37):(213+37*2), ( 200-50):(201+50*2)]
            #img_t1 = img_t1[:,:,(200-37):(213+37*2), ( 200-50):(201+50*2)]
            #gt =             gt[(200-37):(213+37*2), ( 200-50):(201+50*2)]
            
            ##################################################################
            # 单应性变换
            alg_type, img_t1_warp, _, _, effect,  diffOrigin, diffWarp = infer_align.__call__(img_t0, img_t1)
            print(f"alg_type={alg_type}, effect={effect:.5f}")
            
            _, diffWarp = cv2.threshold(diffWarp, 10, 255, cv2.THRESH_BINARY)
            _, diffOrigin = cv2.threshold(diffOrigin, 10, 255, cv2.THRESH_BINARY)            
            
            watcher = [img_t0, img_t1, img_t1_warp, gt, diffWarp, diffOrigin]
            cv2.imwrite("1.png", img_square(watcher, 2, 3))

        
        print("task has been finished.")

    return

if __name__ == "__main__":
    main()
