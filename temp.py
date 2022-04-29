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
        step_frame = 1
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        for i in range(len_all * 500):
            i = i % (len_all-step_frame)
            img_t0 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i).zfill(3)}.png")
            img_t1 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i+step_frame).zfill(3)}.png")
            gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
            _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
            #img_t0 = img_t0[:,:,(200-37):(213+37*2), ( 200-50):(201+50*2)]
            #img_t1 = img_t1[:,:,(200-37):(213+37*2), ( 200-50):(201+50*2)]
            #gt =             gt[(200-37):(213+37*2), ( 200-50):(201+50*2)]
            
            ##################################################################
            # 单应性变换
            alg_type, img_t1_warp, _, _, effect,  diffOrigin, diffWarp = infer_align.__call__(img_t0, img_t1)
            print(f"alg_type={alg_type}, effect={effect:.5f}")
            # _, diffWarp_thres = cv2.threshold(diffWarp, 10, 255, cv2.THRESH_BINARY)
            # _, diffOrigin_thres = cv2.threshold(diffOrigin, 10, 255, cv2.THRESH_BINARY)            
            # diffWarp_thres = cv2.medianBlur(diffWarp_thres, 5)
            # diffOrigin_thres = cv2.medianBlur(diffOrigin_thres, 5)
            # watcher = [img_t0, img_t1, img_t1_warp, gt, diffWarp_thres, diffOrigin_thres]
            # watcher = [img_t0, img_t1, img_t1_warp, gt, diffWarp, diffOrigin]
            # cv2.imwrite(f"{i}.png", img_square(watcher, 2, 3))


            #运动区域候选
            _, diffWarp_thres = cv2.threshold(diffWarp, 10, 255, cv2.THRESH_BINARY)
            diffWarp_thres = cv2.medianBlur(diffWarp_thres, 3)
            
            watcher = [img_t0, img_t1, img_t1_warp, gt, diffWarp_thres, diffOrigin]
            cv2.imwrite(f"1.png", img_square(watcher, 2, 3))
            cv2.waitKey(100)

        
        print("task has been finished.")

    return

if __name__ == "__main__":
    main()
