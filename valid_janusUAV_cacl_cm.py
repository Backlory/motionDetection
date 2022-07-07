'''
Author: Backlory
github: https://github.com/Backlory
Date: 2022-05-29 11:26:51
LastEditors: backlory's desktop dbdx_liyaning@126.com
LastEditTime: 2022-07-07 23:19:51
Description: 

Copyright (c) 2022 by Backlory, All Rights Reserved. 
'''
import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
from utils.indicator import Evaluator
#======================================================================
def main(video_idx, expname="pred", cm=None):
    print("model...")

    print("Processing...")
    evaluator = Evaluator(2)
    if cm is not None:
        evaluator.confusion_matrix = cm
    if True:
        video_idx=video_idx
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        len_all = len_all - 1

        his_info = None
        temp_rate = []
        with torch.no_grad():
            for i in range(len_all):
                gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Validation/video_{str(video_idx)}/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
                _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
                #pred = cv2.imread(f"E:/codes/220630-SmallObjMotionSeg/temp/janusUAVtest/Validation/out_video_{str(video_idx)}/Algorithm_MaskRAFT\{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Validation/video_{str(video_idx)}/{expname}/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
                evaluator.add_batch_np(pred, gt)
                pass
                #cv2.waitKey(100)
            
            mIoU, FWIoU, Acc, mAcc, mPre, mRecall, mF1, AuC = evaluator.evaluateAll()
            print(evaluator.confusion_matrix)
            print(f"mIoU={mIoU:.4f}, FWIoU={FWIoU:.4f}, Acc={Acc:.4f}, mAcc={mAcc:.4f}")
            print(f"mPre={mPre:.4f}, mRecall={mRecall:.4f}, mF1={mF1:.4f}, AuC={AuC:.4f}")
            #if video_idx == 15:
            savedStdout = sys.stdout
            with open(f"log_valid_janusUAV_{expname}.txt", "a+") as f:
                sys.stdout = f
                print(f"{expname}, {mIoU:.4f}, {FWIoU:.4f}, {Acc:.4f}, {mAcc:.4f}, {mPre:.4f}, {mRecall:.4f}, {mF1:.4f}, {AuC:.4f}")
            sys.stdout = savedStdout
        print("task has been finished.")
        return evaluator.confusion_matrix
"name	mIoU	FWIoU	Acc	mAcc	mPre	mRecall	mF1	AuC"
if __name__ == "__main__":
    cm1, cm2, cm3, cm4 = None,None,None,None
    for i in range(1, 16):
        #cm1 = main(i,"pred_ori", cm1)
        #cm2 = main(i,"pred_norp", cm2)
        #cm3 = main(i,"pred_nohomo", cm3)
        #cm4 = main(i,"pred_nohomo_norp", cm4)
        cm1 = main(i,"pred_ori_nohistory", cm1)
        pass
    print(cm1)