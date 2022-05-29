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
def main(video_idx):
    print("model...")

    print("Processing...")
    evaluator = Evaluator(2)
    if True:
        video_idx=video_idx
        step_frame = 1
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        len_all = len_all - step_frame
        try:
            os.mkdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/pred")
        except:
            pass
                
        his_info = None
        temp_rate = []
        with torch.no_grad():
            for i in range(len_all):
                gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
                _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
                
                pred = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/pred/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
                evaluator.add_batch_np(pred, gt)
                pass
                #cv2.waitKey(100)
            
            mIoU, FWIoU, Acc, mAcc, mPre, mRecall, mF1, AuC = evaluator.evaluateAll()
            print(evaluator.confusion_matrix)
            print(f"mIoU={mIoU:.4f}, FWIoU={FWIoU:.4f}, Acc={Acc:.4f}, mAcc={mAcc:.4f}")
            print(f"mPre={mPre:.4f}, mRecall={mRecall:.4f}, mF1={mF1:.4f}, AuC={AuC:.4f}")

            savedStdout = sys.stdout
            with open("log2.txt", "a+") as f:
                sys.stdout = f
                print(f"{mIoU:.4f}, {FWIoU:.4f}, {Acc:.4f}, {mAcc:.4f}, {mPre:.4f}, {mRecall:.4f}, {mF1:.4f}, {AuC:.4f}")
            sys.stdout = savedStdout
        print("task has been finished.")


if __name__ == "__main__":
    for i in range(1, 48):
        main(i)
    