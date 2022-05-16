import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import kornia
from utils.conf import get_conf

from utils.img_display import img_square
from utils.flow_viz import flow_to_image
from utils.timers import tic, toc

from model.thirdparty_RAFT.core.utils import flow_viz

from algorithm.infer_Homo_switcher import Inference_Homo_switcher
from algorithm.infer_Region_Proposal import Inference_Region_Proposal
from algorithm.infer_OpticalFlow import Inference_OpticalFlow

#======================================================================
def main(video_idx):
    print("model...")
    infer_align = Inference_Homo_switcher(args=get_conf("Inference_OpticalFlow"))
    infer_RP = Inference_Region_Proposal(args=get_conf("Inference_OpticalFlow"))
    infer_optical = Inference_OpticalFlow(args=get_conf("Inference_OpticalFlow"))

    print("Processing...")
    if True:
        video_idx=video_idx
        step_frame = 1
        repeatTimes = 1   # 重复多少次
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        #len_all = len(os.listdir(r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video3"))
        
        last_flow = None
        temp_rate = []
        with torch.no_grad():
            for i in range(len_all * repeatTimes):
                i = i % (len_all-step_frame*2)
                img_t0 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i).zfill(3)}.png")
                img_t1 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i+step_frame).zfill(3)}.png")
                
                gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
                _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
                h_img_t1, w_img_t1, _ = img_t1.shape
                
                t = tic()
                tp = tic()
                # 对齐
                alg_type, img_t1_warp, _, _, effect,  diffOrigin, diffWarp, H_warp = infer_align.__call__(img_t0, img_t1)
                print();toc(tp, "对齐", mute=False); tp = tic()
                
                # 区域候选
                moving_mask = infer_RP.__call__(img_t0, img_t1_warp, diffWarp)
                temp_rate_1 = moving_mask.mean() / 255
                toc(tp, "区域候选", mute=False); tp = tic()

                # 光流提取
                flo_ten, fmap1_ten = infer_optical(img_t0, img_t1_warp, moving_mask)
                
                # 光流具现化
                flo = flo_ten[0].permute(1,2,0).cpu().numpy()
                flo = flow_viz.flow_to_image(flo)
                t_use = toc(t)
                
                watcher = [img_t0, img_t1_warp, moving_mask, flo]
                #cv2.imwrite("1.png", img_square(watcher, 2))
                
                os.mkdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/flow")
                flodir = f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/flow/{str(i).zfill(3)}.png"
                cv2.imwrite(flodir, flo)
                
                os.mkdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/flow_ten")
                flotendir = f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/flow_ten/{str(i).zfill(3)}.flo"
                torch.save(flo_ten, flotendir)
                print(f'\r== frame {i} ==> rate={effect}, 运动区比例={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
                pass
                #cv2.waitKey(100)
        print("task has been finished.")

def add_mask(img, mask):
    return cv2.add(img, np.zeros_like(img), mask=mask)
                


if __name__ == "__main__":
    for i in range(1, 40):
        main(i)
    