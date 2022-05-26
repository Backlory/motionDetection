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


from algorithm.infer_all import Inference_all

#======================================================================
def main(video_idx):
    print("model...")
    
    infer_all = Inference_all()

    print("Processing...")
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
                #i = i % (len_all-step_frame*2)
                img_t0 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i).zfill(3)}.png")
                img_t1 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i+step_frame).zfill(3)}.png")
                
                gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
                _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
                h_img_t1, w_img_t1, _ = img_t1.shape
                
                t=tic()
                diffOrigin, moving_mask, out, img_t0_enhancement, img_t0_arrow, \
                    effect, alg_type, temp_rate_1, his_info = infer_all.step(
                    img_t0, img_t1, his_info=his_info
                    )
                t_use = toc(t)

                filedir = f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/pred/{str(i).zfill(3)}.png"
                cv2.imwrite(filedir,  out.cpu().numpy().astype(np.uint8)*255)
                
                print(f'\r== frame {i} ==> rate={effect}, 运动区比例={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
                pass
                #cv2.waitKey(100)
        print("task has been finished.")


if __name__ == "__main__":
    for i in range(1, 48):
        main(i)
    