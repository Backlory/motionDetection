'''
Author: Backlory
github: https://github.com/Backlory
Date: 2022-06-03 09:53:57
LastEditors: backlory's desktop dbdx_liyaning@126.com
LastEditTime: 2022-07-07 22:59:51
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
import kornia
from utils.conf import get_conf

from utils.img_display import img_square
from utils.flow_viz import flow_to_image
from utils.timers import tic, toc


from algorithm.infer_all import Inference_all
from algorithm.infer_Homo_cancel import Inference_Homo_cancel
from algorithm.infer_Region_Proposal_cancel import Inference_Region_Proposal_cancel

#======================================================================
def main(video_idx, expname="pred", skip_frame=1):
    print("model...")
    
    infer_all = Inference_all()
    if "ori" in expname:
        pass
    if "nohomo" in expname:
        infer_all.infer_align = Inference_Homo_cancel()
    if "norp" in expname:
        infer_all.infer_RP = Inference_Region_Proposal_cancel()

        

    print("Processing...")
    if True:
        video_idx=video_idx
        skip_frame = 1
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        len_all = len_all - skip_frame
        try:
            os.mkdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Validation/video_{str(video_idx)}/{expname}")
        except:
            pass
                
        his_info = None
        temp_rate = []
        with torch.no_grad():
            for i in range(len_all):
                #i = i % (len_all-skip_frame*2)
                img_t0 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Validation/video_{str(video_idx)}/video/{str(i).zfill(3)}.png")
                img_t1 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Validation/video_{str(video_idx)}/video/{str(i+skip_frame).zfill(3)}.png")
                
                #gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
                #_, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
                h_img_t1, w_img_t1, _ = img_t1.shape
                
                t=tic()
                diffOrigin, moving_mask, out, img_t0_enhancement, img_t0_arrow, \
                    effect, alg_type, temp_rate_1, his_info, flo_out = infer_all.step(
                    img_t0, img_t1, his_info=None
                    )
                t_use = toc(t)
                temp = img_square([img_t0, img_t0_arrow], 1)
                #cv2.imshow("1", temp)
                #cv2.waitKey(0)
                filedir = f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Validation/video_{str(video_idx)}/{expname}/{str(i).zfill(3)}.png"
                
                cv2.imwrite(filedir,  out.astype(np.uint8)*255)
                
                print(f'\r{expname} {i} ==> rate={effect}, 运动区比例={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
                pass
                #cv2.waitKey(100)
        print("task has been finished.")


if __name__ == "__main__":
    for i in range(1, 16):
        main(i, "pred_ori_nohistory", 1) # 对齐+候选+检测
        #main(i, "pred_norp", 1)    #对齐+检测
        #main(i, "pred_nohomo", 1)  # 候选+检测
        #main(i, "pred_nohomo_norp", 1) #检测
        #main(i, "pred_ori_FPS10", 3) # 对齐+候选+检测
        #main(i, "pred_norp_FPS10", 3)    #对齐+检测
        #main(i, "pred_nohomo_FPS10", 3)  # 候选+检测
        #main(i, "pred_nohomo_norp_FPS10", 3) #检测
        
    