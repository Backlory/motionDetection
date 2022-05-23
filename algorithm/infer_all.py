import sys, os
from tkinter.tix import Tree
import cv2
import numpy as np
import torch
import torch.nn as nn
import random

from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc
from utils.img_display import img_square
from utils.flow_viz import flow_to_image

from model.MDHead import MDHead

from algorithm.infer_VideoProcess import Inference_VideoProcess
from algorithm.infer_Homo_switcher import Inference_Homo_switcher
from algorithm.infer_Region_Proposal import Inference_Region_Proposal
from algorithm.infer_OpticalFlow import Inference_OpticalFlow
from algorithm.infer_MDHead import Inference_MDHead
from algorithm.infer_PostProcess import Inference_PostProcess

class Inference_all():
    def __init__(self, args={
        
        'ifUseGPU' : True,
        'ifDataAugment' : True,
        'ifDatasetAllTheSameTrick' : False,
        'datasetLenTrick' : -1,

        'modelType' : 'weights',
        'continueTaskExpPath' : "weights",
        'continueWeightsFile_weights' : "model_Train_Homo_and_save_bs32_96.pkl",
        'taskerName' : "Tester_Homo",
        
        "RAFT_model" : "model/thirdparty_RAFT/model/raft-sintel.pth",
        "RAFT_path" : r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\Train\video_1\video",
        "RAFT_small" : "store_true",
        "RAFT_mixed_precision" : "store_false",
        "RAFT_alternate_corr" : "store_true",

        "MDHead_weights" : "weights/model_Train_MDHead_and_save_bs8_60.pkl",
    }) -> None:
        self.args = args
        
        self.infer_align = Inference_Homo_switcher(args=self.args)
        self.infer_RP = Inference_Region_Proposal(args=self.args)
        self.infer_optical = Inference_OpticalFlow(args=self.args)
        self.infer_mdhead = Inference_MDHead(args=self.args)
        self.infer_postproc = Inference_PostProcess(args=self.args)
        
    def run_test(self, fps_target=30, dataset = 'u'):
        print("========= run test  =====")
        if dataset == 'u':
            path = r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video_all_1-skip.mp4"
        elif dataset == 'j':
            path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
        else:
            path = r'E:\dataset\dataset-fg-det\Kaggle-Drone-Videos\video_all.mp4'

        self.__call__(path, fps_target)
        return
        
    
    def __call__(self, path, fps_target=30, savedir="temp/1"):
        #
        try:
            os.mkdir(savedir)
        except:
            pass
        #
        cap = cv2.VideoCapture(path)
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=fps_target)
        print(f"run testing wiht fps_target = {tempVideoProcesser.fps_now}")
        
        idx = 0
        his_info = None
        while(True):
            idx += 1
            t = tic()
            # 读取图像
            img_t0, img_t1 = tempVideoProcesser()
            if img_t0 is None:
                print("all frame have been read.")
                break

            # 处理一帧
            diffOrigin, moving_mask, out, img_t0_enhancement, img_t0_arrow, \
                effect, alg_type, temp_rate_1, his_info = self.step(
                img_t0, img_t1, his_info=his_info
                )
            
            t_use = toc(t)
            print(f'\r== frame {idx} ==> rate={effect}, PR_rate={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
            cv2.imwrite(f"{savedir}/{idx}.png", img_t0_enhancement)
            cv2.imshow("origin", img_t0)
            cv2.imshow("moving detecting", img_t0_enhancement)
            cv2.waitKey(1)
            pass
            
        cv2.destroyAllWindows()
        cap.release()


    @torch.no_grad()
    def step(self, img_t0, img_t1, his_info=None):
        
        # 对齐
        alg_type, img_t1_warp, _, _, effect,  diffOrigin, diffWarp, H_warp = self.infer_align.__call__(img_t0, img_t1)
        
        # 运动区域候选
        moving_mask = self.infer_RP.__call__(img_t0, img_t1_warp, diffWarp, his_info)
        temp_rate_1 = moving_mask.mean() / 255
        
        #光流提取
        flo_ten, fmap1_ten = self.infer_optical.__call__(img_t0, img_t1_warp, moving_mask)
        
        # 运动检测
        out = self.infer_mdhead.__call__(flo_ten, fmap1_ten)
        
        # 后处理
        img_t0_enhancement, img_t0_arrow, his_info = self.infer_postproc.__call__(img_t0, out, H_warp, flo_ten, his_info)
        
        # 输出
        diffOrigin = cv2.cvtColor(diffOrigin, cv2.COLOR_GRAY2BGR)
        moving_mask = cv2.cvtColor(moving_mask, cv2.COLOR_GRAY2BGR)
        
        return diffOrigin, moving_mask, out, img_t0_enhancement, img_t0_arrow, \
                effect, alg_type, temp_rate_1, his_info
    
    