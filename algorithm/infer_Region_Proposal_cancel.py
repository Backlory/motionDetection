import sys
from tkinter.tix import Tree
import cv2
import numpy as np
import torch
import torch.nn as nn


from algorithm.infer_Region_Proposal import Inference_Region_Proposal

class Inference_Region_Proposal_cancel(Inference_Region_Proposal):
    def __init__(self, smallobj_size_thres:int=70, args={}) -> None:
        super().__init__(smallobj_size_thres, args)
        self.args = args
        self.smallobj_size_thres = smallobj_size_thres

    def __call__(self, img_base, img_warped, diffWarp, history = None):
        '''
        挑选运动区域.
        history代表上一帧的检测结果，用于增强本帧。
        '''
        moving_mask = np.ones_like(img_base[:,:,0], dtype = np.uint8)
        moving_mask = moving_mask * 255
        
       
        return moving_mask