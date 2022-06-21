import sys
import cv2
from cv2 import RANSAC
import numpy as np
import torch
import random

from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc

from algorithm.infer_Homo_switcher import Inference_Homo_switcher


class Inference_Homo_cancel(Inference_Homo_switcher):
    def __init__(self, args={}) -> None:
        #super(Inference_Homo_cancel).__init__(args)
        pass
        
    def __call__(self, img_base, img_t1):
        img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        img_t1_gray = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
        diffWarp = cv2.absdiff(img_t1_gray, img_base_gray)       #扭前
        
        img_t1_warp = img_t1
        alg_type = 'None'
        diffOrigin_score = 1
        diffWarp_score = 1
        effect = 0
        diffOrigin = diffWarp
        diffWarp = diffWarp
        H_warp = np.eye(3)
        return alg_type, img_t1_warp, diffOrigin_score, diffWarp_score, effect,  diffOrigin, diffWarp, H_warp

   
    