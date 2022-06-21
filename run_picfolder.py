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
        video_dir_base = f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"
        video_dir_base = r"E:/dataset/dataset-fg-det/DAVIS/480p/bmx-bumps/"
        video_dir_base = r"E:/dataset/dataset-fg-det/DAVIS/480p/horsejump-low/"
        video_dir_base = r"E:/dataset/dataset-fg-det/DAVIS/480p/libby/"
        video_dir_base = r"E:/dataset/dataset-fg-det/DAVIS/480p/mallard-fly/"
        video_dir_base = r"E:/dataset/dataset-fg-det/DAVIS/480p/paragliding/"
        video_dir_base = r"E:/dataset/dataset-fg-det/DAVIS/480p/skate-park/"
        video_dir_base = r"E:\dataset\dataset-fg-det\PESMOD\Pexels-Elliot-road\images\\"
        
        len_all = len(os.listdir(video_dir_base))
        len_all = len_all - step_frame
                
        his_info = None
        temp_rate = []
        with torch.no_grad():
            for i in range(len_all):
                #i = i % (len_all-step_frame*2)
                if "DAVIS" in video_dir_base:
                    img_name = f"{str(i).zfill(5)}.jpg"
                    img_name2 = f"{str(i+step_frame).zfill(5)}.jpg"
                elif "PESMOD" in video_dir_base:
                    img_name = f"frame{str(i+1).zfill(4)}.jpg"
                    img_name2 = f"frame{str(i+step_frame+1).zfill(4)}.jpg"

                img_t0 = cv2.imread(f"{video_dir_base}{img_name}")
                img_t1 = cv2.imread(f"{video_dir_base}{img_name2}")
                
                if img_t0.size > 1080*720:
                    img_t0 = cv2.resize(img_t0, (1080,720))
                    img_t1 = cv2.resize(img_t1, (1080,720))


                t=tic()
                diffOrigin, moving_mask, out, img_t0_enhancement, img_t0_arrow, \
                    effect, alg_type, temp_rate_1, his_info = infer_all.step(
                    img_t0, img_t1, his_info=his_info
                    )
                t_use = toc(t)
                out = out.cpu().numpy().astype(np.uint8)*255

                watcher = [img_t0, out, img_t0_enhancement]
                watcher = img_square(watcher, 1)
                h,w,c = watcher.shape
                if h>400:
                    k = w/h
                    w = k * 400
                    h = 400
                    w = int(w)
                    if w >1600:
                        w = 1600
                    watcher = cv2.resize(watcher, (w,h))
                cv2.imshow("temp", watcher)
                
                print(f'\r== frame {i} ==> rate={effect}, 运动区比例={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
                pass
                cv2.waitKey(10)
        print("task has been finished.")


if __name__ == "__main__":
    for i in range(1, 48):
        main(i)
    