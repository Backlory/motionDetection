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
        len_all = len(os.listdir(video_dir_base))
        len_all = len_all - step_frame
                
        his_info = None
        temp_rate = []
        with torch.no_grad():
            for i in range(len_all):
                #i = i % (len_all-step_frame*2)
                img_t0 = cv2.imread(f"{video_dir_base}{str(i).zfill(5)}.jpg")
                img_t1 = cv2.imread(f"{video_dir_base}{str(i+step_frame).zfill(5)}.jpg")

                
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
    datasetpath = "E:/dataset/dataset-fg-det/CDnet2014"
    path_tri = os.path.join(datasetpath, "dataset")
    data, metadata = [], []
    for cdname in os.listdir(path_tri):
        path_cdname = os.path.join(path_tri, cdname)
        for videoname in os.listdir(path_cdname):
            with open(os.path.join(path_cdname, videoname, 'temporalROI.txt'), 'r') as f:
                idx_begin, idx_end = f.readline().split(" ")
                metadata.append(
                    {
                    'videoname':cdname + "_" + videoname,
                    'maskpath':os.path.join(path_cdname, videoname, "ROI.bmp"),
                    'idx_begin':int(idx_begin),
                    'idx_end':int(idx_end)
                    }
                )
                temp = []
                for idx in range(int(idx_begin), int(idx_end)+1):
                    temp.append(os.path.join(path_cdname, videoname, "input", 'in'+str(idx).zfill(6)+'.jpg'))
                data.append(temp)

    for i in range(1, 48):
        main(i)
    