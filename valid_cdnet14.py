from distutils.command.check import check
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

#======================================================================
def validinfer(data, metadata, gt):
    print("model...")
    infer_all = Inference_all()
    infer_all.infer_align_back_s = infer_all.infer_align
    infer_all.infer_align_back_c = Inference_Homo_cancel()
    if "cameraJitter" in metadata["videoname"]:
        infer_all.infer_align = infer_all.infer_align_back_s
    else:
        infer_all.infer_align = infer_all.infer_align_back_c
    print(infer_all.infer_align)

    print("data...")
    cdmask = cv2.imread(metadata["maskpath"]) / 255
    temph, tempw,_ = cv2.imread(data[0]).shape
    cdmask = cv2.resize(cdmask, (tempw, temph))

    #assert(set(cdmask.flatten().tolist()) < set({0,1}))

    print("Processing...")
    if True:
        len_all = len(data)
                
        his_info = None
        temp_rate = []
        with torch.no_grad():
            for i in range(len_all - 1):
                img_t0 = cv2.imread(data[i]) * cdmask
                img_t1 = cv2.imread(data[i+1]) * cdmask
                img_t0 = img_t0.astype(np.uint8)
                img_t1 = img_t1.astype(np.uint8)
                img_gt = cv2.imread(gt[i]) * cdmask.astype(np.uint8)
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                t=tic()
                diffOrigin, moving_mask, out, img_t0_enhancement, img_t0_arrow, \
                    effect, alg_type, temp_rate_1, his_info = infer_all.step(
                    img_t0, img_t1, his_info=his_info
                    )
                t_use = toc(t)
                temp = gt[i].replace("\\dataset\\", "\\results\\")
                temp2=os.path.split((temp))
                try:
                    os.mkdir( temp2[0])
                except:
                    pass
                cv2.imwrite(temp, out* 255)
                watcher = [img_t0, moving_mask, out, img_gt]
                watcher = img_square(watcher, 2)
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

def main():
    datasetpath = "E:/dataset/dataset-fg-det/CDnet2014"
    path_tri = os.path.join(datasetpath, "dataset")
    data, metadata,gt = [], [], []
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

                temp = []
                for idx in range(int(idx_begin), int(idx_end)+1):
                    temp.append(os.path.join(path_cdname, videoname, "groundtruth", 'gt'+str(idx).zfill(6)+'.png'))
                gt.append(temp)
    for i in range(8, len(data)):
        validinfer(data[i], metadata[i], gt[i])
    return 

if __name__ == "__main__":
    main()
    