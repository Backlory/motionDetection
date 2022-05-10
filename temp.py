from pyexpat import model
import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

from utils.img_display import img_square
from utils.flow_viz import flow_to_image

from algorithm.infer_Homo_switcher import Inference_Homo_switcher
from algorithm.infer_Region_Proposal import Inference_Region_Proposal
from utils.timers import tic, toc


from model.thirdparty_RAFT.core.raft import RAFT
from model.thirdparty_RAFT.core.utils import flow_viz
from model.thirdparty_RAFT.core.utils.utils import InputPadder

#======================================================================
def main(video_idx):
    print("model...")
    infer_align = Inference_Homo_switcher()
    infer_RP = Inference_Region_Proposal()

    print("Processing...")
    if True:
        video_idx=video_idx
        step_frame = 1
        repeatTimes = 500   # 重复多少次
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        #len_all = len(os.listdir(r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video3"))
        
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--model',default="model/thirdparty_RAFT/model/raft-sintel.pth", help="restore checkpoint")  #things
        #parser.add_argument('--path', default="demo-frames", help="dataset for evaluation")
        parser.add_argument('--path', default=r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\Train\video_1\video", help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()
        model = RAFT(args)
        status = torch.load(args.model)
        status = {k[7:]:v for k,v in status.items()}
        model.load_state_dict(status)
        model.cuda()
        model.eval()
        last_flow = None
        
        temp_rate = []
        for i in range(len_all * repeatTimes):
            i = i % (len_all-step_frame*2)
            img_t0 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i).zfill(3)}.png")
            img_t1 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i+step_frame).zfill(3)}.png")
            #img_t0 = cv2.imread(f"E:\\dataset\\dataset-fg-det\\UAC_IN_CITY\\video3\\{str(i).zfill(5)}.jpg")
            #img_t1 = cv2.imread(f"E:\\dataset\\dataset-fg-det\\UAC_IN_CITY\\video3\\{str(i+step_frame).zfill(5)}.jpg")
            
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
            

            
            gridlength = 16
            h_gridlength = 8
            moving_mask_grid = torch.tensor(moving_mask)[None, None].float()
            moving_mask_grid_mask1 = nn.MaxPool2d(gridlength)(moving_mask_grid)
            moving_mask_grid_mask1 = moving_mask_grid_mask1[0,0].numpy().astype(np.uint8)
            moving_mask_grid_mask1 = cv2.resize(moving_mask_grid_mask1, (w_img_t1, h_img_t1), interpolation=cv2.INTER_NEAREST)
            #img_t0 = cv2.add(img_t0, np.zeros_like(img_t0), mask=moving_mask_grid_mask1)
            toc(tp, "mask", mute=False); tp = tic()
            
            #################################
            # Farneback光流
            # img_t0_gray = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
            # img_t1_gray = cv2.cvtColor(img_t1_warp, cv2.COLOR_BGR2GRAY)
            # flow = cv2.calcOpticalFlowFarneback(img_t0_gray, img_t1_gray, None, 0.25, 1, 15, 3, 5, 1.2, 0)  
            # flow_img2 = flow_to_image(flow)
            # print(flow.shape)
            
            # RAFT 光流
            image1 = torch.from_numpy(img_t0).permute(2, 0, 1).float()[None].to("cuda:0")
            image2 = torch.from_numpy(img_t1).permute(2, 0, 1).float()[None].to("cuda:0")
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            toc(tp, "img转tensor", mute=False); tp = tic()
            
            with torch.no_grad():
                flow_low, flow_up, coords0, coords1 = model(image1, image2, iters=20, test_mode=True, flow_init = last_flow)
            last_flow = flow_low.detach()
            toc(tp, "RAFT", mute=False); tp = tic()
            
            flo = flow_up[0].permute(1,2,0)
            toc(tp, "permute", mute=False); tp = tic()
            flo = flo.cpu()
            toc(tp, "cpu", mute=False); tp = tic()
            flo = flo.numpy()
            toc(tp, "numpy", mute=False); tp = tic()
            flo = flow_viz.flow_to_image(flo)
            toc(tp, "flow", mute=False); tp = tic()
            
            # 滤波激活输出
            flo_gray = cv2.cvtColor(flo, cv2.COLOR_BGR2GRAY)
            flo_activate = cv2.absdiff(flo_gray, cv2.blur(flo_gray, (64,64)))
            flo_activate = flo_activate / flo_activate.max() * 255
            flo_activate = flo_activate.astype(np.uint8)
            flo_activate = cv2.merge([flo_activate, flo_activate, flo_activate])
            flo_activate = cv2.add(flo_activate, np.zeros_like(flo_activate), mask=moving_mask_grid_mask1)
            toc(tp, "activate", mute=False)
            
            ##################################
            
            t_use = toc(t)
            
            watcher = [img_t0, img_t1_warp, diffWarp, moving_mask_grid_mask1, flo, flo_activate]
            cv2.imwrite(f"1.png", img_square(watcher, 2))
            print(f'\r== frame {i} ==> rate={effect}, 运动区比例={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
            pass
            #cv2.waitKey(100)
        print("task has been finished.")




if __name__ == "__main__":
    main(1)
    