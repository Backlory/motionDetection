import sys
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

from model.MaskRAFT import Mask_RAFT
from model.thirdparty_RAFT.core.utils import flow_viz
from model.thirdparty_RAFT.core.utils.utils import InputPadder


from algorithm.infer_VideoProcess import Inference_VideoProcess
from algorithm.infer_Homo_switcher import Inference_Homo_switcher
from algorithm.infer_Region_Proposal import Inference_Region_Proposal

class Inference_OpticalFlow():
    def __init__(self, smallobj_size_thres:int=70, args={}) -> None:
        self.args = args
        self.smallobj_size_thres = smallobj_size_thres
        
        self.gridLength = 16     #边长  #【】【】【】【】【】
        self.iters = 10
        
        self.padder = InputPadder([1,3,640,640])
        
        self.model = Mask_RAFT(self.gridLength, args)
        
        if self.args['ifUseGPU']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device )
        
        self.model.eval()
        self.pool = nn.MaxPool2d(self.gridLength)

        


    def run_test(self, fps_target=30, dataset = 'u'):
        print("========= run test  =====")
        self.infer_align = Inference_Homo_switcher(args=self.args)
        self.infer_RP = Inference_Region_Proposal(args=self.args)
        if dataset == 'u':
            path = r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video_all_1.mp4"
        elif dataset == 'j':
            path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
        else:
            path = r'E:\dataset\dataset-fg-det\Kaggle-Drone-Videos\video_all.mp4'
        cap = cv2.VideoCapture(path)
        #
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=fps_target)
        print(f"run testing wiht fps_target = {tempVideoProcesser.fps_now}")
        
        t_use_all = []
        temp_rate = []
        idx = 0
        while(True):
            idx += 1
            
            img_t0, img_t1 = tempVideoProcesser()
            if img_t0 is None:
                print("all frame have been read.")
                break
            t = tic()
            # 对齐
            alg_type, img_t1_warp, _, _, effect,  diffOrigin, diffWarp, H_warp = self.infer_align.__call__(img_t0, img_t1)
            h_img_t1, w_img_t1, _ = img_t1.shape
            
            # 运动区域候选
            moving_mask = self.infer_RP.__call__(img_t0, img_t1_warp, diffWarp)
            temp_rate_1 = moving_mask.mean() / 255
            
            
            # ==============================================↓↓↓↓
            flo_ten, fmap1_ten = self.__call__(img_t0, img_t1_warp, moving_mask)
            nn.AvgPool2d(64, 1,63)
            flo = flo_ten[0].permute(1,2,0).cpu().numpy()
            flo = flow_viz.flow_to_image(flo)
            
            flo_gray = cv2.cvtColor(flo, cv2.COLOR_BGR2GRAY)
            flo_gray_blur = cv2.blur(flo_gray, (3, 3))
            flo_activate = cv2.absdiff(flo_gray, flo_gray_blur)
            flo_activate = flo_activate / flo_activate.max() * 255
            flo_activate = flo_activate.astype(np.uint8)
            flo_activate = cv2.cvtColor(flo_activate, cv2.COLOR_GRAY2BGR)
            
            # flo_gray_blur2 = cv2.blur(flo_gray, (16, 16))
            # flo_gray_blur3 = cv2.blur(flo_gray, (32, 32))
            # flo_gray_blur4 = cv2.blur(flo_gray, (64, 64))
            # flo_gray_blur5 = cv2.blur(flo_gray, (128, 128))
            # flo_activate2 = cv2.absdiff(flo_gray, flo_gray_blur2)
            # flo_activate3 = cv2.absdiff(flo_gray, flo_gray_blur3)
            # flo_activate4 = cv2.absdiff(flo_gray, flo_gray_blur4)
            # flo_activate5 = cv2.absdiff(flo_gray, flo_gray_blur5)
            # flo_activate2 = cv2.cvtColor(flo_activate2, cv2.COLOR_GRAY2BGR)
            # flo_activate3 = cv2.cvtColor(flo_activate3, cv2.COLOR_GRAY2BGR)
            # flo_activate4 = cv2.cvtColor(flo_activate4, cv2.COLOR_GRAY2BGR)
            # flo_activate5 = cv2.cvtColor(flo_activate5, cv2.COLOR_GRAY2BGR)
            # watcher = [flo, flo_activate, flo_activate2, flo_activate3, flo_activate4, flo_activate5]
            # cv2.imwrite(f"1.png", img_square(watcher, 2))

            # ==============================================↑↑↑↑
            t_use = toc(t)
            t_use_all.append(t_use)
            print(f'\r== frame {idx} ==> rate={effect}, PR_rate={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
            watcher = [img_t0, img_t1_warp, diffWarp, flo]
            temp = img_square(watcher, 2,2)
            temp = cv2.resize(temp, (640,640),cv2.INTER_AREA)
            #cv2.imwrite(f"1.png", temp)
            cv2.imwrite(f"temp/4/{idx}.png", temp)
            #cv2.imwrite(f"temp/{dataset}/{idx}.png", temp)
            pass
            
            
        #保存到文件
        savedStdout = sys.stdout
        with open("log.txt", "a+") as f:
            sys.stdout = f
            #print(f"{dataset}|{fps_target}|{0}|{idx}|{frameUseless}|{1-frameUseless/idx}|{ss1_all}|{ss2_all}|{1-ss2_all/ss1_all}|{effect_all}|{avg_t_use_all}|{alg_his}")
        sys.stdout = savedStdout
            
        cv2.destroyAllWindows()
        cap.release()

    @torch.no_grad()
    def __call__(self, img_base, img_warped, moving_mask, last_flow = None):
        '''
        挑选运动区域.
        history代表上一帧的检测结果，用于增强本帧。
        '''
        #tp = tic()

        # 原始图像
        image1 = torch.from_numpy(img_base).permute(2, 0, 1).float()[None].to("cuda:0")
        image2 = torch.from_numpy(img_warped).permute(2, 0, 1).float()[None].to("cuda:0")
        
        image1, image2 = self.padder.pad(image1, image2)
        #toc(tp, "image", mute=False); tp = tic()
        
        # mask处理
        gridLength = self.gridLength
        moving_mask_ten = torch.tensor(moving_mask)[None, None].float()
        moving_mask_ten = self.pool(moving_mask_ten)[0,0]
        moving_mask = moving_mask_ten.numpy().astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        moving_mask_2 = cv2.dilate(moving_mask, kernel=kernel)
        
        Mask_small = torch.tensor(moving_mask)[None, None].float().to(image1.device)
        Mask_small_2 = torch.tensor(moving_mask_2)[None, None].float().to(image1.device)
        Masks = [Mask_small, Mask_small_2]
        #toc(tp, "Mask", mute=False); tp = tic()
        
        #################################
        # RAFT 光流
        flow_low, flow_up, coords0, coords1, fmap1 = self.model(image1, image2, Masks, iters=self.iters, test_mode=True, flow_init = last_flow)
        last_flow = flow_low.detach()
        #toc(tp, "RAFT", mute=False); tp = tic()
    
        #img_t0 = add_mask(img_t0, moving_mask_resized)
        #img_t1_warp = add_mask(img_t1_warp, moving_mask_resized)
      
        return flow_up.detach(), fmap1.detach()

    
    