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

from model.MDHead import MDHead

from algorithm.infer_VideoProcess import Inference_VideoProcess
from algorithm.infer_Homo_switcher import Inference_Homo_switcher
from algorithm.infer_Region_Proposal import Inference_Region_Proposal
from algorithm.infer_OpticalFlow import Inference_OpticalFlow
from algorithm.infer_MDHead import Inference_MDHead

class Inference_PostProcess():
    def __init__(self, args={
    }) -> None:
        self.args = args
        

        


    def run_test(self, fps_target=30, dataset = 'u'):
        print("========= run test  =====")
        self.infer_align = Inference_Homo_switcher(args=self.args)
        self.infer_RP = Inference_Region_Proposal(args=self.args)
        self.infer_optical = Inference_OpticalFlow(args=self.args)
        self.infer_mdhead = Inference_MDHead(args=self.args)
        if dataset == 'u':
            path = r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video_all_1-skip.mp4"
        elif dataset == 'j':
            path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
        else:
            path = r'E:\dataset\dataset-fg-det\Kaggle-Drone-Videos\video_all.mp4'
        cap = cv2.VideoCapture(path)
        #
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=fps_target)
        print(f"run testing wiht fps_target = {tempVideoProcesser.fps_now}")
        
        t_use_all = []
        his_info = None
        idx = 0
        while(True):
            idx += 1
            
            img_t0, img_t1 = tempVideoProcesser()
            if img_t0 is None:
                print("all frame have been read.")
                break
            t = tic()
            # 对齐
            tp = tic()
            alg_type, img_t1_warp, _, _, effect,  diffOrigin, diffWarp, H_warp = self.infer_align.__call__(img_t0, img_t1)
            h_img_t1, w_img_t1, _ = img_t1.shape
            toc(tp, "infer_align", 1, False); tp=tic()

            # 运动区域候选
            moving_mask = self.infer_RP.__call__(img_t0, img_t1_warp, diffWarp, his_info)
            temp_rate_1 = moving_mask.mean() / 255
            toc(tp, "infer_RP", 1, False); tp=tic()

            #光流提取
            flo_ten, fmap1_ten = self.infer_optical.__call__(img_t0, img_t1_warp, moving_mask)
            toc(tp, "infer_optical", 1, False); tp=tic()

            out = self.infer_mdhead.__call__(flo_ten, fmap1_ten)
            toc(tp, "infer_mdhead", 1, False); tp=tic()

            # ==============================================↓↓↓↓
            
            img_t0_colorblock,img_t0_arrow, his_info = self.__call__(img_t0, out, H_warp, flo_ten, his_info)
            toc(tp, "postproces", 1, False); tp=tic()
            

            # ==============================================↑↑↑↑
            t_use = toc(t)
            t_use_all.append(t_use)
            print(f'\r== frame {idx} ==> rate={effect}, PR_rate={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
            temp = img_square([img_t0, out, img_t0_colorblock, img_t0_arrow], 1)
            temp = cv2.resize(temp, (1280, 320),cv2.INTER_NEAREST)
            cv2.imshow("1", temp)
            cv2.waitKey(1)
            #cv2.imwrite(f"1.png", temp)
            #cv2.imwrite(f"temp/4/{idx}.png", temp)
            cv2.imwrite(f"temp/{dataset}/{idx}.png", temp)
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
    def __call__(self, img_t0, out, H_warp, flo_ten, his_info=None):
        '''
        '''
        out = out.cpu().numpy().astype(np.uint8)

        # 历史信息解析
        if his_info is not None:
            his_info["lastout"]
            out_addhis = cv2.bitwise_or(out, his_info["lastout"])
        else:
            out_addhis = out
        
        # 运动区增强
        img_t0_enhancement = np.array(img_t0, copy=True)
        img_t0_enhancement[:,:,0] *= (1-out_addhis)
        img_t0_enhancement[:,:,1] *= (1-out_addhis)
        img_t0_enhancement[:,:,2] *= (1-out_addhis)
        img_t0_enhancement[:,:,2] += (out_addhis * 255)
        img_t0_enhancement = img_t0_enhancement.astype(np.uint8)
        
        # 获取运动目标中心位置的光流
        try:
            H_warp_inv = np.linalg.inv(H_warp)
        except:
            H_warp_inv = np.eye(3)
        img_t0_arrow = np.array(img_t0, copy=True)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(out)
        for i in range(1, nlabels):
            regions_size = stats[i,4] 
            if regions_size > 20: # 面积过滤
                x,y,w,h = stats[i,0:4]
                x_center = x + w // 2
                y_center = y + h // 2
                flo_point = flo_ten[0,:, y_center, x_center]
                v_w, v_h  = flo_point
                v_w, v_h = v_w.item(), v_h.item()
                k = (v_w**2 + v_h**2) ** 0.5
                if 0 < k < 10:
                    v_h *= 10/k 
                    v_w *= 10/k 
                v_w, v_h = v_w + x_center, v_h + y_center

                #还原光流
                v_w_warp, v_h_warp, _ = np.matmul(H_warp_inv, np.array([[v_w], [v_h], [1]]))
                v_w_warp, v_h_warp = int(v_w_warp), int(v_h_warp)
                
                # 打印箭头
                img_t0_arrow = cv2.rectangle(img_t0_arrow, (x,y), (x+w,y+h), (0,0,255), 2)
                img_t0_arrow = cv2.arrowedLine(img_t0_arrow, (x_center, y_center), (v_w_warp, v_h_warp), (0,0,0), 2, tipLength=0.2)
                img_t0_enhancement = cv2.arrowedLine(img_t0_enhancement, (x_center, y_center), (v_w_warp, v_h_warp), (0,0,0), 2, tipLength=0.2)
                
        # 保存历史信息
        his_info = {"lastout":out}
        return img_t0_enhancement, img_t0_arrow, his_info

    
    