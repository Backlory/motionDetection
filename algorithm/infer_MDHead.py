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
from utils.toCPP import saveTensorToPt

from model.MDHead import MDHead

from algorithm.infer_VideoProcess import Inference_VideoProcess
from algorithm.infer_Homo_switcher import Inference_Homo_switcher
from algorithm.infer_Region_Proposal import Inference_Region_Proposal
from algorithm.infer_OpticalFlow import Inference_OpticalFlow

class Inference_MDHead():
    def __init__(self, args={
        'ifUseGPU':True, 
        'modelType':'weights',
        'continueTaskExpPath':'weights',
        'continueWeightsFile_weights':'weights/model_Train_MDHead_and_save_bs8_60.pkl'
    }) -> None:
        self.args = args
        
        print(colorstr('Initializing model for MDHead...', 'yellow'))
        self.model = MDHead()
        
        #设备
        if self.args['ifUseGPU']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device )

        #初始化
        print(colorstr('load weights for MDHead ...', 'yellow'))
        temp = torch.load(args["MDHead_weights"])
        self.model.load_state_dict(temp["state_dict"])
        
        # 参数导出
        for idx, item in enumerate(self.model.state_dict()):
            param_name = item[5:]
            param_value = self.model.state_dict()[item]
            saveTensorToPt(f"temp/mdhead-weights/{param_name}.pkl", param_value)
        print("all exported.")
        


    def run_test(self, fps_target=30, dataset = 'u'):
        print("========= run test  =====")
        self.infer_align = Inference_Homo_switcher(args=self.args)
        self.infer_RP = Inference_Region_Proposal(args=self.args)
        self.infer_optical = Inference_OpticalFlow(args=self.args)
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
        history_output = None
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
            moving_mask = self.infer_RP.__call__(img_t0, img_t1_warp, diffWarp)
            temp_rate_1 = moving_mask.mean() / 255
            toc(tp, "infer_RP", 1, False); tp=tic()

            #光流提取
            flo_ten, fmap1_ten = self.infer_optical.__call__(img_t0, img_t1_warp, moving_mask)
            toc(tp, "infer_optical", 1, False); tp=tic()

            
            # ==============================================↓↓↓↓
            out = self.__call__(flo_ten, fmap1_ten)
            toc(tp, "MDHead", 1, False); tp=tic()
            
            out = out.cpu().numpy().astype(np.uint8)
            
            if history_output is not None:
                out_addhis = cv2.bitwise_or(out, history_output)
            else:
                out_addhis = out
            
            img_t0_enhancement = np.array(img_t0, copy=True)
            img_t0_enhancement[:,:,0] *= (1-out_addhis)
            img_t0_enhancement[:,:,1] *= (1-out_addhis)
            img_t0_enhancement[:,:,2] *= (1-out_addhis)
            img_t0_enhancement[:,:,2] += (out_addhis * 255)
            img_t0_enhancement = img_t0_enhancement.astype(np.uint8)
            toc(tp, "postproces", 1, False); tp=tic()
            

            # ==============================================↑↑↑↑
            history_output = np.array(out, copy=True)
            t_use = toc(t)
            t_use_all.append(t_use)
            print(f'\r== frame {idx} ==> rate={effect}, PR_rate={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
            temp = img_square([img_t0, out, img_t0_enhancement], 1)
            temp = cv2.resize(temp, (960, 320),cv2.INTER_AREA)
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
    def __call__(self, flo_ten, fmap1_ten):
        '''
        挑选运动区域.
        history代表上一帧的检测结果，用于增强本帧。
        '''
        #tp = tic()
        out = self.model(flo_ten, fmap1_ten)    #第0维代表前景概率，第一维代表背景概率
        #out = 1 - torch.argmax(out, dim=1)                 #argmax所得为索引，0代表前景，1代表背景
        out = torch.argmin(out, dim=1)[0]                 #argmin正好将argmax反过来，此时0代表背景，1代表前景
        #out = nn.Softmax(1)(out)[0,0]               #softmax为所得
        
        
        return out

    
    