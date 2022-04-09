import sys
import cv2
from cv2 import RANSAC
import numpy as np
import torch
import random

from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc

from model.Homo import Homo_cnn, Homo_fc

from algorithm.infer_VideoProcess import Inference_VideoProcess
from algorithm.infer_Homo_ransac import Inference_Homo_RANSAC
from algorithm.infer_Homo_RSHomoNet import Inference_Homo_RSHomoNet


class Inference_Homo_switcher():
    def __init__(self, args) -> None:
        self.args = args
        self.infer_ransac = Inference_Homo_RANSAC(args)
        self.infer_rshomonet = Inference_Homo_RSHomoNet(args)
        self.effect_list = [0,0,0,0,0]

    def run_test(self, fps_target=30, dataset = 'u'):
        print("==============")
        print(f"run testing wiht fps_target = {fps_target}")
        if dataset == 'u':
            path = r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video_all_1.mp4"
        elif dataset == 'j':
            path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
        else:
            path = r'E:\dataset\dataset-fg-det\Kaggle-Drone-Videos\video_all.mp4'
        cap = cv2.VideoCapture(path)
        #
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=fps_target)
        fps = tempVideoProcesser.fps_now
        
        ss1,ss2 = [],[]
        effect_all = []
        t_use_all = []
        alg_his = {'ransac':0, 'rshomonet':0}
        idx = 0
        frameUseless = 0
        while(True):
            idx += 1

            img_t0, img_t1 = tempVideoProcesser()
            if img_t0 is None:
                print("all frame have been read.")
                break
            # ==============================================↓↓↓↓
            try:
                #t = tic()
                img_t0, img_t1_warp, diffOrigin, diffWarp, if_usefull, alg_type, t_use = self.__call__(img_t0, img_t1)
                #t_use = toc(t)
                # 原始图像，扭曲后图像，原始帧差值，扭曲后帧差值，是否工作，算法类型，时间消耗
                #temp = [img_t0, img_t1, img_t1_warped, cv2.absdiff(img_t0, img_t1_warped), cv2.absdiff(img_t0, img_t1)]
                #cv2.imwrite(f"{round(fps)}_{stride}.png", img_square(temp, 2,3))
                
                # ==============================================↑↑↑↑
                alg_his[alg_type] += 1
                if not if_usefull:
                    frameUseless += 1
                    diffOrigin = 1
                    diffWarp = diffOrigin
                    effect = 1 - diffWarp/diffOrigin
                ss1.append(diffOrigin)
                ss2.append(diffWarp)
                if if_usefull:
                    effect = 1 - diffWarp/diffOrigin
                    effect_all.append(effect)
                t_use_all.append(t_use)
                print(f'\r== frame {idx} ==> diff_origin = {diffOrigin}, diff_warp = {diffWarp}', "rate=", effect,"time=",t_use,'ms','alg_type=',alg_type,  end="")
            except:
                pass
            #cv2.imshow("test_origin", img_t0)
            #cv2.imshow("test_diff_origin",  cv2.absdiff(img_t0, img_t1))
            #cv2.imshow("test_diff_warp",  cv2.absdiff(img_t0, img_t1_warp))
            #cv2.waitKey(1)
            #if cv2.waitKey(int(1000/fps)) == 27: break
        print("\nframeUseless = ", frameUseless)
        
        #保存到文件
        savedStdout = sys.stdout
        with open("log.txt", "a+") as f:
            sys.stdout = f
            effect_all = np.average(effect_all)
            ss1_all = np.average(ss1)
            ss2_all = np.average(ss2)
            avg_t_use_all = np.average(t_use_all)
            print(f"{dataset}|{fps_target}|{0}|{idx}|{frameUseless}|{1-frameUseless/idx}|{ss1_all}|{ss2_all}|{1-ss2_all/ss1_all}|{effect_all}|{avg_t_use_all}")
        sys.stdout = savedStdout
            
        cv2.destroyAllWindows()
        cap.release()


    def __call__(self, img_base, img_t1):
        '''
        将t0向t_base扭曲
        '''
        t = tic()
        assert(img_t1.shape == img_base.shape)
        assert(img_t1.shape[2] == 3)
        h, w, _ = img_t1.shape
        assert(w >= h)
        assert(h == 512)
        
        img_t1_gray = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
        img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)

        H_warp = self.infer_ransac.core(img_t1_gray, img_base_gray)
        img_t1_warp = cv2.warpPerspective(img_t1, H_warp, (w, h))
        
        diffOrigin, diffWarp, effect = self.geteffect(img_base, img_t1, img_t1_warp)
        #效果检查
        '''
        temp = self.effect_list[0] + self.effect_list[1] + self.effect_list[2] + self.effect_list[3] + self.effect_list[4]
        temp = temp / 5
        if effect < temp * 0.8:
            H_warp = self.infer_rshomonet.core(img_t1_gray, img_base_gray, stride=2)
            _img_t1_warp = cv2.warpPerspective(img_t1, H_warp, (w, h))
            _diffOrigin, _diffWarp, _effect = self.geteffect(img_base, img_t1, _img_t1_warp)
            if _effect > effect:
                img_t1_warp = _img_t1_warp
                alg_type = 'RSHomoNet'
                diffOrigin = _diffOrigin
                diffWarp = _diffWarp
                effect = _effect
            else:
                alg_type = 'ransac'
        self.effect_list.append(effect)
        del  self.effect_list[0]
        '''
        alg_type = 'ransac'
        if_usefull = (effect>0)
        if effect<=0:
            img_t1_warp = img_t1
        t_use = toc(t)
        return img_base, img_t1_warp, diffOrigin, diffWarp, if_usefull, alg_type, t_use

    def geteffect(self, img_base, img_t1, img_t1_warp):
        ret, mask = cv2.threshold(img_t1_warp, 1, 1, cv2.THRESH_BINARY)

        diffOrigin = cv2.absdiff(img_t1, img_base)       #扭前
        diffWarp = cv2.absdiff(img_t1_warp, img_base)   #扭后

        diffOrigin = cv2.multiply(diffOrigin, mask)
        diffWarp = cv2.multiply(diffWarp, mask)

        diffOrigin = np.round(np.sum(diffOrigin), 4)
        diffWarp = np.round(np.sum(diffWarp), 4)

        effect = 1 - diffWarp/diffOrigin

        return diffOrigin, diffWarp, effect

    