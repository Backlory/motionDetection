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
    def __init__(self, args={
        'ifUseGPU':True, 
        'modelType':'weights',
        'continueTaskExpPath':'weights',
        'continueWeightsFile_weights':'model_Train_Homo_and_save_bs32_96.pkl'
    }) -> None:
        self.args = args
        self.infer_ransac = Inference_Homo_RANSAC()
        self.infer_rshomonet = Inference_Homo_RSHomoNet(args)
        self.effect_list = np.array([0,0,0,0,0])
        self.frameDifferenceDetector_threshold_alpha = 0.7
        

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
        alg_his = {'ransac':0, 'RSHomoNet':0, 'None':0}
        idx = 0
        frameUseless = 0
        while(True):
            idx += 1

            img_t0, img_t1 = tempVideoProcesser()
            if img_t0 is None:
                print("all frame have been read.")
                break
            # ==============================================↓↓↓↓
            if True:
                t = tic()
                alg_type, img_t1_warp, diffOrigin_score, diffWarp_score, effect,  diffOrigin, diffWarp, H_warp = self.__call__(img_t0, img_t1)
                t_use = toc(t)
                # 原始图像，扭曲后图像，原始帧差值，扭曲后帧差值，是否工作，算法类型，时间消耗
                #temp = [img_t0, img_t1, img_t1_warped, cv2.absdiff(img_t0, img_t1_warped), cv2.absdiff(img_t0, img_t1)]
                #cv2.imwrite(f"{round(fps)}_{stride}.png", img_square(temp, 2,3))
                
                # ==============================================↑↑↑↑
                alg_his[alg_type] += 1
                ss1.append(diffOrigin_score)
                ss2.append(diffWarp_score)
                if alg_type!='None':
                    effect_all.append(effect)
                else:
                    frameUseless += 1
                    effect = 0
                    effect_all.append(effect)   #消融实验
                t_use_all.append(t_use)
                print(f'\r== frame {idx} ==> rate=', effect,"time=",t_use,'ms','alg_type=',alg_type,  end="")

            #cv2.imshow("test_origin", img_t0)
            #cv2.imshow("test_diff_origin",  cv2.absdiff(img_t0, img_t1))
            #cv2.imshow("test_diff_warp",  cv2.absdiff(img_t0, img_t1_warp))
            #cv2.waitKey(1)
            #if cv2.waitKey(int(1000/fps)) == 27: break
        print("\nframeUseless = ", frameUseless)
        print(alg_his)
        #保存到文件
        savedStdout = sys.stdout
        with open("log.txt", "a+") as f:
            sys.stdout = f
            effect_all = np.average(effect_all)
            ss1_all = np.average(ss1)
            ss2_all = np.average(ss2)
            avg_t_use_all = np.average(t_use_all[10:])
            print(f"{dataset}|{fps_target}|{0}|{idx}|{frameUseless}|{1-frameUseless/idx}|{ss1_all}|{ss2_all}|{1-ss2_all/ss1_all}|{effect_all}|{avg_t_use_all}|{alg_his}")
        sys.stdout = savedStdout
            
        cv2.destroyAllWindows()
        cap.release()


    def __call__(self, img_base, img_t1):
        '''
        将t0向t_base扭曲.
        return:
        alg_type - 算法类型
        img_t1_warp - 扭曲后图像
        diffOrigin_score - 原始帧差得分（mask后）
        diffWarp_score - 扭曲帧差得分（mask后）
        effect - 性能
        diffOrigin - 原始帧差图
        diffWarp - 扭曲帧差图
        
        '''
        assert(img_t1.shape == img_base.shape)
        assert(img_t1.shape[2] == 3)
        h, w, _ = img_t1.shape
        assert(w >= h)
        
        img_t1_gray = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
        img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)

        alg_type = 'RANSAC'
        H_warp = self.infer_ransac.core(img_t1_gray, img_base_gray)
        img_t1_warp = cv2.warpPerspective(img_t1, H_warp, (w, h))
        
        diffOrigin_score, diffWarp_score, effect, diffOrigin, diffWarp = self.frameDifferenceDetect(img_base, img_t1, img_t1_warp)
        
        # 对齐容错控制器（被动容错控制思路）
        if True: 
            threshold = np.average(self.effect_list) * self.frameDifferenceDetector_threshold_alpha   #帧差突变阈值
            if effect < threshold or effect <= 0:
                print("检测到RANSAC故障，切换至RSHomoNet模型")
                with torch.no_grad():
                    _H_warp = self.infer_rshomonet.core(img_t1_gray, img_base_gray, stride=2)
                _img_t1_warp = cv2.warpPerspective(img_t1, _H_warp, (w, h))
                _diffOrigin_score, _diffWarp_score, _effect, _diffOrigin, _diffWarp = self.frameDifferenceDetect(img_base, img_t1, _img_t1_warp)
                if _effect > effect and _effect > 0:    #帧差监测器二次检测
                    img_t1_warp = _img_t1_warp
                    alg_type = 'RSHomoNet'
                    diffOrigin_score = _diffOrigin_score
                    diffWarp_score = _diffWarp_score
                    effect = _effect
                    diffOrigin = _diffOrigin
                    diffWarp = _diffWarp
                    H_warp = _H_warp
                elif effect > _effect and effect > 0:
                    print("恢复RANSAC输出。")
                else:
                    print("检测到RSHomoNet故障，取消对齐输出")
                    img_t1_warp = img_t1
                    alg_type = 'None'
                    diffOrigin_score = 1
                    diffWarp_score = 1
                    effect = 0
                    diffOrigin = diffWarp
                    diffWarp = diffWarp
                    H_warp = np.eye(3)
            self.effect_list = np.append(self.effect_list, effect)
            self.effect_list = np.delete(self.effect_list, 0)
        return alg_type, img_t1_warp, diffOrigin_score, diffWarp_score, effect,  diffOrigin, diffWarp, H_warp

    def frameDifferenceDetect(self, img_base, img_t1, img_t1_warp):
    
        img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
        img_t1_warp = cv2.cvtColor(img_t1_warp, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(img_t1_warp, 1, 1, cv2.THRESH_BINARY)

        diffOrigin = cv2.absdiff(img_t1, img_base)       #扭前
        diffWarp = cv2.absdiff(img_t1_warp, img_base)   #扭后

        diffOrigin = cv2.multiply(diffOrigin, mask)
        diffWarp = cv2.multiply(diffWarp, mask)

        diffOrigin_score = np.round(np.sum(diffOrigin), 4)
        diffWarp_score = np.round(np.sum(diffWarp), 4)
        effect = 1 - (diffWarp_score+1)/(diffOrigin_score+1)
        
        return diffOrigin_score, diffWarp_score, effect, diffOrigin, diffWarp

    