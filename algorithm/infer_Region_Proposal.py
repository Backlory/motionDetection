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

from model.Homo import Homo_cnn, Homo_fc

from algorithm.infer_VideoProcess import Inference_VideoProcess
from algorithm.infer_Homo_switcher import Inference_Homo_switcher

class Inference_Region_Proposal():
    def __init__(self, smallobj_size_thres:int=70, args={}) -> None:
        self.args = args
        self.smallobj_size_thres = smallobj_size_thres
        self.infer_align = Inference_Homo_switcher()


    def run_test(self, fps_target=30, dataset = 'u'):
        print("==============")
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
            
            alg_type, img_t1_warp, _, _, effect,  diffOrigin, diffWarp, H_warp = self.infer_align.__call__(img_t0, img_t1)
            h_img_t1, w_img_t1, _ = img_t1.shape
            
            # 对齐变换带来的背景光流，从img_t1到img_t1_warp的
            img_x = np.repeat(np.arange(w_img_t1)[None], h_img_t1, 0)
            img_y = np.repeat(np.arange(h_img_t1)[:,None], w_img_t1, 1)
            temp = ( H_warp - np.eye(3) )
            img_dx = temp[0,0] * img_x + temp[0,1] * img_y + temp[0,2]
            img_dy = temp[1,0] * img_x + temp[1,1] * img_y + temp[1,2]
            flow = np.stack([img_dx, img_dy], axis=2)
            flow_img = flow_to_image(flow)

            #运动区域候选
            t = tic()
            tensors_input, tensors_t1, diffWarp_thres, img_t0_boxed = self.__call__(img_t0, img_t1_warp, diffWarp)
            t_use = toc(t)
            
            # ==============================================↓↓↓↓
            
            if True: #转grid
                gridlength = 8
                h_gridlength = 4
                diffWarp_thres_grid = torch.tensor(diffWarp_thres)[None, None].float()
                diffWarp_thres_grid_mask1 = nn.MaxPool2d(gridlength)(diffWarp_thres_grid)
                diffWarp_thres_grid_mask1 = diffWarp_thres_grid_mask1[0,0].numpy().astype(np.uint8)
                diffWarp_thres_grid_mask1 = cv2.resize(diffWarp_thres_grid_mask1, (w_img_t1,h_img_t1), interpolation=cv2.INTER_NEAREST)
                
                diffWarp_thres_grid = diffWarp_thres_grid[:,:,h_gridlength:-h_gridlength, h_gridlength:-h_gridlength]
                diffWarp_thres_grid_mask2 = nn.MaxPool2d(gridlength)(diffWarp_thres_grid)
                diffWarp_thres_grid_mask2 = diffWarp_thres_grid_mask2[0,0].numpy().astype(np.uint8)
                diffWarp_thres_grid_mask2 = cv2.resize(diffWarp_thres_grid_mask2, (w_img_t1-gridlength,h_img_t1-gridlength), interpolation=cv2.INTER_NEAREST)
                diffWarp_thres_grid_mask2 = cv2.copyMakeBorder(
                    diffWarp_thres_grid_mask2, 
                    h_gridlength, 
                    h_gridlength, 
                    h_gridlength, 
                    h_gridlength, 
                    cv2.BORDER_CONSTANT, 
                    value=0
                    ) 
                diffWarp_thres_grid1 = cv2.add(img_t0, np.zeros(np.shape(img_t0), dtype=np.uint8), mask=diffWarp_thres_grid_mask1)
                diffWarp_thres_grid2 = cv2.add(img_t0, np.zeros(np.shape(img_t0), dtype=np.uint8), mask=diffWarp_thres_grid_mask2)
                
                for i in range(640 // gridlength):
                    diffWarp_thres_grid1 = cv2.line(
                        diffWarp_thres_grid1, 
                        (gridlength*i, 0), 
                        (gridlength*i, 640), 
                        (0,50,0),
                        thickness=1
                        )
                    diffWarp_thres_grid1 = cv2.line(
                        diffWarp_thres_grid1, 
                        (0, gridlength*i), 
                        (640, gridlength*i), 
                        (0,50,0), 
                        thickness=1
                        )
                    diffWarp_thres_grid2 = cv2.line(
                        diffWarp_thres_grid2, 
                        (gridlength*i, 0), 
                        (gridlength*i, 640), 
                        (0,50,0),
                        thickness=1
                        )
                    diffWarp_thres_grid2 = cv2.line(
                        diffWarp_thres_grid2, 
                        (0, gridlength*i), 
                        (640, gridlength*i), 
                        (0,50,0), 
                        thickness=1
                        )
                for i in range(640 // gridlength):
                    diffWarp_thres_grid2 = cv2.line(
                        diffWarp_thres_grid2,
                        (gridlength*i+h_gridlength, 0), 
                        (gridlength*i+h_gridlength, 640), 
                        (0,0,100), 
                        thickness=2
                        )
                    diffWarp_thres_grid2 = cv2.line(
                        diffWarp_thres_grid2, 
                        (0, gridlength*i+h_gridlength), 
                        (640, gridlength*i+h_gridlength), 
                        (0,0,100), 
                        thickness=2
                        )
                temp_rate_1 = diffWarp_thres_grid_mask1.mean() / 510  + diffWarp_thres_grid_mask2.mean()/510
                temp_rate.append( temp_rate_1 )
                watcher = [img_t0_boxed,  img_t1_warp, None, diffWarp, diffWarp_thres, diffWarp_thres_grid1, diffWarp_thres_grid2, flow_img]
            
            # ==============================================↑↑↑↑
            t_use_all.append(t_use)
            print(f'\r== frame {idx} ==> rate={effect}, PR_rate={temp_rate_1:.5f}, time={t_use}ms, alg_type={alg_type}',  end="")
            cv2.imwrite(f"1.png", img_square(watcher, 2, 4))
            pass
            
            
        #保存到文件
        savedStdout = sys.stdout
        with open("log.txt", "a+") as f:
            sys.stdout = f
            #print(f"{dataset}|{fps_target}|{0}|{idx}|{frameUseless}|{1-frameUseless/idx}|{ss1_all}|{ss2_all}|{1-ss2_all/ss1_all}|{effect_all}|{avg_t_use_all}|{alg_his}")
        sys.stdout = savedStdout
            
        cv2.destroyAllWindows()
        cap.release()


    def __call__(self, img_base, img_warped, diffWarp, thresh_IOU = 0.7):
        '''
        挑选运动区域
        return:
        '''
        assert(img_base.shape == img_warped.shape)
        assert(img_base.shape[2] == 3)
        h_img_base, w_img_base, _ = img_base.shape
        assert(w_img_base >= h_img_base)

        
        
        _, diffWarp_thres = cv2.threshold(diffWarp, 10, 255, cv2.THRESH_BINARY)
        diffWarp_thres = cv2.medianBlur(diffWarp_thres, 3)
        
        #structs = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        #diffWarp_thres = cv2.dilate(diffWarp_thres, structs)
        #diffWarp_thres = cv2.erode(diffWarp_thres, structs)
        
        #structs = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        #diffWarp_thres = cv2.erode(diffWarp_thres, structs)
        
        diffWarp_thres = self.droplittlearea(diffWarp_thres, thres_area=self.smallobj_size_thres)
        
        diffWarp_thres = 255 - diffWarp_thres
        diffWarp_thres = self.droplittlearea(diffWarp_thres, thres_area=10000)
        diffWarp_thres = 255 - diffWarp_thres
        
        tensors_input = []
        tensors_t1 = []
        
        # 区域边界框判断
        _, contours, hierarchy = cv2.findContours(diffWarp_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bbox = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            bbox.append([x,y,w,h])
        bbox = np.array(bbox)
        
        # 区域筛选
        ps = 32
        h_ps = 16
        
        img_base_boxed = np.array(img_base, copy=True)
        #区域扩张
        for idx in range(len(bbox)):
            x,y,w,h = bbox[idx]
            if w < ps and h < ps:
                y = y + h//2 - h_ps
                x = x + w//2 - h_ps
                h, w = ps, ps
                bbox[idx] = x,y,w,h
            
        if bbox.size > 0:
            # NMI 非极大值抑制
            x1 = bbox[:, 0]
            y1 = bbox[:, 1]
            x2 = bbox[:, 0] + bbox[:, 2]
            y2 = bbox[:, 1] + bbox[:, 3]
            w_all = bbox[:,2]
            h_all = bbox[:,3]
            areas = w_all * h_all
            order = areas.argsort()[::-1]
            bbox_selected = []
            while order.size > 0:
                i = order[0]
                bbox_selected.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])#所有左上角点中的最右下角者
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])#所有右下角点中最左上角者
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                # 相交性判断
                w = np.maximum(0, xx2-xx1)
                h = np.maximum(0, yy2-yy1)
                inter = w * h
                # 根据当前IOU判断
                ious = inter / (areas[i] + areas[order[1:]] - inter)
                index = np.where(ious <= thresh_IOU)[0]
                # 根据嵌套判断
                temp = inter / areas[order[1:]]
                index = np.where(temp < 0.8)[0]
                
                # 该框有关判断结束
                order = order[index+1]
            bbox = bbox[bbox_selected]

            # 扩张过滤
            w_padding = np.max(bbox[:, 2])
            h_padding = np.max(bbox[:, 3])
            w_padding = max(w_padding, ps)
            h_padding = max(h_padding, ps)
            img_base = cv2.copyMakeBorder(img_base, h_padding, h_padding, w_padding, w_padding,
                                            cv2.BORDER_CONSTANT, value=0)
            img_warped = cv2.copyMakeBorder(img_warped, h_padding, h_padding, w_padding, w_padding,
                                            cv2.BORDER_CONSTANT, value=0) 
        else:
            bbox = []
        for bbox_item in bbox:
            x,y,w,h = bbox_item
            print(w, h)
            if w < ps and h < ps:
                y = y + h//2 - h_ps
                x = x + w//2 - h_ps
                h, w = ps, ps
            x = x + w_padding
            y = y + h_padding
            temp = np.array(img_base[y:y+h, x:x+w, :], copy=True)
            temp2 = np.array(img_warped[y-h:y+2*h, x-w:x+2*w, :], copy=True)

            #temp = cv2.resize(temp, (ps, ps), interpolation=cv2.INTER_AREA)
            #temp2 = cv2.resize(temp2, (ps*3, ps*3), interpolation=cv2.INTER_AREA)
            #cv2.imwrite("1.png", temp)
            #cv2.imwrite("1.png", temp2)
            x = x - w_padding
            y = y - h_padding
            img_base_boxed = cv2.rectangle(img_base_boxed, (x,y), (x+w,y+h), (0,0,255), 2)
        
            
        return tensors_input, tensors_t1, diffWarp_thres, img_base_boxed

    def droplittlearea(self, diffWarp_thres, thres_area = 9, thres_hmw=5, thres_k = 0.05):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(diffWarp_thres)
                #total_region_size = np.sum(stats[1:nlabels,4])
        stats = np.array(stats)
        for i in range(1, nlabels):
            regions_size = stats[i,4]
            area = stats[i,2] * stats[i,3]
            k = regions_size / area
            hmw = stats[i,2] / stats[i,3]
            if regions_size < thres_area or hmw > thres_hmw or hmw < 1/thres_hmw or k < thres_k:
                x0 = stats[i,0]
                y0 = stats[i,1]
                x1 = stats[i,0]+stats[i,2]
                y1 = stats[i,1]+stats[i,3]
                diffWarp_thres[y0:y1, x0:x1] = 0
        return diffWarp_thres

    