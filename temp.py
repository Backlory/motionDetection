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

#======================================================================
def main(video_idx):
    print("model...")
    infer_align = Inference_Homo_switcher()

    print("Processing...")
    if True:
        video_idx=video_idx
        step_frame = 1
        repeatTimes = 500   # 重复多少次
        len_all = len(os.listdir(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/"))
        #len_all = len(os.listdir(r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video3"))
        
        #his_diffWarp_thres = None
        temp_rate = []
        for i in range(len_all * repeatTimes):
            i = i % (len_all-step_frame*2)
            img_t0 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i).zfill(3)}.png")
            img_t1 = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/video/{str(i+step_frame).zfill(3)}.png")
            #img_t0 = cv2.imread(f"E:\\dataset\\dataset-fg-det\\UAC_IN_CITY\\video3\\{str(i).zfill(5)}.jpg")
            #img_t1 = cv2.imread(f"E:\\dataset\\dataset-fg-det\\UAC_IN_CITY\\video3\\{str(i+step_frame).zfill(5)}.jpg")
            #img_t0 = cv2.resize(img_t0, (640, 640))
            #img_t1 = cv2.resize(img_t1, (640, 640))
            
            gt = cv2.imread(f"E:/dataset/dataset-fg-det/Janus_UAV_Dataset/Train/video_{str(video_idx)}/gt_mov/{str(i).zfill(3)}.png", cv2.IMREAD_GRAYSCALE) 
            _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
            #img_t0 = img_t0[:,:,(200-37):(213+37*2), ( 200-50):(201+50*2)]
            #img_t1 = img_t1[:,:,(200-37):(213+37*2), ( 200-50):(201+50*2)]
            #gt =             gt[(200-37):(213+37*2), ( 200-50):(201+50*2)]
            
            #################################
            # 单应性变换
            alg_type, img_t1_warp, _, _, effect,  diffOrigin, diffWarp, H_warp = infer_align.__call__(img_t0, img_t1)
            print(i, f"alg_type={alg_type}, effect={effect:.5f}")
            # _, diffWarp_thres = cv2.threshold(diffWarp, 10, 255, cv2.THRESH_BINARY)
            # _, diffOrigin_thres = cv2.threshold(diffOrigin, 10, 255, cv2.THRESH_BINARY)            
            # diffWarp_thres = cv2.medianBlur(diffWarp_thres, 5)
            # diffOrigin_thres = cv2.medianBlur(diffOrigin_thres, 5)
            # watcher = [img_t0, img_t1, img_t1_warp, gt, diffWarp_thres, diffOrigin_thres]
            # watcher = [img_t0, img_t1, img_t1_warp, gt, diffWarp, diffOrigin]
            # cv2.imwrite(f"{i}.png", img_square(watcher, 2, 3))
            
            # 对齐变换带来的背景光流，从img_t1到img_t1_warp的
            
            img_x = np.repeat(np.arange(640)[None], 640, 0)
            img_y = np.repeat(np.arange(640)[:,None], 640, 1)
            temp = ( H_warp - np.eye(3) )
            img_dx = temp[0,0] * img_x + temp[0,1] * img_y + temp[0,2]
            img_dy = temp[1,0] * img_x + temp[1,1] * img_y + temp[1,2]
            flow = np.stack([img_dx, img_dy], axis=2)
            flow_img = flow_to_image(flow)

            #运动区域候选
            smallobj_size_thres = 40
            _, diffWarp_thres = cv2.threshold(diffWarp, 10, 255, cv2.THRESH_BINARY)
            diffWarp_thres = cv2.medianBlur(diffWarp_thres, 3)
            
            #structs = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            #diffWarp_thres = cv2.dilate(diffWarp_thres, structs)
            #diffWarp_thres = cv2.erode(diffWarp_thres, structs)
            
            #structs = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
            #diffWarp_thres = cv2.erode(diffWarp_thres, structs)
            
            diffWarp_thres = droplittlearea(diffWarp_thres, thres_area=smallobj_size_thres)
            
            diffWarp_thres = 255 - diffWarp_thres
            diffWarp_thres = droplittlearea(diffWarp_thres, thres_area=10000)
            diffWarp_thres = 255 - diffWarp_thres

            #if i > 0:
            #    diffWarp_thres = cv2.bitwise_and(diffWarp_thres, his_diffWarp_thres)
            #his_diffWarp_thres = diffWarp_thres
            
            
            if True: #转grid
                gridlength = 32
                h_gridlength = 16
                diffWarp_thres_grid = torch.tensor(diffWarp_thres)[None, None].float()
                diffWarp_thres_grid_mask1 = nn.MaxPool2d(gridlength)(diffWarp_thres_grid)
                diffWarp_thres_grid_mask1 = diffWarp_thres_grid_mask1[0,0].numpy().astype(np.uint8)
                diffWarp_thres_grid_mask1 = cv2.resize(diffWarp_thres_grid_mask1, (640,640), interpolation=cv2.INTER_NEAREST)
                
                diffWarp_thres_grid = diffWarp_thres_grid[:,:,h_gridlength:-h_gridlength, h_gridlength:-h_gridlength]
                diffWarp_thres_grid_mask2 = nn.MaxPool2d(gridlength)(diffWarp_thres_grid)
                diffWarp_thres_grid_mask2 = diffWarp_thres_grid_mask2[0,0].numpy().astype(np.uint8)
                diffWarp_thres_grid_mask2 = cv2.resize(diffWarp_thres_grid_mask2, (640-gridlength,640-gridlength), interpolation=cv2.INTER_NEAREST)
                diffWarp_thres_grid_mask2 = cv2.copyMakeBorder(
                    diffWarp_thres_grid_mask2, 
                    h_gridlength, 
                    h_gridlength, 
                    h_gridlength, 
                    h_gridlength,
                    cv2.BORDER_CONSTANT, 
                    value=0
                    )
                
                
                
                #cv2.imwrite(f"1.png", img_square([diffWarp_thres_grid], 1, 1))
            if False: #转bbox
                _, contours, hierarchy = cv2.findContours(diffWarp_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                bbox = []
                for c in contours:
                    x,y,w,h = cv2.boundingRect(c)
                    bbox.append([x,y,w,h])
                bbox = np.array(bbox)
                thresh = 0.7
                if bbox.size > 0:
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
                        index = np.where(ious <= thresh)[0]
                        # 根据嵌套判断
                        temp = inter / areas[order[1:]]
                        index = np.where(temp < 0.8)[0]
                        # 其他框
                        order = order[index+1]
                    bbox = bbox[bbox_selected]
                    for bbox_item in bbox:
                        x,y,w,h = bbox_item
                        x_c = x + w//2
                        y_c = y + h//2
                        w = w * 2
                        h = h * 2
                        if w < 20: w = 20 
                        if h < 20: h = 20 
                        x = x_c - w//2
                        y = y_c - h//2
                        img_t0 = cv2.rectangle(img_t0, (x,y), (x+w,y+h), (0,0,255), 2)
                

            temp_rate.append( diffWarp_thres_grid_mask1.mean() / 510  + diffWarp_thres_grid_mask2.mean()/510)

             
            #watcher = [img_t0, img_t1, img_t1_warp, gt, diffWarp_thres, diffWarp_thres_grid]
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
            
            watcher = [img_t0,  img_t1_warp, gt, diffWarp, diffWarp_thres, diffWarp_thres_grid1, diffWarp_thres_grid2, flow_img]
            cv2.imwrite(f"{i}.png", img_square(watcher, 2, 4))
            pass
            #cv2.waitKey(100)
            
        print(f"len_all = {len_all}, piexl = {len_all*640*640}, 屏蔽 = {(len_all*640*640) * (1 - np.mean(temp_rate))}")
        print(f"temp_rate = {1 - np.mean(temp_rate):.5f}")
        print(len_all, (len_all*640*640), int((len_all*640*640) * (1 - np.mean(temp_rate))), 1 - np.mean(temp_rate) )
        print("task has been finished.")

    return len_all, (len_all*640*640), int((len_all*640*640) * (1 - np.mean(temp_rate))), 1 - np.mean(temp_rate)


def droplittlearea(diffWarp_thres, thres_area = 9, thres_hmw=5, thres_k = 0.05):
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

if __name__ == "__main__":
    #f = open("temp.txt", 'w')
    #sys.stdout = f
    main(i)
    #f.close()
