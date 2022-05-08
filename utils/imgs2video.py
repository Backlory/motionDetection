import numpy as np
import os, sys
import sys

from matplotlib import pyplot as plt
import cv2

def img2video(img_dir_list, video_dir):
    '''
    将多个图片拼接成mp4视频。
    img_dir_list 文件路径列表
    
    video_dir 最终生成的视频路径
    
    e.g.

    img_dir_list = os.listdir('videos')
    img2video(img_dir_list, 'videos/vides1.mp4')

    '''
    print('preparing data...')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #os.system('rm video/processed.mp4')
    temp = cv2.imread(img_dir_list[0])
    fps = 30
    video_out = cv2.VideoWriter(video_dir, fourcc, fps, (temp.shape[1], temp.shape[0]))
    k = 1
    for i in img_dir_list[:-1]:
        print(k, img_dir_list[k])
        temp = cv2.imread(img_dir_list[k])
        if temp is None:
            print("opencv read a None at ",k,'->',img_dir_list[k])
            break
        video_out.write(temp)
        k+=1
    print('end.')
    #os.system('ls -l video/processed.mp4')
    video_out.release()

def get_image_from_dir(img_dirs:str):
    '''
    img_dir_list = get_image_from_dir('input')
    '''
    temp = len(os.listdir(img_dirs))
    img_dir_list = [img_dirs+"/" + str(x)+".png" for x in range(1, temp)]
    return img_dir_list

if __name__ == '__main__':
    img_dir_list = get_image_from_dir("3")
    img2video(img_dir_list, '3.mp4')
    
