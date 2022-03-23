import cv2
import numpy as np
import torch

from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc

class Inference_VideoProcess():
    def __init__(self, cap=None, fps_target = None, args={}) -> None:
        self.args = args
        
        self.cap = None
        self.img_t0 = None
        self.img_t1 = None 
        self.flagTranspose = False
        self.skip_frame = 0
        self.fps_now = None
        
        if cap is not None: 
            self.set_cap(cap)
            if fps_target is not None:
                self.set_fps_target(fps_target)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def set_cap(self, cap):
        _, img_t1 = cap.read()
        assert(img_t1 is not None)
        h, w, _ = img_t1.shape
        if h > w: # h > w
            self.flagTranspose = True
        self.w_target = int(w * (512/h) )
        self.h_target = 512
        
        
        self.cap = cap
        self.img_t1 = self.preProcess(img_t1)
        

    def set_fps_target(self, fps_target):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.skip_frame = max(int(fps / fps_target - 1), 0)
        self.fps_now = fps / (1 + self.skip_frame)
    
    def __call__(self):
        # time passes
        self.img_t0 = self.img_t1

        # read new frame
        for _ in range(self.skip_frame+1):
            _, img_t1 = self.cap.read()
        if img_t1 is None:
            print("test complete.")
            return None, None
        
        # preProcess
        self.img_t1 = self.preProcess(img_t1)

        # assert
        assert(self.img_t0.shape == self.img_t1.shape)
        return self.img_t0, self.img_t1

    def preProcess(self, img_t1):
        # transpose
        if self.flagTranspose:
            img_t1 = np.transpose(img_t1, [1,0,2])
        # resize
        img_t1 = cv2.resize(img_t1, (self.w_target, self.h_target))
        return img_t1

    def postProcess(self, img_t0):
        if self.flagTranspose:
            img_t0 = np.transpose(img_t0, [1,0,2])
        return img_t0

    def run_test(self):
        path = r"E:\dataset\UAC_IN_CITY\video3.mp4"
        cap = cv2.VideoCapture(path)
        #
        self.set_cap(cap)
        self.set_fps_target(10)
        fps = self.fps_now
        
        cv2.namedWindow("test_origin_t0",cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("test_origin_t0", 512,512)
        cv2.namedWindow("test_origin_t1",cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("test_origin_t1", 512,512)
        cv2.namedWindow("test_origin_diff",cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("test_origin_diff", 512,512)
        idx = 0
        while(True):
            idx += 1

            img_t0, img_t1 = self()
            if img_t0 is None:
                print("all frame have been read.")
                break

            print(f'== frame {idx} ==')
            cv2.imshow("test_origin_t0", img_t0)
            cv2.imshow("test_origin_t1", img_t1)
            cv2.imshow("test_origin_diff", cv2.subtract(img_t0, img_t1))
            
            if cv2.waitKey(int(1000/fps)) == 27:
                break
        cv2.destroyAllWindows()
        cap.release()
        