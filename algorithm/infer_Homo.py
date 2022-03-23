import cv2
import numpy as np
import torch

from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc

from model.Homo import HomographyNet

from algorithm.infer_VideoProcess import Inference_VideoProcess
class Inference_Homo():
    def __init__(self, args) -> None:
        self.args = args
        
        # 设备
        print(colorstr('Initializing device...', 'yellow'))
        if self.args['ifUseGPU']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        else:
            self.device = torch.device('cpu')

        print(colorstr('Initializing model...', 'yellow'))
        if self.args['modelType'] == 'weights':
            self.model = HomographyNet()
            path = self.args['continueTaskExpPath'] + '/' + self.args['continueWeightsFile_weights']
            temp_states = torch.load(path)
            self.model.load_state_dict(temp_states['state_dict'])
            self.model.to(self.device).eval()
        elif self.args['modelType'] == 'script':
            path = self.args['continueTaskExpPath'] + '/' + self.args['continueWeightsFile_script']
            self.model = torch.jit.load(path)
        
    
    def __call__(self, img_t0, img_t1):
        assert(img_t0.shape == img_t1.shape)
        assert(img_t0.shape[2] == 3)
        h, w, _ = img_t0.shape
        assert(w > h)
        
        return img_t1

    def run_test(self):
        path = r"E:\dataset\UAC_IN_CITY\video3.mp4"
        cap = cv2.VideoCapture(path)
        #
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=10)
        fps = tempVideoProcesser.fps_now
        
        cv2.namedWindow("test_origin",cv2.WINDOW_FREERATIO)
        cv2.namedWindow("test_diff_origin",cv2.WINDOW_FREERATIO)
        cv2.namedWindow("test_diff_warp",cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("test_origin", 512,512)
        cv2.resizeWindow("test_diff_origin", 512,512)
        cv2.resizeWindow("test_diff_warp", 512,512)
        
        idx = 0
        while(True):
            idx += 1

            img_t0, img_t1 = tempVideoProcesser()
            if img_t0 is None:
                print("all frame have been read.")
                break
            # ==============================================↓↓↓↓
            # 
            img_t1_warped = self(img_t0, img_t1)
            #
            # ==============================================↑↑↑↑
            diff1 = cv2.subtract(img_t0, img_t1)
            diff2 = cv2.subtract(img_t0, img_t1_warped)
            temp1 = np.round(np.mean(diff1), 4)
            temp2 = np.round(np.mean(diff2), 4)
            effect = str(1-temp2/temp1) if temp2<temp1 else "-"+str(1-temp1/temp2)
            print(f'== frame {idx} ==> diff_origin = {temp1}, diff_warp = {temp2}, Effect={effect}%')
            cv2.imshow("test_origin", img_t0)
            cv2.imshow("test_diff_origin", diff1)
            cv2.imshow("test_diff_warp", diff2)

            if cv2.waitKey(int(1000/fps)) == 27:
                break
        cv2.destroyAllWindows()
        cap.release()