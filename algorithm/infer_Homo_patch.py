raise DeprecationWarning("this model has been deprecated!")

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
        self.ss1,self.ss2 = 0, 0
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
        
    def _get_homo_128(self, patch_t0, patch_t1):
        assert(patch_t0.shape == (128, 128, 1))
        patch_t0 = torch.Tensor(patch_t0/255).float().permute(2,0,1)[None]    #hwc->chw
        patch_t1 = torch.Tensor(patch_t1/255).float().permute(2,0,1)[None]
        patch_t0 = patch_t0 * 2 - 1
        patch_t1 = patch_t1 * 2 - 1
        patch_t0 = patch_t0.to(self.device)
        patch_t1 = patch_t1.to(self.device)
        output = self.model(patch_t0, patch_t1)
        delta = output[0].detach().cpu().numpy()
        
        fp = np.array([ (32, 32),
                        (160, 32),
                        (160, 160),
                        (32, 160)],
                        dtype=np.float32)
        pfp = np.float32(fp + delta * 8)
        H_warp = cv2.getPerspectiveTransform(fp, pfp)
        H2 = np.array([ 1, 0, -32,
                        0, 1, -32,
                        0, 0,   1 ]).reshape(3,3) #平移矩阵
        H_warp_patch = np.matmul(np.matmul(H2, H_warp), np.linalg.inv(H2)) 
        return H_warp_patch

    def __call__(self, img_t0, img_t1):
        '''
        t0向t1
        '''
        assert(img_t0.shape == img_t1.shape)
        assert(img_t0.shape[2] == 3)
        h, w, _ = img_t0.shape
        assert(w >= h)
        assert(h == 512)
        #
        img_t0_gray = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
        img_t1_gray = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
        #
        ps = 128 * 4
        p_tl = (256-int(ps/2), int(w/2)-int(ps/2))
        p_rb = (p_tl[0]+ps, p_tl[1]+ps)
        #
        patch_t0 = img_t0_gray[p_tl[0]:p_rb[0], p_tl[1]:p_rb[1]]
        patch_t1 = img_t1_gray[p_tl[0]:p_rb[0], p_tl[1]:p_rb[1]]
        patch_t0 = cv2.resize(patch_t0, (128,128))[:, :, np.newaxis]
        patch_t1 = cv2.resize(patch_t1, (128,128))[:, :, np.newaxis]
        #
        H_warp_patch = self._get_homo_128(patch_t0, patch_t1)
        #
        patch_t0_w = cv2.warpPerspective(patch_t0, H_warp_patch, (128,128))
        h,w,_ = patch_t0.shape
        patch_t0 = patch_t0[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95),:]
        patch_t1 = patch_t1[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95),:]
        patch_t0_w = patch_t0_w[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
        cv2.imwrite("0.png", img_t0_gray)
        cv2.imwrite("1.png", patch_t0)
        cv2.imwrite("2.png", patch_t1)
        cv2.imwrite("3.png", img_square([patch_t0, patch_t1, patch_t0_w, cv2.absdiff(patch_t1, patch_t0_w), cv2.absdiff(patch_t1, patch_t0)], 1,5))
        print(np.sum(cv2.absdiff(patch_t1, patch_t0_w)), np.sum(cv2.absdiff(patch_t1, patch_t0)))
        self.ss1 += np.sum(cv2.absdiff(patch_t1, patch_t0_w))
        self.ss2 += np.sum(cv2.absdiff(patch_t1, patch_t0))
        print(self.ss1, self.ss2)
        #img_t0_warp = cv2.warpPerspective(img_t0, H_warp, (w, h))
        #img_t0_warp = cv2.warpPerspective(img_t0_gray, H_warp, (192,192))
        #patch_t0_w = img_t0_w[int(0.25*ps):int(1.25*ps), int(0.25*ps):int(1.25*ps)][:,:,np.newaxis]
        
        
        return img_t0

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
            img_t1_warped = self(img_t1, img_t0)
            #
            # ==============================================↑↑↑↑
            h,w,_ = img_t0.shape
            img_t0 = img_t0[int(h*0.05):int(h*0.95), int(w*0.05):int(h*0.95),:]
            img_t1 = img_t1[int(h*0.05):int(h*0.95), int(w*0.05):int(h*0.95),:]
            img_t1_warped = img_t1_warped[int(h*0.05):int(h*0.95), int(w*0.05):int(h*0.95),:]
            #
            diff1 = cv2.absdiff(img_t0, img_t1)
            diff2 = cv2.absdiff(img_t0, img_t1_warped)
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