import sys
import cv2
from importlib_metadata import files
import numpy as np
import torch

from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc

from model.Homo import Homo_cnn, Homo_fc

from algorithm.infer_VideoProcess import Inference_VideoProcess
class Inference_Homo_RSHomoNet():
    def __init__(self, args) -> None:
        self.args = args
        # 设备
        print(colorstr('Initializing device...', 'yellow'))
        if self.args['ifUseGPU']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        else:
            self.device = torch.device('cpu')

        # 模型
        print(colorstr('Initializing model...', 'yellow'))
        if self.args['modelType'] == 'weights':
            self.model_cnn = Homo_cnn()
            self.model_fc = Homo_fc()
            path = self.args['continueTaskExpPath'] + '/' + self.args['continueWeightsFile_weights']
            from utils.pytorchmodel import updata_adaptive
            temp_states = {}
            for k,v in torch.load(path)['state_dict'].items():
                k_ = k.replace("cnn.",'cnn')
                temp_states[k_] = v
            self.model_cnn = updata_adaptive(self.model_cnn, temp_states)
            self.model_fc = updata_adaptive(self.model_fc, temp_states)
            self.model_cnn.to(self.device).eval()
            self.model_fc.to(self.device).eval()
        elif self.args['modelType'] == 'script':
            path = self.args['continueTaskExpPath'] + '/' + self.args['continueWeightsFile_script']
            self.model = torch.jit.load(path)

    def run_test(self, fps_target=30, stride = 4, alpha=0):
        print("==============")
        print(f"run testing wiht fps_target = {fps_target}, stride = {stride}")
        #path = r"E:\dataset\dataset-fg-det\UAC_IN_CITY\video_all_1.mp4"
        #path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
        path = r'E:\dataset\dataset-fg-det\Kaggle-Drone-Videos\video_all.mp4'
        cap = cv2.VideoCapture(path)
        #
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=fps_target)
        fps = tempVideoProcesser.fps_now
        '''
        cv2.namedWindow("test_origin",cv2.WINDOW_FREERATIO)
        cv2.namedWindow("test_diff_origin",cv2.WINDOW_FREERATIO)
        cv2.namedWindow("test_diff_warp",cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("test_origin", 512,512)
        cv2.resizeWindow("test_diff_origin", 512,512)
        cv2.resizeWindow("test_diff_warp", 512,512)
        '''
        self.ss1,self.ss2 = [],[]
        self.effect_all = []
        t_use_all = []
        idx = 0
        effect = 0
        frameUseless = 0
        while(True):
            idx += 1
            #if idx>10:break

            img_t0, img_t1 = tempVideoProcesser()
            if img_t0 is None:
                print("all frame have been read.")
                break
            # ==============================================↓↓↓↓
            #
            #img_t1_warped = self._test_one_patch(img_t1, img_t0)
            #img_t1_warped = self._test_patches(img_t1, img_t0)
            img_t0, img_t1_warped, diffOrigin, diffWarp, if_usefull, t_use = self.__call__(img_t1, img_t0, stride=stride, alpha=alpha)
            #temp = [img_t0, img_t1, img_t1_warped, cv2.absdiff(img_t0, img_t1_warped), cv2.absdiff(img_t0, img_t1)]
            #cv2.imwrite(f"{round(fps)}_{stride}.png", img_square(temp, 2,3))
            
            # ==============================================↑↑↑↑
            if not if_usefull:
                frameUseless += 1
                diffOrigin = 1
                diffWarp = diffOrigin
            self.ss1.append(diffOrigin)
            self.ss2.append(diffWarp)
            if if_usefull:
                effect = 1 - diffWarp/diffOrigin
                self.effect_all.append(effect)
            t_use_all.append(t_use)
            print(f'\r== frame {idx} ==> diff_origin = {diffOrigin}, diff_warp = {diffWarp}', "rate=", effect,"time=",t_use,'ms',  end="")
            #cv2.imshow("test_origin", img_t0)
            #cv2.imshow("test_diff_origin",  cv2.absdiff(img_t0, img_t1))
            #cv2.imshow("test_diff_warp",  cv2.absdiff(img_t0, img_t1_warped))
            #cv2.waitKey(1)
            #if cv2.waitKey(int(1000/fps)) == 27: break
        print("\nframeUseless = ", frameUseless)
        
        #保存到文件
        savedStdout = sys.stdout
        with open("log.txt", "a+") as f:
            sys.stdout = f
            effect_all = np.average(self.effect_all)
            ss1_all = np.average(self.ss1)
            ss2_all = np.average(self.ss2)
            avg_t_use_all = np.average(t_use_all)
            print(f"{alpha}|{fps_target}|{stride}|{idx}|{frameUseless}|{1-frameUseless/idx}|{ss1_all}|{ss2_all}|{1-ss2_all/ss1_all}|{effect_all}|{avg_t_use_all}")
        sys.stdout = savedStdout
            
        cv2.destroyAllWindows()
        cap.release()

    def _get_feas_batch(self, patch_t0s, patch_t1s):
        assert(len(patch_t0s.shape) == 4)
        patch_t0 = torch.Tensor(patch_t0s/255).float()
        patch_t1 = torch.Tensor(patch_t1s/255).float()
        patch_t0 = patch_t0 * 2 - 1
        patch_t1 = patch_t1 * 2 - 1
        patch_t0 = patch_t0.to(self.device)
        patch_t1 = patch_t1.to(self.device)
        features = self.model_cnn(patch_t0, patch_t1)[4]
        return features

    def _get_fea(self, patch_t0, patch_t1):
        assert(len(patch_t0.shape) == 2)
        patch_t0 = torch.Tensor(patch_t0/255).float()[None,None]
        patch_t1 = torch.Tensor(patch_t1/255).float()[None,None]
        patch_t0 = patch_t0 * 2 - 1
        patch_t1 = patch_t1 * 2 - 1
        patch_t0 = patch_t0.to(self.device)
        patch_t1 = patch_t1.to(self.device)
        features = self.model_cnn(patch_t0, patch_t1)[4]
        return features

    def _get_output(self, features):
        output = self.model_fc(features)
        delta = output.detach().cpu().numpy()
        delta = np.float32(delta * 8)
        return delta

    def __call__(self, img_t0, img_base, stride=4, alpha=0, checkusefull = True):
        '''
        alpha是运动预测，将t0向t1扭曲
        '''
        assert(img_t0.shape == img_base.shape)
        assert(img_t0.shape[2] == 3)
        h, w, _ = img_t0.shape
        assert(w >= h)
        assert(h == 512)
        #
        img_t0_gray = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
        img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        
        
        # 网格采样。先卷再切，再输出。破坏邻域关 系，不行，故先切再卷
        
        
        #串行法
        #delta = []
        #for i in range(stride):
        #    for j in range(stride):
        #        features1 = self._get_fea(img_t0_gray_resized[i::stride, j::stride], img_base_gray_resized[i::stride, j::stride])
        #        delta1 = self._get_output(features1)[0]
        #        delta.append(delta1)
        t = tic()
        H_warp = self.core(img_t0_gray, img_base_gray, stride)
        t_use=toc(t,"Homo",mute=True)
        

        
        #历史预测及其更新
        '''
        delta_pred = None  
        if self.delta_last is not None:
            temp_v = delta - self.delta_last
            if self.delta_last_v is not None:
                temp_a = temp_v - self.delta_last_v
                if self.delta_last_a is not None:
                    delta_pred = self.delta_last + self.delta_last_v + self.delta_last_a / 2 #x=x0+vt+at^2/2
                self.delta_last_a = temp_a
            self.delta_last_v = temp_v
        self.delta_last = delta
        if delta_pred is not None:
            pfp = pfp * (1-alpha) + (fp + delta_pred) * alpha
        '''

        # 执行单应性变换
        img_t0_warp = cv2.warpPerspective(img_t0, H_warp, (w, h))
        h,w,_ = img_t0.shape
        #img_t0 = img_t0[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95),:]
        #img_base = img_base[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95),:]
        #img_t0_warp = img_t0_warp[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95),:]
        
        # 有效性检查
        if checkusefull:
            ret, mask = cv2.threshold(img_t0_warp, 1, 1, cv2.THRESH_BINARY)
            
            diffOrigin = cv2.absdiff(img_t0, img_base)       #扭前
            diffWarp = cv2.absdiff(img_t0_warp, img_base)   #扭后
            
            diffOrigin = cv2.multiply(diffOrigin, mask)
            diffWarp = cv2.multiply(diffWarp, mask)
            
            diffOrigin = np.round(np.sum(diffOrigin), 4)
            diffWarp = np.round(np.sum(diffWarp), 4)
            
            if_usefull = (diffOrigin > diffWarp)
            if not if_usefull:
                img_t0_warp = img_t0
            return img_base, img_t0_warp, diffOrigin, diffWarp, if_usefull, t_use
        else:
            return img_base, img_t0_warp

    def core(self, img_t0_gray, img_base_gray, stride):
        
        h, w = img_t0_gray.shape
        img_t0_gray_resized = cv2.resize(img_t0_gray, (128 * stride, 128 * stride))
        img_base_gray_resized = cv2.resize(img_base_gray, (128 * stride, 128 * stride))
        # 并行法
        patch_t0s = []
        patch_t1s = []
        for i in range(stride):
            for j in range(stride):
                patch_t0s.append(img_t0_gray_resized[i::stride, j::stride])
                patch_t1s.append(img_base_gray_resized[i::stride, j::stride])
        patch_t0s = np.array(patch_t0s)[:,np.newaxis,:,:]
        patch_t1s = np.array(patch_t1s)[:,np.newaxis,:,:]
        features1 = self._get_feas_batch(patch_t0s, patch_t1s)
        delta = self._get_output(features1)
        
        delta_all = np.array(delta)
        temp1 = [[0,0],[128,0],[128,128],[0,128]]
        if stride>1:
            delta = np.sort(delta_all, axis=0)[stride-1:(stride**2-stride+1),:,:]
            delta = np.average(delta, axis=0)
        else:
            delta = delta_all

        # 计算当前帧的角点偏移量
        fp = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        delta[:,0] = delta[:,0] / 128 * w
        delta[:,1] = delta[:,1] / 128 * h
        pfp = (fp + delta)
        H_warp = self.getPerspectiveTransform(fp, pfp)
        #可视化
        if False:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = plt.gca()
            ax.invert_yaxis()
            for i in range(len(delta_all)):
                temp1 = [[0,0],[128,0],[128,128],[0,128]]
                temp2 = delta_all[i].tolist()
                for i in range(len(temp1)):
                    x,y = temp1[i]
                    u,v = temp2[i]
                    plt.quiver(x,y,u,v)
            plt.figure()
            for i in range(4):
                x,y = temp1[i]
                u,v = delta[i]
                plt.quiver(x,y,u,v)
            ax.invert_yaxis()
            plt.show()
        return H_warp

    def getPerspectiveTransform(self, fp, pfp):
        H = cv2.getPerspectiveTransform(fp, pfp)
        return H

    def _test_patches(self, img_t0, img_t1):
        '''
        t0向t1
        '''
        def _test_patches_get_point_motion_vector(self, features, p_tl=[0,0], size = 128) -> list: 
            #输出模型在256*256下的标准输出
            _, c, h, w = features.shape
            assert(c == 128)
            k = h/16    #应该为16，但现在却为h，故缩小了k倍
            size_input = 256 * k
            
            p_tl_fea = int(p_tl[0] / size_input * h), int(p_tl[1] / size_input * w)
            fea_patch = features[:, :, p_tl_fea[0]:p_tl_fea[0]+8, p_tl_fea[1]:p_tl_fea[1]+8]
            
            output = self.model_fc(fea_patch)
            delta = output[0].detach().cpu().numpy()
            delta = np.float32(delta * 8)
            
            size_origin = int(size / k)
            fp = np.array([ p_tl,
                            (p_tl[0]+size_origin, p_tl[1]),
                            (p_tl[0]+size_origin, p_tl[1]+size_origin),
                            (p_tl[0],      p_tl[1]+size_origin)], np.float32)
            return fp, delta * k
        assert(img_t0.shape == img_t1.shape)
        assert(img_t0.shape[2] == 3)
        h, w, _ = img_t0.shape
        assert(w >= h)
        assert(h == 512)
        #
        img_t0_gray = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
        img_t1_gray = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
        img_t0_gray = cv2.resize(img_t0_gray, (256, 256))
        img_t1_gray = cv2.resize(img_t1_gray, (256, 256))
        
        # 特征提取
        features = self._get_fea(img_t0_gray, img_t1_gray)   #1, 128, 16, 16
        
        _, _, h, w = features.shape
        point_pair_ori = []
        point_pair_mov = []
        
        for p_tl in [(0,0), (0, 128), (128, 128), (128, 0),(64,64)]:
            fp, delta = _test_patches_get_point_motion_vector(self, features, p_tl, size = 128)
            try:
                point_pair_ori = np.concatenate([point_pair_ori, fp], axis = 0)
                point_pair_mov = np.concatenate([point_pair_mov, delta], axis = 0)
            except:
                point_pair_ori = fp
                point_pair_mov = delta
        if True:
            features = torch.nn.AdaptiveAvgPool2d(int(h/2))(features)   #128*128用的
            fp, delta = _test_patches_get_point_motion_vector(self, features, (0,0), size = 128)
            point_pair_ori = np.concatenate([point_pair_ori, fp], axis = 0)
            point_pair_mov = np.concatenate([point_pair_mov, delta], axis = 0)
        #
        import matplotlib.pyplot as plt
        ax = plt.gca()
        ax.invert_yaxis()
        for i in range(len(point_pair_ori)):
            x,y = point_pair_ori[i]
            u,v = point_pair_mov[i]
            plt.quiver(x,y,u,v)
        plt.show()
        #
        img_t0_warp = img_t0
        return img_t0_warp

    def _test_one_patch(self, img_t1, img_t0):
        def _test_get_homo_128(self, patch_t0, patch_t1):
            assert(patch_t0.shape == (128, 128, 1))
            patch_t0 = torch.Tensor(patch_t0/255).float().permute(2,0,1)[None]    #hwc->chw
            patch_t1 = torch.Tensor(patch_t1/255).float().permute(2,0,1)[None]
            patch_t0 = patch_t0 * 2 - 1
            patch_t1 = patch_t1 * 2 - 1
            patch_t0 = patch_t0.to(self.device)
            patch_t1 = patch_t1.to(self.device)
            features = self.model_cnn(patch_t0, patch_t1)[4]
            output = self.model_fc(features)
            delta = output[0].detach().cpu().numpy()
            
            fp = np.array([ (32, 32),
                            (160, 32),
                            (160, 160),
                            (32, 160)],
                            dtype=np.float32)
            pfp = np.float32(fp + delta * 8)

            if False:
                import matplotlib.pyplot as plt
                ax = plt.gca()
                ax.invert_yaxis()
                for i in range(4):
                    x,y = fp[i]
                    u,v = pfp[i] - fp[i]
                    plt.quiver(x,y,u,v)
                plt.show()

            H_warp = cv2.getPerspectiveTransform(fp, pfp)
            H_warp_from_0 = cv2.getPerspectiveTransform(fp-32, pfp-32)
            H2 = np.array([ 1, 0, -32,
                            0, 1, -32,
                            0, 0,   1 ]).reshape(3,3) #平移矩阵
            H_warp_patch = np.matmul(np.matmul(H2, H_warp), np.linalg.inv(H2)) 
            return H_warp_patch


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
        
        # 单个patch测试
        H_warp_patch = _test_get_homo_128(self, patch_t0, patch_t1)
        patch_t0_w = cv2.warpPerspective(patch_t0, H_warp_patch, (128,128))
        h,w,_ = patch_t0.shape
        patch_t0 = patch_t0[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95),:]
        patch_t1 = patch_t1[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95),:]
        patch_t0_w = patch_t0_w[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
        cv2.imwrite("0.png", img_t0_gray)
        cv2.imwrite("1.png", patch_t0)
        cv2.imwrite("2.png", patch_t1)
        temp = img_square([patch_t0, patch_t1, patch_t0_w, cv2.absdiff(patch_t1, patch_t0_w), cv2.absdiff(patch_t1, patch_t0)], 1,5)
        cv2.imwrite("3.png", temp)
        cv2.imshow("123",temp)
        cv2.waitKey(0)
        print(np.sum(cv2.absdiff(patch_t1, patch_t0_w)), np.sum(cv2.absdiff(patch_t1, patch_t0)))
        self.ss1 += np.sum(cv2.absdiff(patch_t1, patch_t0_w))
        self.ss2 += np.sum(cv2.absdiff(patch_t1, patch_t0))
        print(self.ss1, self.ss2)
        return img_t0


    def time_test(self, stride=4):
        print("==============")
        print(f"run time testing wiht stride = {stride}")
        #
        path = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4"
        cap = cv2.VideoCapture(path)
        #
        tempVideoProcesser = Inference_VideoProcess(cap=cap,fps_target=30)
        fps = tempVideoProcesser.fps_now
        img_base, img_t0 = tempVideoProcesser()
        
        assert(img_t0.shape == img_base.shape)
        assert(img_t0.shape[2] == 3)
        h, w, _ = img_t0.shape
        assert(w >= h)
        assert(h == 512)
        #
        img_t0_gray = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
        img_base_gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        
        img_t0_gray_resized = cv2.resize(img_t0_gray, (128 * stride, 128 * stride))
        img_base_gray_resized = cv2.resize(img_base_gray, (128 * stride, 128 * stride))
        
        delta = []
        features1 = self._get_fea(img_t0_gray_resized[0::stride, 0::stride], img_base_gray_resized[0::stride, 0::stride])
        delta1 = self._get_output(features1)
        print(delta1.shape)
        
        t = tic()
        for _ in range(300):
            delta = []
            for i in range(stride):
                for j in range(stride):
                    features1 = self._get_fea(img_t0_gray_resized[i::stride, j::stride], img_base_gray_resized[i::stride, j::stride])
                    delta1 = self._get_output(features1)[0]
                    delta.append(delta1)
        print(np.array(delta).shape, delta[0])
        toc(t, "inference * 16", 300, False)

        t = tic()
        for _ in range(300):
            patch_t0s = []
            patch_t1s = []
            for i in range(stride):
                for j in range(stride):
                    patch_t0s.append(img_t0_gray_resized[i::stride, j::stride])
                    patch_t1s.append(img_base_gray_resized[i::stride, j::stride])
            patch_t0s = np.array(patch_t0s)[:,np.newaxis,:,:]
            patch_t1s = np.array(patch_t1s)[:,np.newaxis,:,:]
        
            features1 = self._get_feas_batch(patch_t0s, patch_t1s)
            delta = self._get_output(features1)
        print(np.array(delta).shape, delta[0])
        toc(t, "inference * 16", 300, False)
        for idx, img in enumerate(patch_t0s):
            cv2.imwrite(f"s={stride}_{idx}.png", img[0])
            