import random
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from data.mypath import Path
from data.dataset_COCO2017 import Dataset_COCO2017
from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc

from tasker._base_tasker import _Tasker_base
from model.Homo import HomographyNet
class Train_Homo_and_save(_Tasker_base):
    def __init__(self, args):
        super().__init__(args)
        '''
        self.args
        self.experiment_dir
        self.save_path
        self.device
        self.logger
        '''
        # 模型
        print(colorstr('Initializing model...', 'yellow'))
        self.model = HomographyNet()
        self.model.to(self.device).train()
        
        # 数据
        print(colorstr('Initializing dataset...', 'yellow'))
        self.batchSize = self.args['batchSize']
        Dataset_generater = Dataset_COCO2017(Path.db_root_dir('coco'), self.args)
        Dataset_train = Dataset_generater.generate('train')
        self.TrainLoader = DataLoader(Dataset_train, 
                                self.batchSize,
                                num_workers=self.args['numWorkers'],
                                drop_last=True, 
                                pin_memory=True)
        Dataset_valid = Dataset_generater.generate('valid')
        self.ValidLoader = DataLoader(Dataset_valid, 
                                self.batchSize,
                                num_workers=self.args['numWorkers'],
                                drop_last=True, 
                                pin_memory=True)
        
        #优化器
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(),lr=self.args['lr_init'], momentum=0.9)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr_init'])
        
        #训练参数
        if 'iterations' in self.args.keys():
            self.iterations = self.args['iterations']
            self.epoches = int(self.iterations / len(Dataset_train))
        else:
            self.iterations = -1
            self.epoches = self.args['epoch']
        print('Initialization complete.')
    
    def run(self):
        print('run!')
        lr_type = 'exp10'    #exp10, step
        lr_steps = [0, 10, 20, 30, 40, 9999], 
        lr_scale = [1, 0.5, 0.5, 0.5, 0.5,0.5]
        epoch = 0
        
        # 加载模型
        if self.args['ifContinueTask']:
            self.continueTask()

        #半精度优化
        scaler = GradScaler(enabled=True)

        best_loss = 9999
        while epoch < self.epoches:
            print( colorstr(f'\n======== epoch = {epoch+1} ========', 'yellow') )
            # lr手动调整
            if lr_type == 'step':
                for i in range(len(lr_steps)-1):
                    if epoch == lr_steps[i]:
                        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * lr_scale[i]
                        break
                print('lr will change at epoch ', lr_steps, '.')
            elif lr_type == 'exp10':
                lr = 10**(-epoch/self.epoches *2 - 4)    #1e-4 ~ 1e-8
                self.optimizer.param_groups[0]['lr'] = lr
            print('optimizer = ', str(self.optimizer), '\nlr = ', self.optimizer.param_groups[0]['lr'])
            self.logger.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], epoch)
            #
            ########## train
            #self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
            #
            TrainLoader = self.TrainLoader
            self.model.train()
            loss_list = []
            print('Loading data...')
            t1 = tic()
            for i, data_item in enumerate(self.TrainLoader):
                self.logger.add_scalar('learning rate realtime', 
                                        self.optimizer.param_groups[0]['lr'], 
                                        epoch*len(TrainLoader)+i)
                img_t0, patch_t0, patch_t1, target = data_item
                patch_t0, patch_t1, target = self.preprocess(patch_t0, patch_t1, target)
                #
                self.optimizer.zero_grad()
                output = self.model(patch_t0, patch_t1) #flow_L1, flow_L2, flow_L3, flow_L4
                
                loss = self.criterion(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)
                scaler.update()
                
                #loss.backward()
                #self.optimizer.step()
                #
                #每个epoch保留10个提示
                self.logger.add_scalar('train_loss', loss.item(), epoch*len(TrainLoader)+i)
                loss_list.append(loss.item())
                print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(TrainLoader)}]\t loss: {loss:.4f}', end="")
                
                if (i+1) % (max(len(TrainLoader) // 10, 1)) == 0: 
                    temp_l = np.mean(loss_list)
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(TrainLoader)}]\t avg_loss: {temp_l:.4f}')
                    toc(t1,"1/10 of all", (i+1) // (max(len(TrainLoader) // 10, 1)), mute=False)
                    
                tensor2np = lambda x:((x/2+0.5).detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8)
                #每个epoch保存10次图片
                if (i+1) % (max(len(TrainLoader) // 10, 1)) == 0: 
                    p0 = tensor2np(patch_t0[0])
                    p1 = tensor2np(patch_t1[0])
                    i0 = (img_t0[0].detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8)
                    delta = output[0].detach().cpu().numpy()
                    p0_w = self.output2patch(p0, delta)
                    watcher = [p0, p1, p0_w, cv2.subtract(p0_w, p1)]
                    #watcher = [img_t0[0], img_t1[0]]
                    img = img_square(watcher, 2, 2)
                    cv2.imwrite(self.save_path+f'/tri_cycle_{i}.png', img)
                    self.logger.add_image(f'train_img_sample',
                                            img, 
                                            epoch*len(TrainLoader)+i, 
                                            dataformats='HWC')
            
            self.logger.add_scalar('train_loss_avg', 
                                    np.mean(loss_list), 
                                    epoch+1)
            toc(t1,"one epoch", mute=False)
            # 保存
            self.model.eval()
            self.save_model(epoch)
            
            #
            ######## validate
            print("validating..")
            self.valid(epoch)
            
            epoch += 1
        temp = self.get_weight_save_dir(epoch)
        self.args['continue_states_path'] = temp

    def valid(self, epoch=0):
        ValidLoader = self.ValidLoader
        self.model.eval()
        loss_list = []

        print( colorstr(f'preparing device...', 'yellow') )
        self.model.to(self.device).eval()
        
        if self.args['ifContinueTask']:
            self.continueTask()

        criterion = torch.nn.MSELoss()
        
        with torch.no_grad():
            for i, data_item in enumerate(ValidLoader):
                img_t0, patch_t0, patch_t1, target = data_item
                patch_t0, patch_t1, target = self.preprocess(patch_t0, patch_t1, target)
                #
                output = self.model(patch_t0, patch_t1) #flow_L1, flow_L2, flow_L3, flow_L4
                loss = criterion(output, target)
                #
                #每个epoch保留10个提示
                self.logger.add_scalar('val_loss', loss.item(), epoch*len(ValidLoader)+i)
                loss_list.append(loss.item())
                print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(ValidLoader)}]\t loss: {loss:.4f}', end="")
                if (i+1) % (max(len(ValidLoader) // 10, 1)) == 0 or self.pulse(60): 
                    temp_l = np.mean(loss_list)
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(ValidLoader)}]\t avg_loss: {temp_l:.4f}')
                
                tensor2np = lambda x:((x/2+0.5).detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8)
                #每个epoch保存10次图片
                if (i+1) % (max(len(ValidLoader) // 10, 1)) == 0: 
                    p0 = tensor2np(patch_t0[0])
                    p1 = tensor2np(patch_t1[0])
                    i0 = (img_t0[0].detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8)
                    delta = output[0].detach().cpu().numpy()
                    p0_w = self.output2patch(p0, delta)
                    watcher = [p0, p1, p0_w, cv2.subtract(p0_w, p1)]
                    #watcher = [img_t0[0], img_t1[0]]
                    img = img_square(watcher, 2, 2)
                    cv2.imwrite(self.save_path+f'/val_cycle_{i}.png', img)
                    self.logger.add_image(f'val_img_sample',
                                            img, 
                                            epoch*len(ValidLoader)+i, 
                                            dataformats='HWC')
        self.logger.add_scalar('valid_loss_avg', 
                                np.mean(loss_list), 
                                epoch+1)
            

    def preprocess(self, img_t0, img_t1, target):  #
        '''
        将数据分发的结果送入GPU，转置，转float
        '''
        img_t0 = img_t0.to(self.device).float()
        img_t1 = img_t1.to(self.device).float()
        target = target.to(self.device).float()
        #
        img_t0 = torch.nn.functional.interpolate(   input=img_t0, size=(128, 128), 
                                                    mode='bilinear', align_corners=False)
        img_t1 = torch.nn.functional.interpolate(   input=img_t1, size=(128, 128), 
                                                    mode='bilinear',align_corners=False)
        #
        return img_t0, img_t1, target

    def save_model(self, epoch):
        self.model.eval()
        self.model.cpu()
        state = {   'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer_dict': self.optimizer.state_dict() }
        temp = self.get_weight_save_dir(epoch)
        torch.save(state, temp)
        print(f"\nmodel saved at {temp}.")
        #
        img_t0 = torch.rand(1, 1, 128, 128)
        img_t1 = torch.rand(1, 1, 128, 128)
        traced_model = torch.jit.trace(self.model, (img_t0, img_t1))
        traced_model.save(F'{self.save_path}/libtorch_jit_model.pkl')
        script_model = torch.jit.script(self.model)
        script_model.save(F'{self.save_path}/libtorch_script_model.pkl')
        self.model.to(self.device)
        

    def model2onnx(self):
        print('loading state dicts of model ...')
        temp_states = torch.load(self.args['continue_exp_path'] + '/' + self.args['continue_states_name'])
        self.model.load_state_dict(temp_states['state_dict'])
        self.args['continue_train'] = False

        import os, sys
        onnx_model_path = "models"
        onnx_model_name = 'model.onnx'
        full_model_path = os.path.join(onnx_model_path, onnx_model_name)
        generated_input = ( torch.autograd.Variable(torch.randn(1, 3, 640, 640).cuda()),
                            torch.autograd.Variable(torch.randn(1, 3, 640, 640).cuda())
                            )
        torch.onnx.export(self.model,
                            generated_input,
                            full_model_path,
                            verbose=True,
                            input_names=["input"],
                            output_names=["output"],
                            opset_version=11
                        )

    def get_weight_save_dir(self, epoch):
        '''获取模型保存的路径。'''
        name = self.args["taskerName"]
        bs = self.args["batchSize"]
        temp = f'{self.save_path}/model_{name}_bs{bs}_{epoch+1}.pkl'
        return temp

    def continueTask(self):
        print('loading state dicts of model and optimizer ...')
        temp_states = torch.load(self.args['continueTaskExpPath'] + '/' + self.args['continueWeightsFile'])
        epoch = temp_states['epoch'] + 1       #本轮epoch=上轮+1
        self.model.load_state_dict(temp_states['state_dict'])
        self.optimizer.load_state_dict(temp_states['optimizer_dict'])
        self.args['ifContinueTask'] = False

    def output2patch(self, patch_t0, delta):
        
        ps = 128
        fp = np.array([(0.25,0.25),(1.25,0.25),(1.25,1.25),(0.25,1.25)],
                        dtype=np.float32) * ps
        pfp = np.float32(fp + delta /16 * ps)
        H_warp = cv2.getPerspectiveTransform(fp, pfp)
        #
        H2 = np.array([1,0,-ps*0.25, 0,1, -ps*0.25,0,0,1]).reshape(3,3)
        H_warp_patch = np.matmul(np.matmul(H2, H_warp), np.linalg.inv(H2)) 
        patch_t0_w = cv2.warpPerspective(patch_t0, H_warp_patch, (ps, ps))
        #
        #img_t0_w = cv2.warpPerspective(img_t0, H_warp, (ps*2, ps*2))
        #patch_t0_w = img_t0_w[int(0.25*ps):int(1.25*ps), int(0.25*ps):int(1.25*ps)][:,:,np.newaxis]
        
        return patch_t0_w

    def load_pretrain(self):
        import os
        state_dict = self.model.state_dict()
        state_dict_pretrained = torch.load(os.path.join('weights','pretrained_homo_coco.pt'))['state_dict']
        state_dict_update = {}
        for i in range(20):
            ori_sd = list(state_dict.keys())
            pre_sd = list(state_dict_pretrained.keys())
            state_dict_update[ori_sd[i]] = state_dict_pretrained[pre_sd[i]]
        self.model.load_state_dict(state_dict_update)
        return