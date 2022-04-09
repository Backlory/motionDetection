import random
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.mypath import Path
from data.dataset_JanusUAV_5 import Dataset_JanusUAV
from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc

from tasker._base_tasker import _Tasker_base
from model.MovingDetect import MovingDetectNet

class Train_MovingDetection_and_save(_Tasker_base):
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
        self.model = MovingDetectNet()
        self.model.to(self.device).train()
        
        # 数据
        print(colorstr('Initializing dataset...', 'yellow'))
        self.batchSize = self.args['batchSize']
        Dataset_generater = Dataset_JanusUAV(Path.db_root_dir('janus_uav'), self.args)
        Dataset_train = Dataset_generater.generate('train')
        self.TrainLoader = DataLoader(Dataset_train, 
                                self.batchSize,
                                num_workers=self.args['numWorkers'],
                                drop_last=True, 
                                pin_memory=False)
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
            epoch = self.epoch_resume

        #半精度优化
        try:
            assert(self.device.type == "cuda")
            from torch.cuda.amp import GradScaler, autocast
        except:
            print("no cuda device available, so amp will be forbidden.")
            from utils.cuda import GradScaler, autocast
        self.scaler = GradScaler(enabled=True)

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
                t_start = tic()
                self.logger.add_scalar('learning rate realtime', 
                                        self.optimizer.param_groups[0]['lr'], 
                                        epoch*len(TrainLoader)+i)
                imgs, gts = data_item
                imgs, gts = self.preprocess(imgs, gts)
                #toc(t2,"预处理",mute=False)
                #
                self.optimizer.zero_grad()
                
                with autocast():
                    self.model.clear()
                    loss = 0
                    outputs = []
                    for idx in range(len(imgs) - 1):
                        output = self.model(imgs[idx], imgs[idx+1]) 
                        outputs.append(output)
                        loss = loss + self.criterion(output, gts[idx])
                #toc(t2,"推理",mute=False)
                self.scaler.scale(loss).backward()
                #scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                #toc(t2,"更新梯度",mute=False)
                
                #loss.backward()
                #self.optimizer.step()
                #
                #每个epoch保留10个提示
                self.logger.add_scalar('train_loss', loss.item(), epoch*len(TrainLoader)+i)
                loss_list.append(loss.item())
                if (i+1) % (max(len(TrainLoader) // 10, 1)) == 0: 
                    temp_l = np.mean(loss_list)
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(TrainLoader)}]-{toc(t_start)}ms\t avg_loss: {temp_l:.4f}')
                    toc(t1,"1/10 of all", (i+1) // (max(len(TrainLoader) // 10, 1)), mute=False)
                else:
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(TrainLoader)}]-{toc(t_start)}ms\t loss: {loss:.4f}', end="")
                #toc(t2,"消息更新",mute=False)
                
                #每个epoch保存10次图片
                if (i+1) % (max(len(TrainLoader) // 10, 1)) == 0: 
                    tensors2np = lambda x:(x[0].detach().cpu().numpy()).transpose(1,2,0).astype(np.uint8)
                    temp1 = [tensors2np(img*255) for img in imgs]
                    temp2 = [tensors2np(img*255) for img in outputs] + [None]
                    temp3 = [tensors2np(img*255) for img in gts]
                    watcher = temp1 + temp2 + temp3
                    img = img_square(watcher, 3, 5)
                    cv2.imwrite(self.save_path+f'/tri_cycle_{i}.png', img)
                    self.logger.add_image(f'train_img_sample',
                                            img, 
                                            epoch*len(TrainLoader)+i, 
                                            dataformats='HWC')
                #toc(t2,"保存图片",mute=False)
                
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
            epoch = self.epoch_resume

        criterion = torch.nn.MSELoss()
        
        with torch.no_grad():
            for i, data_item in enumerate(ValidLoader):
                imgs, gts = data_item
                imgs, gts = self.preprocess(imgs, gts)
                #
                self.model.clear()
                loss = 0
                outputs = []
                for idx in range(len(imgs) - 1):
                    output = self.model(imgs[idx], imgs[idx+1]) #flow_L1, flow_L2, flow_L3, flow_L4
                    outputs.append(output)
                    loss = loss + self.criterion(output, gts[idx])
                #
                #每个epoch保留10个提示
                self.logger.add_scalar('val_loss', loss.item(), epoch*len(ValidLoader)+i)
                loss_list.append(loss.item())
                if (i+1) % (max(len(ValidLoader) // 10, 1)) == 0 or self.pulse(60): 
                    temp_l = np.mean(loss_list)
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(ValidLoader)}]\t avg_loss: {temp_l:.4f}')
                else:
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(ValidLoader)}]\t loss: {loss:.4f}', end="")
                
                #每个epoch保存10次图片
                if (i+1) % (max(len(ValidLoader) // 10, 1)) == 0: 
                    tensors2np = lambda x:(x[0].detach().cpu().numpy()).transpose(1,2,0).astype(np.uint8)
                    temp1 = [tensors2np(img*255) for img in imgs]
                    temp2 = [tensors2np(img*255) for img in outputs] + [None]
                    temp3 = [tensors2np(img*255) for img in gts]
                    watcher = temp1 + temp2 + temp3
                    #watcher = [img_t0[0], img_t1[0]]
                    img = img_square(watcher, 3, 5)
                    cv2.imwrite(self.save_path+f'/val_cycle_{i}.png', img)
                    self.logger.add_image(f'valid_img_sample',
                                            img, 
                                            epoch*len(ValidLoader)+i, 
                                            dataformats='HWC')
        
        temp_l = np.mean(loss_list)
        print("\n========")
        print(f'Epoch[{epoch+1}/{self.epoches}][{i}/{len(ValidLoader)}]\t avg_loss: {temp_l:.4f}')
                
        self.logger.add_scalar('valid_loss_avg', 
                                np.mean(loss_list), 
                                epoch+1)
            

    def preprocess(self, imgs, gts):  #
        '''
        将数据分发的结果送入GPU，转置，转float
        '''
        for i in range(len(imgs)):
            imgs[i] = imgs[i].to(self.device).float()
            gts[i] = gts[i].to(self.device).float()
        return imgs, gts

    def save_model(self, epoch):
        self.model.eval()
        self.model = self.model.cpu()
        state = {   'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer_dict': self.optimizer.state_dict() }
        temp = self.get_weight_save_dir(epoch)
        torch.save(state, temp)
        print(f"\nmodel saved at {temp}.")
        #
        self.model.to(self.device)
        

    
    def get_weight_save_dir(self, epoch):
        '''获取模型保存的路径。'''
        name = self.args["taskerName"]
        bs = self.args["batchSize"]
        temp = f'{self.save_path}/model_{name}_bs{bs}_{epoch+1}.pkl'
        return temp

    def continueTask(self):
        print('loading state dicts of model and optimizer ...')
        temp_states = torch.load(self.args['continueTaskExpPath'] + '/' + self.args['continueWeightsFile'])
        self.epoch_resume = temp_states['epoch'] + 1       #本轮epoch=上轮+1
        self.model.load_state_dict(temp_states['state_dict'])
        self.optimizer.load_state_dict(temp_states['optimizer_dict'])
        self.args['ifContinueTask'] = False

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
