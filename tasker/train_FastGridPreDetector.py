import random
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from data.mypath import Path
from data.dataset_JanusUAV_1 import Dataset_JanusUAV
from model.FastGridPreDetector import FastGridPreDetector
from utils.loss import lossFunc_Grid

from tasker._base_tasker import _Tasker_base
from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc


class Train_FastGridPreDetector_and_save(_Tasker_base):
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
        self.model = FastGridPreDetector()
        self.model.to(self.device).train()
        
        # 数据
        print(colorstr('Initializing dataset...', 'yellow'))
        self.batchSize = self.args['batchSize']
        Dataset_generater = Dataset_JanusUAV(Path.db_root_dir('janus_uav'), self.args)
        Dataset_train = Dataset_generater.generate('train')
        self.TrainLoader = DataLoader(Dataset_train, 
                                self.batchSize,
                                num_workers=self.args['numWorkers'],
                                drop_last=False, 
                                pin_memory=True)
        Dataset_valid = Dataset_generater.generate('valid')
        self.ValidLoader = DataLoader(Dataset_valid, 
                                self.batchSize,
                                num_workers=self.args['numWorkers'],
                                drop_last=False, 
                                pin_memory=True)
        
        #优化器
        self.criterion = lossFunc_Grid()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.args['lr_init'])
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr_init'])
        
        #训练参数
        self.iterations = -1
        if 'iterations' in self.args.keys():
            if self.args['iterations']>0:
                self.iterations = self.args['iterations']
                self.epoches = int(self.iterations / len(Dataset_train))
            else:
                self.epoches = self.args['epoches']
        else:
            self.epoches = self.args['epoches']
        print('Initialization complete.')
    
    def run(self):
        print('run!')
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
                    outputs = self.model(imgs) 
                    loss,l1,l2 = self.criterion(outputs, gts)
                #toc(t2,"推理",mute=False)
                self.scaler.scale(loss).backward()
                #scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                #toc(t2,"更新梯度",mute=False)
                
                #
                #每个epoch保留10个提示
                self.logger.add_scalar('train_loss', loss.item(), epoch*len(TrainLoader)+i)
                loss_list.append(loss.item())
                if (i+1) % (max(len(TrainLoader) // 10, 1)) == 0: 
                    temp_l = np.mean(loss_list)
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(TrainLoader)}]-{toc(t_start)}ms\t avg_loss: {temp_l:.4f}')
                    toc(t1,"1/10 of all", (i+1) // (max(len(TrainLoader) // 10, 1)), mute=False)
                else:
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(TrainLoader)}]-{toc(t_start)}ms\t loss: {loss:.4f}=dice({l1:.4f})+focal({l2:.4f})', end="")
                #toc(t2,"消息更新",mute=False)
                
                #每个epoch保存10次图片
                if (i+1) % (max(len(TrainLoader) // 10, 1)) == 0: 
                    sm = nn.Softmax(0)
                    watcher = []
                    watcher.append(imgs[0])
                    watcher.append(gts[0,0])
                    watcher.append(sm(outputs[0][0])[0])
                    watcher.append(sm(outputs[1][0])[0])
                    watcher.append(sm(outputs[2][0])[0])
                    
                    img = img_square(watcher, 2, 3)
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

        self.criterion = lossFunc_Grid()
        
        with torch.no_grad():
            for i, data_item in enumerate(ValidLoader):
                imgs, gts = data_item
                imgs, gts = self.preprocess(imgs, gts)
                #
                outputs = self.model(imgs) #flow_L1, flow_L2, flow_L3, flow_L4
                loss = self.criterion(outputs, gts)[0]
                loss_list.append(loss.item())
                #每个epoch保存10次图片
                if (i+1) % (max(len(ValidLoader) // 10, 1)) == 0: 
                    sm = nn.Softmax(0)
                    watcher = []
                    watcher.append(imgs[0])
                    watcher.append(gts[0,0])
                    watcher.append(sm(outputs[0][0])[0])
                    watcher.append(sm(outputs[1][0])[0])
                    watcher.append(sm(outputs[2][0])[0])
                    
                    img = img_square(watcher, 2, 3)
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
        imgs = imgs.to(self.device).float()
        gts = gts.to(self.device).float()
        return imgs, gts

    def save_model(self, epoch):
        self.model.eval()
        self.model = self.model.cpu()
        state = {   'epoches': epoch,
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
        self.epoch_resume = temp_states['epoches'] + 1       #本轮epoch=上轮+1
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
