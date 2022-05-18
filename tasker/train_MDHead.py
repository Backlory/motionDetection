import random
import os, sys
from tqdm import trange
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.mypath import Path
from data.dataset_JanusFLOW import Dataset_JanusFLOW
from model.MDHead import MDHead

from tasker._base_tasker import _Tasker_base

from utils.indicator import Evaluator
from utils.loss import loss_Dice_Focal
from utils.img_display import save_pic, img_square
from utils.mics import colorstr
from utils.timers import tic, toc


class Train_MDHead_and_save(_Tasker_base):
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
        self.model = MDHead()
        self.model.to(self.device).train()
        
        # 数据
        print(colorstr('Initializing dataset...', 'yellow'))
        self.batchSize = self.args['batchSize']
        Dataset_generater = Dataset_JanusFLOW(Path.db_root_dir('janus_uav'), self.args)
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
        self.criterion = loss_Dice_Focal()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.args['lr_init'])
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=20)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10,
        verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08)
        
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

        #
    
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
            try:
                print(f"lr_scheduler.lr = {self.scheduler.get_last_lr()[0]:.7f}")
            except:
                pass
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
                inputs, targets = data_item
                inputs, targets = self.preprocess(inputs, targets)
                #toc(t2,"预处理",mute=False)
                #
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(inputs[0], inputs[1]) 
                    loss,l1,l2 = self.criterion(outputs, targets)
                #toc(t2,"推理",mute=False)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                #self.scheduler.step()
                #toc(t2,"更新梯度",mute=False)
                
                #
                #每个epoch保留10个提示
                self.logger.add_scalar('train_loss', loss.item(), epoch*len(TrainLoader)+i)
                loss_list.append(loss.item())
                if (i+1) % (max(len(TrainLoader) // 10, 1)) == 0: 
                    temp_l = np.mean(loss_list)
                    print('\r',' ' * 100, end="")
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(TrainLoader)}]-{toc(t_start)} ms\t avg_loss: {temp_l:.4f}\n', end="")
                    #toc(t1,"1/10 of all", (i+1) // (max(len(TrainLoader) // 10, 1)), mute=False)
                else:
                    print('\r',' ' * 100, end="")
                    print(f'\rEpoch[{epoch+1}/{self.epoches}][{i}/{len(TrainLoader)}]-{toc(t_start)} ms\t loss: {loss:.4f}=dice({l1:.4f})+focal({l2:.4f})', end="")
                #toc(t2,"消息更新",mute=False)
                
                #每个epoch保存10次图片
                if (i+1) % (max(len(TrainLoader) // 10, 1)) == 0: 
                    outputs = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()
                    watcher  = [outputs[0,0], outputs[1,0], outputs[2,0], outputs[3,0]]
                    watcher += [targets[0,0], targets[1,0], targets[2,0], targets[3,0]]
                    img = img_square(watcher, 2)
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
            if ((epoch + 1) % max(self.epoches//5, 1) == 0):
                self.valid(epoch, True)
            else:
                self.valid(epoch)

            try:
                sys.stdout.refresh()
            except:
                pass
            
            epoch += 1
        temp = self.get_weight_save_dir(epoch)
        self.args['continue_states_path'] = temp

    def valid(self, epoch=0, confuse_coculate = False):
        ValidLoader = self.ValidLoader
        self.model.eval()
        loss_list = []

        print( colorstr(f'preparing device...', 'yellow') )
        self.model.to(self.device).eval()
        
        if self.args['ifContinueTask']:
            self.continueTask()
            epoch = self.epoch_resume

        self.criterion = loss_Dice_Focal()
        self.evaluator = Evaluator(2)
        with torch.no_grad():
            print( colorstr(f'validing...', 'yellow') )
            for i, data_item in enumerate(ValidLoader):
                inputs, target = data_item
                inputs, targets = self.preprocess(inputs, target)
                #
                outputs = self.model(inputs[0], inputs[1]) 
                loss = self.criterion(outputs, targets)[0]
                loss_list.append(loss.item())

                if confuse_coculate:
                    self.evaluator.add_batch(outputs, targets)
                #每个epoch保存10次图片
                if (i+1) % (max(len(ValidLoader) // 10, 1)) == 0: 
                    if i < max(len(ValidLoader) // 10 * 10, len(ValidLoader))-1:
                        print(f"\r{i+1}/{len(ValidLoader)}...", end="")
                    else:
                        print(f"\r{i+1}/{len(ValidLoader)}...")
                    outputs = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()
                    watcher  = [outputs[0,0], outputs[1,0], outputs[2,0], outputs[3,0]]
                    watcher += [targets[0,0], targets[1,0], targets[2,0], targets[3,0]]
                    img = img_square(watcher, 2, 4)
                    cv2.imwrite(self.save_path+f'/val_cycle_{i}.png', img)
                    self.logger.add_image(f'valid_img_sample',
                                            img, 
                                            epoch*len(ValidLoader)+i, 
                                            dataformats='HWC')
        temp_l = np.mean(loss_list)
        self.scheduler.step(temp_l)
        print(f'Epoch[{epoch+1}/{self.epoches}][{i}/{len(ValidLoader)}]\t avg_loss_val: {temp_l:.4f}')
        
        if confuse_coculate:
            mIoU, FWIoU, Acc, mAcc, mPre, mRecall, mF1, AuC = self.evaluator.evaluateAll()
            print(self.evaluator.confusion_matrix)
            print(f"mIoU={mIoU:.4f}, FWIoU={FWIoU:.4f}, Acc={Acc:.4f}, mAcc={mAcc:.4f}")
            print(f"mPre={mPre:.4f}, mRecall={mRecall:.4f}, mF1={mF1:.4f}, AuC={AuC:.4f}")

        self.logger.add_scalar('valid_loss_avg', 
                                np.mean(loss_list), 
                                epoch+1)
            

    def preprocess(self, inputs, target):  #
        '''
        将数据分发的结果送入GPU，转置，转float
        '''
        inputs = [x.to(self.device).float() for x in inputs]
        target = target.to(self.device).float()
        
        return inputs, target

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
