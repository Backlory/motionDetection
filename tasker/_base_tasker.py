import numpy as np
import torch
import os, sys, sys
import datetime
from tensorboardX import SummaryWriter

from utils.mics import colorstr
from utils.timers import tic, toc

import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        #
        if message == "" or message == "\n" or message == "\r":
            return
        if message[0] == '\r' and message[-1] != "\n":
            message = "\n"
        if message[0] == "\r":
            message = message[1:]
        if message[0] == "\033":
            message = message[5:-4]

        if message != "\n" and message.strip() != "":
            message = "["+ datetime.datetime.now().strftime(r'%Y-%m-%d-%H:%M:%S') + "]: " +  message + "\n"
            self.log.write(message)
    def __del__(self):
        self.log.close()
    def flush(self):
        pass
    def refresh(self):
        self.log.close()
        self.log = open(self.filename, 'a')



class _Tasker_base():
    def __init__(self, args):
        super().__init__()
        
        print(colorstr('Initializing experiment arguments...', 'yellow'))
        self.args = args
        self.experiment_dir = None
        self.save_path = None
        self.device = None
        self.logger = None
        # 脉冲定时器
        self.pulse_start_time = tic()
        # 总实验文件夹
        print(colorstr('Initializing project log folder...', 'yellow'))
        self.experiment_dir = './exps/'
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
        
        # 本次实验文件夹
        if not self.args['record']:
            self.save_path = "temp"
        elif self.args['ifContinueTask']:
            self.save_path = self.args['continueTaskExpPath']
        else:
            print(colorstr('Initializing folder for this experiment...', 'yellow'))
            self.save_path = self.experiment_dir
            self.save_path += datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S_')
            #
            self.save_path += self.args['taskerName']
            #
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            temp = str(self.args)[1:]
            temp = temp.replace('{',        '')
            temp = temp.replace('}',        '')
            temp = temp.replace(',',        '\n')
            temp = temp.replace('\n \'',     '\n')
            temp = temp.replace('\':',      ':')
            temp = temp.replace(':',        ':\t')
            with open(self.save_path+'/config.yaml', 'w') as f:
                f.write(temp)
        sys.stdout = Logger(self.save_path+'\history.log', sys.stdout)
        sys.stderr = Logger(self.save_path+'\history.log_file', sys.stderr)
        # 设备
        print(colorstr('Initializing device...', 'yellow'))
        if self.args['ifUseGPU']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        else:
            self.device = torch.device('cpu')
        
        # 日志记录器
        print(colorstr('Initializing tensorboard summarywriter...', 'yellow'))
        self.logger = SummaryWriter(logdir = self.save_path + '/tsLogger')
        if sys.platform == "linux":
            activate_dir = f'{self.save_path}/tensorboard.sh'
        else:
            activate_dir = f'{self.save_path}/tensorboard.txt'
        with open(activate_dir, 'w') as f:
            f.write(f'activate torchenv1100\n')
            f.write(f'tensorboard --logdir=tsLogger\n')
            f.close()
        if sys.platform == "linux":
            os.system(f'chmod 777 {activate_dir}')
    #
    #
    
    def __del__(self):
        self.logger.close()

    def pulse(self, seconds):
        '''脉冲定时器。如果当前时间超过了seconds秒，则返回脉冲信号True。
        
        短时间内只能用一次
        
        eg.

        if self.pulse(60):
            xxxx
        '''
        if int(toc(self.pulse_start_time)/1000) > seconds:
            self.pulse_start_time = tic()
            return True
        else:
            return False