from tasker.train_Homo import Train_Homo_and_save
from tasker.train_MovingDetection import Train_MovingDetection_and_save

from algorithm.infer_VideoProcess import Inference_VideoProcess
from algorithm.infer_Homo import Inference_Homo

from utils.conf import get_conf
import datetime

if __name__ == "__main__":
    #
    if False:
        args = get_conf('Train_Homo_and_save')
        Tasker = Train_Homo_and_save(args)
        #Tasker.load_pretrain()
    if False:
        args = get_conf('Test_Homo_in_validset')
        Tasker = Train_Homo_and_save(args)
        Tasker.valid(0)
        Tasker.run()
        exit(0)
    #
    if False:
        Infer = Inference_VideoProcess()
        Infer.run_test()
        exit(0)
    if False:
        with open("log.txt", "a+") as f:
            f.write( datetime.datetime.now().strftime(r'%Y-%m-%d-%H:%M:%S'))
            f.writelines("!!!!!!!!!!!!!!!!!!!!!!!单应性变化情况!!!!!!!!!!!!!!!!!!!!\n")
        args = get_conf('infer_Homo')
        Infer = Inference_Homo(args)
        #Infer.time_test(4)
        #Infer.time_test(3)
        #Infer.time_test(2)
        #Infer.time_test(1)
        for alpha in [0]:
            for fps in [1,3,6,10,15,30]:
                for stride in [1,2,3,4]:
                    Infer.run_test(fps, stride, alpha)
        exit(0)

    if True:
        with open("log.txt", "a+") as f:
            f.write( datetime.datetime.now().strftime(r'%Y-%m-%d-%H:%M:%S'))
            f.writelines(": -> !!!!!!!!!!!!!!!!!!!!!!!运动感知!!!!!!!!!!!!!!!!!!!!\n")
        
        args = get_conf('Train_MovingDetection_and_save')
        Tasker = Train_MovingDetection_and_save(args)
        Tasker.run()
    