from tasker.train_Homo import Train_Homo_and_save
#from tasker.train_MovingDetection import Train_MovingDetection_and_save

from algorithm.infer_VideoProcess import Inference_VideoProcess
from algorithm.infer_Homo import Inference_Homo

from utils.conf import get_conf


if __name__ == "__main__":
    #
    if False:
        args = get_conf('Train_Homo_and_save')
        Tasker = Train_Homo_and_save(args)
        #Tasker.load_pretrain()
        Tasker.run()
    if True:
        args = get_conf('Test_Homo_in_validset')
        Tasker = Train_Homo_and_save(args)
        Tasker.valid(0)
        exit(0)
    if False:
        Infer = Inference_VideoProcess()
        Infer.run_test()
        exit(0)
    if True:
        args = get_conf('infer_Homo')
        Infer = Inference_Homo(args)
        Infer.run_test()
        exit(0)

    if True:
        args = get_conf('Train_MovingDetection')
        Tasker = Train_MovingDetection_and_save(args)
        Tasker.run()