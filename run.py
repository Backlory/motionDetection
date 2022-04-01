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
    if True:
        with open("log.txt", "a+") as f:
            f.writelines("!!!!!!!!!!!!!!!!!!!!!!!单应性变化情况!!!!!!!!!!!!!!!!!!!!\n")
            f.writelines("\n")
        args = get_conf('infer_Homo')
        Infer = Inference_Homo(args)
        Infer.time_test()
        for alpha in [0]:
            for stride in [4]:
                Infer.run_test(1, stride, alpha)
                Infer.run_test(3, stride, alpha)
                Infer.run_test(6, stride, alpha)
                Infer.run_test(10, stride, alpha)
                Infer.run_test(15, stride, alpha)
                Infer.run_test(30, stride, alpha)
        '''
        '''
        exit(0)

    if True:
        args = get_conf('Train_MovingDetection')
        Tasker = Train_MovingDetection_and_save(args)
        Tasker.run()