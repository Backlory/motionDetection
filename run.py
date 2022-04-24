

from tasker.train_FastGridPreDetector import Train_FastGridPreDetector_and_save

from utils.conf import get_conf
import datetime

def log(strs):
    with open("log.txt", "a+") as f:
        f.write( datetime.datetime.now().strftime(r'%Y-%m-%d-%H:%M:%S'))
        f.writelines(strs)

if __name__ == "__main__":
    #
    if False:
        from tasker.train_Homo import Train_Homo_and_save
        args = get_conf('Train_Homo_and_save')
        Tasker = Train_Homo_and_save(args)
        #Tasker.load_pretrain()
    if False:
        from tasker.train_Homo import Train_Homo_and_save
        args = get_conf('Test_Homo_in_validset')
        Tasker = Train_Homo_and_save(args)
        Tasker.valid(0)
        Tasker.run()
        exit(0)
    #
    if False:
        from algorithm.infer_VideoProcess import Inference_VideoProcess
        Infer = Inference_VideoProcess()
        Infer.run_test()
        exit(0)
    if False:
        from algorithm.infer_Homo_RSHomoNet import Inference_Homo_RSHomoNet
        with open("log.txt", "a+") as f:
            f.write( datetime.datetime.now().strftime(r'%Y-%m-%d-%H:%M:%S'))
            f.writelines("!!!!!!!!!!!!!!!!!!!!!!!单应性变化情况!!!!!!!!!!!!!!!!!!!!\n")
        args = get_conf('infer_Homo')
        Infer = Inference_Homo_RSHomoNet(args)
        Infer.time_test(4)
        Infer.time_test(3)
        Infer.time_test(2)
        Infer.time_test(1)
        for alpha in [0]:
            for fps in [1,3,6,10,15,30]:
                for stride in [1,2,3,4]:
                    Infer.run_test(fps, stride, alpha)
        #最终stride取2
        exit(0)
    if False:
        from algorithm.infer_Homo_ransac import Inference_Homo_RANSAC
        log('!!!!!!!!!!!!!!!!!!!!!!!单应性测试，RANSAC算法!!!!!!!!!!!!!!!!!!!!\n')
        args = get_conf('infer_Homo')
        Infer = Inference_Homo_RANSAC(args)
        Infer.time_test()
        for fps in [1,3,6,10,15,30]:
            Infer.run_test(fps)
        exit(0)

    if False:
        from algorithm.infer_Homo_switcher import Inference_Homo_switcher
        log('!!!!!!!!!!!!!!!!!!!!!!!单应性测试，switch算法!!!!!!!!!!!!!!!!!!!!\n')
        args = get_conf('infer_Homo')
        Infer = Inference_Homo_switcher(args)
        for ds in ['k','j','u']:
            for fps in [1,3,6,10,15,30]:
                Infer.run_test(fps, ds)
        exit(0)

        
    if True:
        log('!!!!!!!!!!!!!!!!!!!!!!!运动区域分割!!!!!!!!!!!!!!!!!!!!\n')
        args = get_conf('Train_FastGridPreDetector_and_save')
        #args['datasetLenTrick'] = 10
        Tasker = Train_FastGridPreDetector_and_save(args)
        Tasker.run()
        exit(0)
    