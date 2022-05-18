


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
        arg = get_conf('Train_Homo_and_save')
        Tasker = Train_Homo_and_save(arg)
        #Tasker.load_pretrain()
    if False:
        from tasker.train_Homo import Train_Homo_and_save
        arg = get_conf('Test_Homo_in_validset')
        Tasker = Train_Homo_and_save(arg)
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
        arg = get_conf('infer_Homo')
        Infer = Inference_Homo_RSHomoNet(arg)
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
        arg = get_conf('infer_Homo')
        Infer = Inference_Homo_RANSAC(arg)
        Infer.time_test()
        for fps in [1,3,6,10,15,30]:
            Infer.run_test(fps)
        exit(0)

    if False:
        from algorithm.infer_Homo_switcher import Inference_Homo_switcher
        log('!!!!!!!!!!!!!!!!!!!!!!!单应性测试，switch算法!!!!!!!!!!!!!!!!!!!!\n')
        arg = get_conf('infer_Homo')
        Infer = Inference_Homo_switcher(arg)
        for ds in ['k','j','u']:
            for fps in [1,3,6,10,15,30]:
                Infer.run_test(fps, ds)
        exit(0)

    if False:
        from tasker.train_FastGridPreDetector import Train_FastGridPreDetector_and_save
        log('!!!!!!!!!!!!!!!!!!!!!!!运动区域分割!!!!!!!!!!!!!!!!!!!!\n')
        arg = get_conf('Train_FastGridPreDetector_and_save')
        #arg['datasetLenTrick'] = 10
        Tasker = Train_FastGridPreDetector_and_save(arg)
        Tasker.run()
        exit(0)

    if False:
        from algorithm.infer_Region_Proposal import Inference_Region_Proposal
        log('!!!!!!!!!!!!!!!!!!!!!!!运动区域分割!!!!!!!!!!!!!!!!!!!!\n')
        arg = get_conf('Inference_Region_Proposal')
        #arg['datasetLenTrick'] = 10
        Tasker = Inference_Region_Proposal(arg=arg)
        Tasker.run_test(dataset = 'j')
        exit(0)

    if False:
        from algorithm.infer_OpticalFlow import Inference_OpticalFlow
        log('!!!!!!!!!!!!!!!!!!!!!!!光流提取!!!!!!!!!!!!!!!!!!!!\n')
        arg = get_conf('Inference_OpticalFlow')
        Tasker = Inference_OpticalFlow(arg=arg)
        Tasker.run_test(dataset = 'j')
        exit(0)
        

    if True:
        from tasker.train_MDHead import Train_MDHead_and_save
        log('!!!!!!!!!!!!!!!!!!!!!!!运动检测Head训练!!!!!!!!!!!!!!!!!!!!\n')
        arg = get_conf('Train_MDHead_and_save')
        #arg['datasetLenTrick'] = 10
        arg['batchSize'] = 8
        arg['lr_init'] = 0.0005
        arg['numWorkers'] = 0
        #arg['ifDatasetAllTheSameTrick'] = True
        Tasker = Train_MDHead_and_save(arg)
        Tasker.run()
        exit(0)
    