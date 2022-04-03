def get_conf(taskerName = "tasker"):
    arg = {}
    arg['ifUseGPU'] = True
    #
    arg['ifDataAugment'] = True
    arg['ifDatasetAllTheSameTrick'] = False
    arg['datasetLenTrick'] = -1
    #
    if "Train_Homo_and_save" in taskerName:
        arg['record'] = True
        arg['ifContinueTask'] = False
        arg['continueTaskExpPath'] = "exps/20220320_13_56_01_Train_Homo_and_save"
        arg['continueWeightsFile'] = "model_Train_Homo_and_save_bs32_78.pkl"
        arg['taskerName'] = "Train_Homo_and_save"
        arg['batchSize'] = 32
        arg['numWorkers'] = 2
        arg['lr_init'] = 0.005
        arg['iterations'] = 8000000

    if "Test_Homo_in_validset" in taskerName:
        arg['record'] = False
        arg['ifContinueTask'] = True
        arg['continueTaskExpPath'] = "weights"
        arg['continueWeightsFile'] = "model_Train_Homo_and_save_bs32_96.pkl"
        arg['taskerName'] = "Test_Homo_in_validset"
        arg['batchSize'] = 32
        arg['numWorkers'] = 2
        arg['lr_init'] = 0.005
        arg['iterations'] = 50

    if "infer_Homo"in taskerName:
        arg['modelType'] = 'weights' #weights, script
        arg['continueTaskExpPath'] = "weights"
        arg['continueWeightsFile_script'] = "Homo_libtorch_script_model.pkl"
        arg['continueWeightsFile_weights'] = "model_Train_Homo_and_save_bs32_96.pkl"
        arg['taskerName'] = "Tester_Homo"

    if "Train_MovingDetection_and_save" in taskerName:
        arg['record'] = True
        arg['ifContinueTask'] = False
        arg['continueTaskExpPath'] = "exps/20220320_13_56_01_Train_Homo_and_save"
        arg['continueWeightsFile'] = "model_Train_Homo_and_save_bs32_78.pkl"
        arg['taskerName'] = "Train_MovingDetection_and_save"
        arg['img_size_h'] = 640
        arg['img_size_w'] = 640
        arg['batchSize'] = 8
        arg['numWorkers'] = 2
        arg['lr_init'] = 0.005
        arg['iterations'] = 800000
    return arg