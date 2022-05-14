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

    if "Train_FastGridPreDetector_and_save" in taskerName:
        arg['record'] = True
        arg['ifContinueTask'] = True
        arg['continueTaskExpPath'] = "exps/20220426_12_01_36_Train_FastGridPreDetector_and_save"
        arg['continueWeightsFile'] = "model_Train_FastGridPreDetector_and_save_bs8_55.pkl"
        arg['taskerName'] = "Train_FastGridPreDetector_and_save"
        arg['batchSize'] = 16
        arg['numWorkers'] = 1
        arg['lr_init'] = 0.00001
        arg['iterations'] = -1
        arg['epoches'] = 300

    if "Inference_Region_Proposal" in taskerName:
        arg['modelType'] = 'weights' #weights, script
        arg['continueTaskExpPath'] = "weights"
        arg['continueWeightsFile_weights'] = "model_Train_Homo_and_save_bs32_96.pkl"
        arg['taskerName'] = "Tester_Homo"

    if "Inference_OpticalFlow" in taskerName:
        arg['modelType'] = 'weights' #weights, script
        arg['continueTaskExpPath'] = "weights"
        arg['continueWeightsFile_weights'] = "model_Train_Homo_and_save_bs32_96.pkl"
        arg['taskerName'] = "Tester_Homo"
        
        arg["RAFT_model"] = "model/thirdparty_RAFT/model/raft-sintel.pth"
        arg["RAFT_path"] = r"E:\dataset\dataset-fg-det\Janus_UAV_Dataset\Train\video_1\video"
        arg["RAFT_small"] = "store_true"    #store_true代表False
        arg["RAFT_mixed_precision"] = "store_false"
        arg["RAFT_alternate_corr"] = "store_true"


    return arg