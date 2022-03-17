def get_conf(taskerName = "tasker"):
    arg = {}
    arg['ifContinueTask'] = False
    arg['continueTaskExpPath'] = ""
    arg['continueWeightsFile'] = ""
    #
    arg['ifUseGPU'] = True
    #
    arg['ifDataAugment'] = True
    arg['ifDatasetAllTheSameTrick'] = False
    arg['datasetLenTrick'] = -1
    #
    if "Train_Homo_and_save" in taskerName:
        arg['taskerName'] = "Train_Homo_and_save"
        arg['batchSize'] = 32
        arg['numWorkers'] = 2
        arg['lr_init'] = 0.005
        arg['iterations'] = 4000000

    return arg