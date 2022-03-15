def get_conf(taskerName = "tasker"):
    arg = {}
    arg['ifContinueTask'] = False
    arg['continueTaskExpPath'] = None
    arg['ifUseGPU'] = True
    arg['ifDatasetAllTheSameTrick'] = False
    arg['ifDataAugment'] = True
    arg['datasetLenTrick'] = -1
    #
    if "Train_Homo_and_save" in taskerName:
        arg['taskerName'] = "Train_Homo_and_save"

    return arg