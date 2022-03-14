def get_conf(taskerName = "tasker"):
    arg = {}
    arg['ifContinueTask'] = False
    arg['continueTaskExpPath'] = None
    arg['ifUseGPU'] = True
    #
    if "Train_Homo_and_save" in taskerName:
        arg['taskerName'] = "Train_Homo_and_save"

    return arg