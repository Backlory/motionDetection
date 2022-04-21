import torch
import torch.nn as nn

def updata_adaptive(model, params_dict_new):
    '''
    from pytorchmodel import updata_adaptive

    model = model()
    params_dict_new = torch.load("1.pt")
    for k,v in params_dict_new.items():
        k_ = k
        params_dict_new[k_] = v
    model = updata_adaptive(model, params_dict_new)
    '''
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in params_dict_new.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model


def traced_model(model, dirname="temp/libtorch_script_model_Homo_cpu.pkl'"):
    model.eval()
    model = model.cpu()
    script_model = torch.jit.script(model)
    script_model.save(dirname)
    return script_model