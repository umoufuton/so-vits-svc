import torch
from torch import nn
from torch.nn import functional as F
from models import (
    SynthesizerTrn,
)
import utils
from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = ','.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def removeOptimizer(input_model, output_model):
    hps = utils.get_hparams_from_dir("configs")

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(0)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    state_dict_g = torch.load(input_model)
    new_dict_g = copyStateDict(state_dict_g)
    keys = []
    for k,v in new_dict_g['model'].items():
        keys.append(k)

    new_dict_g = {k:new_dict_g['model'][k] for k in keys}

    torch.save({'model': new_dict_g,
                'iteration': 0,
                'optimizer': optim_g.state_dict(),
                'learning_rate': 0.0001}, output_model)