import torch
import torch.nn.functional as F
import torch.nn as nn
# import model
import weight.weight_func as w_func
# from weight_func import *
# import weight_func_llsq as wf_llsq
import os

from models.of_model import FNSg
from models.of_model_llsq import FNSg_llsq

# model = of_model_llsq.FNSg_llsq(bitW=8, bitA=8).to('cpu')
model = FNSg().to('cpu')

# network_data = torch.load('/home/kairx/kairx/OF_CNN/yc_demo/demo/dataset/FNSg_llsq/model_best.pth.tar')
network_data = torch.load('/home/kairx/kairx/OF_CNN/yc_demo/demo/dataset/FNSg/model_best.pth.tar')
# model_of.load_state_dict(network_data['state_dict'], strict=False)
model.load_state_dict(network_data['state_dict'], strict=False)
# model.load_state_dict(network_data['state_dict'])

for of_i, (of_param_tensor) in model.state_dict().items():
    print (of_i)
    file_name = 'dataset/FNSg/w_FNSg_npy/' + of_i
    weight = of_param_tensor
    w_func.conv_w_npy(weight, file_name)