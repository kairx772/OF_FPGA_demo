import torch
import numpy as np
# import check.net_t
# import check.net_np

from models.of_model import FNSg
from models.of_model_np import FNSg_np
from models.of_model_llsq import FNSg_llsq

device = 'cpu'
model = FNSg().to('cpu')
# ckpt = torch.load(path + 'weight.cpt')
#network_data = torch.load('/home/kairx/kairx/OF_CNN/yc_demo/demo/dataset/FNSg/model_best.pth.tar')
# model.load_state_dict(ckpt['net'])
#model.load_state_dict(network_data['state_dict'], strict=False)

model_np = FNSg_np()
print (model_np.__dict__)

# for (of_i, (of_param_tensor)), (index, dict_val) in zip(model_np.__dict__.items(), model.state_dict().items()):
#     print (of_i, index)
#     of_param_tensor.weight = dict_val.numpy()
# print ('====')


#print(model.state_dict()['classifier.0.weight'][0,0])
