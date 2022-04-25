from collections import OrderedDict
import torch
from torch import nn

from Nets import ResBlock1

def make_net_linear(dims: list):
    layer_dict = OrderedDict()
    for k in range(len(dims) - 1):
        layer_dict[f'linear{k}'] = nn.Linear(in_features=dims[k], out_features=dims[k + 1])
        layer_dict[f'relu{k}'] = nn.ReLU()
    return nn.Sequential(layer_dict)

model = make_net_linear([28*28,100,100,10])
#print(model)

def make_resnet(dims: list,ResBlock, no_reslayers,h=1 ,act_fun = "ReLU"):
    if act_fun == "ReLU":
        AF = nn.ReLU()
    if act_fun == "Tanh":
        AF = nn.Tanh()
    if act_fun == "Sigmoid":
        AF = nn.Sigmoid()

    layer_dict = OrderedDict()
    layer_dict[f'linear0'] = nn.Linear(in_features=dims[0], out_features=dims[1], bias=False)
    #layer_dict[f'actfun0'] = AF
    for k in range(1,no_reslayers+1):
        layer_dict[f'res{k}'] = ResBlock(dims[1],dims[1],h )#nn.Linear(in_features=dims[1], out_features=dims[1])
        # layer_dict[f'actfun{k}'] = AF
    k_end = no_reslayers+1
    layer_dict[f'linear{k_end}'] = nn.Linear(in_features=dims[1], out_features=dims[2], bias=False)
    #layer_dict[f'actfun{k_end}'] = AF
    return nn.Sequential(layer_dict)

model_res = make_resnet([28*28,100,10], ResBlock1, no_reslayers=5,h=0.25)
#print(model_res)

def gen_hierarch_models(no_levels,coarse_no_reslayers,dims,ResBlock, act_fun="ReLU"):
    no_reslayers = coarse_no_reslayers # we start to build the coarsest net first
    h = 1
    for k in range(no_levels):
        model = make_resnet(dims,ResBlock,no_reslayers,h=h,act_fun=act_fun)
        #save model
        PATH = "nets/model_atlevel_"+str(k)+".pt"
        torch.save(model.state_dict(),PATH)
        no_reslayers = 2*no_reslayers-1
        h = 0.5*h

    return print("Models of all levels are saved in directory 'nets'!")



gen_hierarch_models(3,2,[28*28,100,10],ResBlock1)