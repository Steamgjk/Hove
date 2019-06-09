# -*- coding: utf-8 -*
'''Train CIFAR10 with PyTorch.'''
## TODO: 1. Interaction between C and Python  2. Dynamic add/delete net layer  3. Aggregate Gradient
#g3 g4 g5 g7
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from models import *
from utils import progress_bar
import torch.nn.init as init
from torchsummary import summary
from ctypes import *
from models import globalvar as gl
from ctypes import *
import pickle
import time
import operator
import queue as Queue
import torch.distributed as dist
import torch.multiprocessing as mp

fake_input = torch.randn(1,3,224,224)
def profile_vgg_shape():
    profile_list = []
    test_net = VGG("VGG19")
    out = fake_input
    cnt = 0
    for layer in test_net.features:
        dict_item = {}
        if isinstance(layer, HookLayer):
            print("HookLayer jump off")
            continue
        shp = out.size()
        layer_type = type(layer)
        dict_item["idx"]= cnt
        dict_item["type"] = layer_type
        dict_item["shape"] =  list(shp)
        profile_list.append(dict_item)
        out = layer(out) 
        #print(cnt,"\t", shp, "\t", layer_type )
        cnt +=  1
    out = out.view(out.size(0), -1)
    for layer in test_net.fc_layers:
        dict_item={}
        shp = out.size()
        layer_type = type(layer)
        dict_item["idx"]= cnt
        dict_item["type"] = layer_type
        dict_item["shape"] = list(shp)
        profile_list.append(dict_item)
        out = layer(out)
        #print(cnt,"\t", shp, "\t", layer_type )
        cnt += 1
    return profile_list

if __name__ == '__main__':
    
    profile_list = profile_vgg_shape()
    with open("./vgg_info.dump", "wb") as f:
        pickle.dump(profile_list, f)
    print("dump finish")
    
    with open("./vgg_info.dump", "rb") as f:
        profile_list = pickle.load(f)
        for layer_info in profile_list:
            print(type(layer_info["shape"]))
