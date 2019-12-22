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
flayer = 0
blayer = 0
def print_forward_shape(module, input, output):
    global flayer
    
    print("forward input: ",flayer, " ", type(module))
    for val in input:
        if val is None:
            print("None")
        else:
            print(val.size())
    
    '''
    print("forward output:", flayer, " ", type(module), " ", type(output), " ", output.data.size())
    for val in output:
        if val is None:
            print("None")
        else:
            print(val.size())
    '''
    flayer += 1

def print_backward_shape(module, input, output):
    global blayer
    '''
    print("backward input: ", blayer, " ", type(module))
    for val in input:
        if val is None:
            print("None")
        else:
            print(val.size())
    '''
    print("backward output:", blayer," ", type(module), " ", type(output))

    for val in output:
        if val is None:
            print("None")
        else:
            print(val.size())
    blayer += 1

fake_input = torch.randn(2,3,224,224)
fake_target = torch.from_numpy(np.random.randint(0,999,size=int(2)))
criterion = nn.CrossEntropyLoss()
print(fake_target.size())
def hook_profile():
    test_net = VGG("VGG19")
    #test_net.register_forward_hook(print_forward_shape)
    #test_net.register_backward_hook(print_backward_shape)
    
    for layer in test_net.features:
        layer.register_forward_hook(print_forward_shape)
        layer.register_backward_hook(print_backward_shape)
    for layer in test_net.fc_layers:
        layer.register_forward_hook(print_forward_shape)
        layer.register_backward_hook(print_backward_shape)
    target = fake_target
    outputs = test_net(fake_input)
    loss = criterion(outputs, target)
    loss.backward()
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
def profile_alexnet_shape():
    profile_list = []
    test_net = myAlexNet("AlexNet")
    out = fake_input
    cnt = 0
    print("features  ", len(test_net.feature_arr))
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
        print(cnt,"\t", shp, "\t", layer_type )
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
        print(cnt,"\t", shp, "\t", layer_type )
        cnt += 1
    return profile_list

def profile_resnet_shape():
    profile_list = []
    test_net = ResNet152()
    fake_input = torch.randn(1,3,32,32)
    out = fake_input
    cnt = 0
    #out = F.relu(test_net.bn1(test_net.conv1(x)))
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
        print(cnt,"\t", shp, "\t", layer_type )
        cnt +=  1

    return profile_list


def profile_googlenet_shape():
    profile_list = []
    test_net = GoogLeNet()
    fake_input = torch.randn(1,3,32,32)
    out = fake_input
    cnt = 0
    #out = F.relu(test_net.bn1(test_net.conv1(x)))
    for layer in test_net.features:
        dict_item = {}
        if isinstance(layer, GoogleNetHookLayer):
            print("HookLayer jump off")
            continue
        shp = out.size()
        layer_type = type(layer)
        dict_item["idx"]= cnt
        dict_item["type"] = layer_type
        dict_item["shape"] =  list(shp)
        profile_list.append(dict_item)
        out = layer(out) 
        print(cnt,"\t", shp, "\t", layer_type )
        cnt +=  1

    return profile_list


def test_cuda_mem():
    cuda_tensor = torch.zeros([10000,100000])
    another = torch.ones([10000,100000])
    back_ocu = torch.ones([10000,100000])
    back_ocu = back_ocu.cuda()
    cuda_tensor = cuda_tensor.cuda()
    tes = torch.ones(1)*3
    tes = tes.cuda()
    is_cuda = True
    results = []
    while True:
        '''
        res = cuda_tensor * tes
        print(cuda_tensor.device, " ", tes.device)
        print(res.device)
        res = res.cpu()
        results.append(res)
        time.sleep(1)
        print("len ", len(results))
        
        '''
        if is_cuda:
            cuda_tensor = cuda_tensor.cuda()
            another = another.cuda()
            time.sleep(1)
            is_cuda = (not is_cuda)
            print("is_cuda")
        else:
            cuda_tensor =cuda_tensor.cuda()
            another = another.cpu()
            time.sleep(1)
            is_cuda = (not is_cuda)
            print("not cuda")
        print("switch")
        
    time.sleep(100)
if __name__ == '__main__':
    #test_cuda_mem()
    #hook_profile()
    #test()
    profile_list = profile_googlenet_shape()
    with open("./googlenet_info.dump", "wb") as f:
        pickle.dump(profile_list, f)
    print("dump finish")
    cnt = 0
    with open("./googlenet_info.dump", "rb") as f:
        profile_list = pickle.load(f)
        for layer_info in profile_list:
            print(cnt, "\t", (layer_info["shape"]), "\t", layer_info["type"] )
            cnt +=1

    '''
    profile_list = profile_resnet_shape()
    with open("./resnet_info.dump", "wb") as f:
        pickle.dump(profile_list, f)
    print("dump finish")
    cnt = 0
    with open("./resnet_info.dump", "rb") as f:
        profile_list = pickle.load(f)
        for layer_info in profile_list:
            print(cnt, "\t", (layer_info["shape"]), "\t", layer_info["type"] )
            cnt +=1
    '''
    '''
    profile_list = profile_alexnet_shape()
    with open("./alexnet_info.dump", "wb") as f:
        pickle.dump(profile_list, f)
    print("dump finish")
    '''
    '''
    cnt = 0
    with open("./alexnet_info.dump", "rb") as f:
        profile_list = pickle.load(f)
        for layer_info in profile_list:
            print(cnt, "\t", (layer_info["shape"]), "\t", layer_info["type"] )
            cnt +=1
    '''
    '''
    profile_list = profile_vgg_shape()
    with open("./vgg_info.dump", "wb") as f:
        pickle.dump(profile_list, f)
    print("dump finish")
    
    with open("./vgg_info.dump", "rb") as f:
        profile_list = pickle.load(f)
        for layer_info in profile_list:
            print(type(layer_info["shape"]))
    '''
