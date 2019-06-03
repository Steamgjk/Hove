# -*- coding: utf-8 -*
'''Train CIFAR10 with PyTorch.'''
## TODO: 1. Interaction between C and Python  2. Dynamic add/delete net layer  3. Aggregate Gradient
#ps -aux|grep python|cut -c 9-15|xargs kill -9
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
import torch.distributed as dist

# In-place aggregation

#test_net = VGG("VGG19")
#summary(test_net, (3, 32, 32))

os.environ["CUDA_VISIBLE_DEVICES"]=''
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--pid', default=0, type=int, help='worker id')
parser.add_argument('--pn', default=1, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
args = parser.parse_args()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device =  'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
aggregate_cnt_dict={}
aggregate_data_dict = {}
worker_num = args.wn
ps_id = args.pid # 1|2|3|
ps_num = args.pn

ps_id = 0

batch_size = 4
fake_input = torch.zeros(batch_size,3,224,224)
fake_target = torch.from_numpy(np.random.randint(0,999,size=batch_size))

pickle_dump_span = 0
pickle_dump_len = 0
pickle_load_span = 0
pickle_load_len = 0
aggregation_span = 0
aggregation_size = 0

global cache_out

def profile_tensor_shape():
    profie_shape = []
    test_net = VGG("VGG19")
    fake_out = test_net(fake_input)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(fake_out, fake_target)
    loss.backward()
    for name, parameters in test_net.named_parameters():
        if(parameters.grad is not None):
            profie_shape.append(parameters.grad.size())

    return profie_shape

def get_aggregated_para(para_list,div_partition):
    #print("para_list_len=",len(para_list))
    para_sum = para_list[0]
    for idx in range(len(para_list)):
        para_sum.add_(para_list[idx])
    para_sum.div_(div_partition)
    return para_sum
def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '12.12.10.18'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def aggregate_func():  
    print("ps_id(rank)=",ps_id," size=",(ps_num+worker_num))  
    init_processes(ps_id,ps_num+worker_num) 

    print("worker_num=",worker_num)
    shape_list = profile_tensor_shape()

    input("Connection Initialized worker") 
    t_sta = time.time()
    while True:
        for shape_item in shape_list:
            #print("shape_item ",shape_item)
            cache_out = torch.zeros(shape_item)
            dist.reduce(tensor=cache_out, dst=ps_id,op=dist.ReduceOp.SUM)
            reduced_tensor = cache_out.clone()
            reduced_tensor = reduced_tensor/worker_num
            dist.broadcast(tensor = reduced_tensor, src = ps_id)



aggregate_func()

