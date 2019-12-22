# -*- coding: utf-8 -*
'''Train CIFAR10 with PyTorch.'''
## TODO: 1. Interaction between C and Python  2. Dynamic add/delete net layer  3. Aggregate Gradient
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
import numba.cuda as cuda
import random
# In-place aggregation

#test_net = VGG("VGG19")
#summary(test_net, (3, 32, 32))

os.environ["CUDA_VISIBLE_DEVICES"]='1'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--pn', default=1, type=int, help='worker id')
parser.add_argument('--subitern', default=1, type=int, help='sub itern')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--sleepn', default=1, type=int, help='sleep time')
parser.add_argument('--prb', default=10, type=float, help='probability')
args = parser.parse_args()

if args.wid == 0:
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]='1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


worker_num = args.wn
worker_id = args.wid # 1|2|3|
#fake_input = torch.randn(128,3,224,224)
#fake_target = torch.randint(0,1,(128,1000))

sub_batch_size = args.subbs
sub_iter_num = args.subitern
fake_input = torch.randn(sub_batch_size,3,32,32)
fake_target = torch.from_numpy(np.random.randint(0,999,size=sub_batch_size))
print(fake_input.size())
print(fake_target.size())

#VGG19 54
criterion = nn.CrossEntropyLoss()

sub_net = GoogLeNet(sta_lidx = -1, ed_lidx = -1)
sub_net.to(device)
sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    dist.init_process_group(backend, rank=rank, world_size=size)

def train():
    #Launch recv td
    print("worker_id(rank)",worker_id, "  size:",str(worker_num)," batch_size=",int(sub_batch_size*sub_iter_num) )
    init_processes(worker_id,worker_num, 'gloo')

    print("Worker End Connection Initialized") 
    global sub_net,sub_optimizer,device
    is_cpu_mode = False
    sub_net.train()
    inputs = None
    outputs = None
    train_loss = 0
    correct = 0
    total = 0
    iteration_num = 100
    iter_n = 0
    loss = None
    sub_optimizer.zero_grad()
    sta = 0
    ed = 0
    time_list=[]
    #with torch.autograd.profiler.emit_nvtx():
        #cuda.profile_start()
    while True:
        if random.randint(0,99) < args.prb:
            print("I need sleep {:d} s".format(args.sleepn))
            if args.sleepn > 0:
                time.sleep(args.sleepn)
        for si in range(sub_iter_num):
            inputs = fake_input.to(device)
            targets = fake_target.to(device)
            outputs = sub_net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        for name, parameters in sub_net.named_parameters():
            if(parameters.grad is not None):
                '''
                if name.find("fc_layers")>= 0:
                    continue
                if name.find("classifier") >= 0:
                    continue
                '''
                #print("name=",name,"\tparam sz=",parameters.grad.size())
                grad_content = parameters.grad.to("cpu")
                dist.all_reduce(tensor=grad_content, op = dist.ReduceOp.SUM)
                grad_content = grad_content/worker_num
                parameters.grad = grad_content.to(device)

        sub_optimizer.step()
        sub_optimizer.zero_grad()
        iter_n = iter_n + 1

        time_list.append(time.time())
        iter_num = len(time_list)-1
        if iter_num > 0:
            print("Iter : ", int(iter_num),"\t", float(time_list[-1]*1.0 - time_list[0])/iter_num)


train()

