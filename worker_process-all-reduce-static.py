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
parser.add_argument('--bs', default=1, type=int, help='batch size')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

ps_num = args.pn
worker_num = args.wn
worker_id = args.wid # 1|2|3|
#fake_input = torch.randn(128,3,224,224)
#fake_target = torch.randint(0,1,(128,1000))
batch_size = args.bs
fake_input = torch.randn(batch_size,3,224,224)
fake_target = torch.from_numpy(np.random.randint(0,999,size=batch_size))
print(fake_input.size())
print(fake_target.size())

#VGG19 54
criterion = nn.CrossEntropyLoss()

sub_net = VGG('VGG19', sta_lidx = -1, end_lidx = -1)
sub_net.to(device)
sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '12.12.10.13'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

def train():
    #Launch recv td
    print("worker_id(rank)",worker_id, "  size:",str(worker_num)," batch_size=",batch_size )
    init_processes(worker_id,worker_num, 'gloo')

    input("Worker End Connection Initialized") 
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
    sta = time.time()
    #with torch.autograd.profiler.emit_nvtx():
    while iter_n<= 100:
        inputs = fake_input.to(device)
        targets = fake_target.to(device)
        outputs = sub_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        comm_time_sta = time.time()
        para_num = 0
        for name, parameters in sub_net.named_parameters():
            if(parameters.grad is not None):
                grad_content = parameters.grad.to("cpu")
                para_num += grad_content.numel()
                dist.all_reduce(tensor=grad_content, op = dist.ReduceOp.SUM)
                grad_content = grad_content/worker_num
                parameters.grad = grad_content.to(device)
        comm_time_ed = time.time()
        sub_optimizer.step()
        sub_optimizer.zero_grad()
        #print("iter=",iter_n," comm_time=",str(comm_time_ed-comm_time_ed))
        iter_n = iter_n + 1
        
        if iter_n%10 == 0:
            ed = time.time()
            print("iter_n=",iter_n," time=",(ed-sta*1.0), "comm_num=",para_num)
        if iter_n == iteration_num:
            exit(0)        



train()

