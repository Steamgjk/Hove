# -*- coding: utf-8 -*
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


def train():
    #Launch recv td
    print("worker_id(rank)",worker_id, "  size:",str(worker_num)," batch_size=",batch_size )
    input("Worker End Connection Initialized") 
    
    sub_net.train()
    inputs = None
    outputs = None
    train_loss = 0
    correct = 0
    total = 0
    iteration_num = 10000
    iter_n = 0
    loss = None
    sub_optimizer.zero_grad()
    sta = time.time()
    
    while True:
        inputs = fake_input.to(device)
        targets = fake_target.to(device)
        outputs = sub_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        sub_optimizer.step()
        sub_optimizer.zero_grad()
        iter_n = iter_n + 1
        if iter_n%10 == 0:
            ed = time.time()
            print("iter_n=",iter_n," time=",(ed-sta*1.0))
        if iter_n == iteration_num:
            exit(0)        



train()

