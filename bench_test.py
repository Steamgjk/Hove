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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--bs', default=4, type=int, help='batch size')
parser.add_argument('--layern', default=1, type=int, help='layer_num')
parser.add_argument('--in_channeln', default=128, type=int, help='in channel num')
parser.add_argument('--out_channeln', default=128, type=int, help='out channel num')
parser.add_argument('--itern', default=100, type=int, help='iter num')
parser.add_argument('--picsz', default=[128,224,224], nargs='+', type=int)
parser.add_argument('--nproc', default=1, type=int, help='process num')
args = parser.parse_args()
mp.set_start_method("spawn", force=True)

class BenchTest(nn.Module):
    def __init__(self, layer_num=1, in_channels=128, out_channels=128):
        super(BenchTest, self).__init__()
        self.layers = []
        #self.layers += [nn.Conv2d(3, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(inplace=True)]

        for i in range(layer_num):
            self.layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True)]
        
        self.features = nn.Sequential(*(self.layers))

    def forward(self, sta_input):
        out = sta_input
        for layer in self.features:
            out = layer(out)
        out = out.view(out.size(0), -1)
        return out
def train_proc(rank, bs, btest, sub_optimizer, iter_num, inputs, targets, criterion):
    for i in range(iter_num):
        outputs = btest(inputs)
        loss = criterion(outputs, targets)
        #fp_ed = time.time()
        loss.backward()
        sub_optimizer.step()
        sub_optimizer.zero_grad()

if __name__ == '__main__':
    bs = args.bs
    layer_num = args.layern
    in_channel_num = args.in_channeln
    out_channel_num = args.out_channeln
    iter_num = args.itern
    num_proc = args.nproc
    picsz = args.picsz
    
    btest = BenchTest(layer_num=layer_num,in_channels=in_channel_num, out_channels=out_channel_num).cuda()
    btest.share_memory()
    sub_optimizer = optim.SGD(btest.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    btest.train()
    input_sz = [bs] + picsz
    inputs = torch.randn(input_sz)
    inputs = inputs.cuda()
    targets = torch.from_numpy(np.random.randint(0,999,size=bs))
    targets = targets.cuda()
    criterion = nn.CrossEntropyLoss()
    #multiprocess
    '''
    proc_list = []
    sta = time.time()
    for rank in range(num_proc):
        train_p = mp.Process(target=train_proc, kwargs={"rank":rank, "bs":bs, "btest":btest, "sub_optimizer": sub_optimizer, "iter_num": iter_num, "inputs":inputs, "targets":targets, "criterion":criterion})
        train_p.start()
        proc_list.append(train_p)

    for proc in proc_list:
        proc.join()

    ed = time.time()
    print("layer_num=",layer_num, " iter_time=", (ed*1.0-sta)/100 )
    '''
     
    sta = time.time()
    fp_span = 0
    bp_span = 0
    update_span = 0
    fb_span = 0
    for i in range(iter_num):
        sta = time.time()
        outputs = btest(inputs)
        ed = time.time()
        fb_span += ed -sta
        loss = criterion(outputs, targets)
        #fp_ed = time.time()
        sta = time.time()
        loss.backward()
        ed = time.time()
        fb_span += ed - sta
        sub_optimizer.step()
        sub_optimizer.zero_grad()
    ed = time.time()
    print("layer_num=",layer_num, " iter_time=", (1.0*fb_span)/100 )

