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
import numba.cuda as cuda
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--succ', default=-1, type=int, help='successor worker')
parser.add_argument('--pred', default=-1, type=int, help='predecessor worker')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--conv_wid', default=-1, type=int, help='conv worker id')
parser.add_argument('--conv_wn', default=3, type=int, help='conv worker number')
parser.add_argument('--fc_wid', default=-1, type=int, help='fc worker id')
parser.add_argument('--fc_wn', default=1, type=int, help='fc worker number')
parser.add_argument('--pd', default=1, type=int, help='parallel degree')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--nproc', default=1, type=int, help='number of procs')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--t_iter', default="30", type=int, help='terminal iteration')
parser.add_argument('--partition', default=[0,26, 0, 26, 26, 53, 53,-1], nargs='+', type=int)
args = parser.parse_args()

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    dist.init_process_group(backend, rank=rank, world_size=size)


if __name__ == '__main__':
    wid = args.wid
    init_processes(wid, 2, backend='gloo')
    print("INIT ")
    send_ele = torch.ones(10, dtype=torch.int32)
    recv_ele = torch.zeros(10, dtype=torch.int32)
    if wid == 0:
    	send_ele.mul_(2)
    	print(wid, " ", send_ele)
    	send_req = dist.isend(tensor= send_ele, dst = 1)
    	recv_req = dist.irecv(tensor= recv_ele, src = 1)
    	'''
    	send_req.wait()
    	print("send req fin ", send_req.is_completed())
    	print("recv_req ", recv_req.is_completed())
    	input("Stop...")
    	'''
    	recv_req.wait()
    	print(wid, " ", recv_ele)
    	'''
    	send_req = dist.isend(tensor=tensor_ele, dst = 1)
    	tensor_ele.mul_(4)
    	send_req2 = dist.isend(tensor=tensor_ele, dst = 1)
    	send_req.wait()
    	send_req2.wait()
    	'''
    else:
    	send_ele.mul_(2)
    	print(wid, " ", send_ele)
    	send_req = dist.isend(tensor= send_ele, dst = 0)
    	recv_req = dist.irecv(tensor= recv_ele, src = 0)
    	'''
    	send_req.wait()
    	print("send_req fin ",send_req.is_completed())
    	print("recv_req ", recv_req.is_completed())
    	while True:
    		print("loop recv_req ", recv_req.is_completed())
    		time.sleep(1)
    	'''
    	recv_req.wait()
    	print(wid, " ", recv_ele)
    	'''
    	print(wid, " ", tensor_ele)
    	recv_req = dist.irecv(tensor=tensor_ele, src = 0)
    	arecv_req = dist.irecv(tensor=atensor_ele, src = 0)
    	arecv_req.wait()
    	print("a tensor ", atensor_ele)
    	recv_req.wait()
    	print("ensor ", tensor_ele)
    	print("after ", tensor_ele)
    	'''
    