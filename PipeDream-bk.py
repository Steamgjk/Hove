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
# In-place aggregation
#os.environ["CUDA_VISIBLE_DEVICES"]='1'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=8, type=int, help='worker number')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--subitern', default=4, type=int, help='sub batch size')
parser.add_argument('--itern', default=400, type=int, help='sub batch size')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--partition', default=[0,4,4,7,7,14,14,23,23,33,33,40,40,53,53,-1], nargs='+', type=int)
args = parser.parse_args()


WK_BASE = 0
TS_BASE = args.wn
WC_BASE = TS_BASE + args.wn
TC_BASE = WC_BASE + args.wn
SY_BASE = TC_BASE + args.wn
WORLD_SIZE = 5 * args.wn

criterion = nn.CrossEntropyLoss()
fake_input = torch.randn([args.subbs,3,224,224], dtype=torch.float)
fake_target = torch.from_numpy(np.random.randint(0,999,size=int(args.subbs)))
fake_input = fake_input.share_memory_()
fake_target = fake_target.share_memory_()


def ini_data_storage(wid):
    global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE
    f= open("./vgg_info.dump", "rb")
    profile_list = pickle.load(f)
    f.close()
    sta_lidx = args.partition[wid+wid]
    end_lidx = args.partition[wid+wid+1]
    SUB_MODEL = myVGG("VGG19", sta_lidx = sta_lidx, end_lidx = end_lidx)
    SUB_MODEL.to("cuda")
    SUB_MODEL.share_memory()
    SUB_OPTIMIZER = optim.SGD(SUB_MODEL.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    INPUT_SIZE = []
    for item in profile_list[sta_lidx]["shape"]:
    	INPUT_SIZE.append(item)
    OUTPUT_SIZE = []
    for item in profile_list[end_lidx]["shape"]:
    	OUTPUT_SIZE.append(item)
    INPUT_SIZE[0] *= args.subbs
    OUTPUT_SIZE[0] *= args.subbs
    INPUT_PLACEHOLDER = torch.zeros(INPUT_SIZE)
    INPUT_PLACEHOLDER.requires_grad = True
    OUTPUT_PLACEHOLDER = torch.zeros(OUTPUT_SIZE)
    INPUT_PLACEHOLDER = INPUT_PLACEHOLDER.cuda()
    OUTPUT_PLACEHOLDER = OUTPUT_PLACEHOLDER.cuda()

def is_head(wid):
	if wid == 0:
		return True
	else:
		return False
def is_tail(wid):
	if wid == args.wn -1:
		return True
	else:
		return False
def get_succ(wid):
	return (wid+1)
def get_pre(wid):
	return (wid-1)

def train(wid):
	global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE
	seq_list = []
	req_list = []
	SUB_OPTIMIZER.zero_grad()
	if is_head(wid):
		for i in range(args.subitern):
			INPUT_PLACEHOLDER.copy_(fake_input)
			OUTPUT_PLACEHOLDER = SUB_MODEL(INPUT_PLACEHOLDER)
			back_ctx = torch.zeros(OUTPUT_SIZE)
			succ = get_succ(wid)
			dist.send(tensor = OUTPUT_PLACEHOLDER.cpu(), dst = succ)
		for i in range(args.subitern):
			dist.recv(tensor = back_ctx, src = succ)
			OUTPUT_PLACEHOLDER.backward(back_ctx.cuda())
	elif is_tail(wid):
		pre = get_pre(wid)
		for i in range(args.subitern):
			input_data = torch.zeros(INPUT_SIZE)
			dist.recv(tensor=input_data, src = pre)
			INPUT_PLACEHOLDER.copy_(input_data)
			OUTPUT_PLACEHOLDER = SUB_MODEL(INPUT_PLACEHOLDER)
			loss = criterion(OUTPUT_PLACEHOLDER, fake_target.cuda())
			loss.backward()
			back_ctx = HookFunc.backward_ctx
			dist.send(tensor = back_ctx.cpu(), dst = pre)
			print("sub_iter=",int(i))
	else:
		pre = get_pre(wid)
		succ = get_succ(wid)
		for i in range(args.subitern):
			input_data = torch.zeros(INPUT_SIZE)
			dist.recv(tensor=input_data, src = pre)
			INPUT_PLACEHOLDER.copy_(input_data)
			OUTPUT_PLACEHOLDER = SUB_MODEL(INPUT_PLACEHOLDER)
			dist.send(tensor = OUTPUT_PLACEHOLDER.cpu(), dst = succ)
		for i in range(args.subitern):
			in_back_ctx =  torch.zeros(OUTPUT_SIZE)
			dist.recv(tensor = in_back_ctx, src = succ)
			OUTPUT_PLACEHOLDER.backward(in_back_ctx.cuda())
			out_back_ctx = HookFunc.backward_ctx
			out_back_ctx = out_back_ctx.cpu()
			dist.send(tensor=out_back_ctx, dst = pre)


def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("GLOO: ", os.environ['MASTER_ADDR'], " ",  os.environ['MASTER_PORT'])
    dist.init_process_group(backend, rank=rank, world_size=size)

if __name__ == '__main__':
	global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE
	init_processes(args.wid, args.wn)
	ini_data_storage(args.wid)
	SUB_OPTIMIZER.zero_grad()
	time_list = []
	for i in range(args.itern):
		train(args.wid)
		'''
		for j in range(args.subitern):
			train(args.wid)
			print("sub_iter=",int(j))
		'''
		SUB_OPTIMIZER.step()
		time_list.append(time.time())
		iter_num = len(time_list)-1
		if iter_num>0:
			print("iter_num=",iter_num,"\t",float(time_list[-1]-time_list[0]*1.0)/iter_num)



    	

