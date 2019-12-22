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
parser.add_argument('--fcwn', default=4, type=int, help='worker number')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--partition', default=[0,26,1, 26,53,1, 53,-1,3, 26,53,2,0,26,2], nargs='+', type=int)
args = parser.parse_args()

global TOKEN_SLICES,WORKER_DATA_INFO,TOKEN_WORKER_INFO,TOKEN_CNTERS,CACHE_WEIGHT,FLAG_BITS,REQUEST_TENSORS,SYNC_FLAG
SEND_TENSOT_SIZE = 30
RECV_TENSOR_SIZE = 2
REQUEST_TENSOR_SIZE = 4
TOKEN_LAYERS = 5
TOKEN_CAPACITY = 32
WORKER_CAPCITY = 32
REQUEST_TENSOR_NUM = TOKEN_CAPACITY*TOKEN_LAYERS
TOKEN_WEIGHT = [1,8,32,8,1]
INT2BIT = torch.ones(TOKEN_CAPACITY)
for i in range(TOKEN_CAPACITY):
	INT2BIT[i] = (INT2BIT[i]<<i)

TOKEN_SLICES = torch.zeros([TOKEN_LAYERS, TOKEN_CAPACITY], dtype = torch.int32)
WORKER_DATA_INFO = torch.zeros([TOKEN_LAYERS, WORKER_CAPCITY], dtype = torch.int32) 
TOKEN_WORKER_INFO = torch.ones([TOKEN_LAYERS, TOKEN_CAPACITY], dtype = torch.int32).mul_(-1) 
TOKEN_CNTERS = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
CACHE_WEIGHT = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
FLAG_BITS  =  torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
REQUEST_TENSORS = torch.zeros([WORKER_CAPCITY, TOKEN_LAYERS, TOKEN_CAPACITY, REQUEST_TENSOR_SIZE], dtype=torch.int32)
SYNC_FLAG = torch.zeros(1, dtype=torch.int32)
TOKEN_SLICES = TOKEN_SLICES.share_memory_()
WORKER_DATA_INFO = WORKER_DATA_INFO.share_memory_()
TOKEN_WORKER_INFO = TOKEN_WORKER_INFO.share_memory_()
TOKEN_CNTERS = TOKEN_CNTERS.share_memory_()
CACHE_WEIGHT = CACHE_WEIGHT.share_memory_()
FLAG_BITS = FLAG_BITS.share_memory_()
REQUEST_TENSORS = REQUEST_TENSORS.share_memory_()
SYNC_FLAG =	SYNC_FLAG.share_memory_()
TOKEN_LOCATION = torch.zeros([TOKEN_LAYERS, TOKEN_CAPACITY], dtype = torch.int32) 
TOKEN_LOCATION = TOKEN_LOCATION.share_memory_()
#####
CACHE_LOCKS = [mp.Lock() for i in range(TOKEN_LAYERS)]
TOKEN_LOCKS = [mp.Lock() for i in range(TOKEN_LAYERS)]

tet_list = [1,2,3,4]

def reset_env():
	CACHE_WEIGHT.zero_()
	TOKEN_SLICES.zero_()
	SYNC_FLAG.zero_()
	TOKEN_LOCATION.zero_()
	TOKEN_WORKER_INFO.zero_().add_(-1)
	#enque initialization
	TOKEN_LOCKS[0].acquire()
	TOKEN_CNTERS.zero_()
	#enqueu layer-0 tokens
	TOKEN_CNTERS[0] = TOKEN_CAPACITY
	TOKEN_LOCKS[0].release()
	print("reset env:", TOKEN_CNTERS)

def fetch_token(wid):
	print("fetch_token:", TOKEN_CNTERS)


def train_sync_proc(wid):
	if wid == 0:
		
		time.sleep(5)
		print(tet_list)
		#fetch_token(wid)
		#func1()
		'''
		print("lock_acqure")
		TOKEN_LOCKS[0].acquire()
		print("after sleep")
		print("SYNC ", SYNC_FLAG, " ", no_share_variable)
		'''

	else:
		tet_list[0] = 222
		print("ddd:",tet_list)

		#func2()
		'''
		TOKEN_LOCKS[0].acquire()
		print("wid = 1 acquired lock")
		SYNC_FLAG += 1
		no_share_variable += 1
		print("SYNC1 ", SYNC_FLAG, " ", no_share_variable)
		time.sleep(10)
		print("wid=1 now I will free the lock for you")
		TOKEN_LOCKS[0].release()
		'''


if __name__ == '__main__':
	#mp.set_start_method("spawn", force=True)
	ts_list = []
	reset_env()

	for wid in range(2):
	    ts_p = mp.Process(target=train_sync_proc, kwargs={"wid":wid})
	    ts_p.start()
	    ts_list.append(ts_p)
	for wid in range(2):
	    ts_list[wid].join()