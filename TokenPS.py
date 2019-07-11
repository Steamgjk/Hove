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

mp.set_start_method("spawn", force=True)
# In-place aggregation
#os.environ["CUDA_VISIBLE_DEVICES"]='1'

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

if args.wid == 0:
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]='1'

SIZE = args.wn * 2
FLEX_DEGREE=[1,2,4,2,1]
FC_LIST=[2]
qu_lock = mp.Lock()
qu_list = [Queue.Queue(8), Queue.Queue(8),Queue.Queue(8),Queue.Queue(8),Queue.Queue(8) ]
cache_lock = mp.Lock()
cache_list = [[Queue.Queue(1)], Queue.Queue(2),Queue.Queue(4),Queue.Queue(2),Queue.Queue(1)]
QU_LIST_LEN = len(qu_list)

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    dist.init_process_group(backend, rank=rank, world_size=size)
    #fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    #wn = 4
    allreduce_ranks = [16,17]
    fp_gather_ranks = [0,4,9]
    bp_scatter_ranks = [3,7,10]
    allreduce_group = dist.new_group(ranks=allreduce_ranks, backend=backend)
    fp_gather_group = dist.new_group(ranks=fp_gather_ranks, backend=backend)
    bp_scatter_group = dist.new_group(ranks=bp_scatter_ranks, backend=backend)
    return allreduce_group,fp_gather_group,bp_scatter_group
def isFC(workload2add):
	global FC_LIST
	if workload2add in FC_LIST:
		return True
	else:
		return False
def add_and_fetch_workload(workload_no, wid):
	global qu_lock, qu_list,cache_lock, cache_list, QU_LIST_LEN,FLEX_DEGREE
	#tricky depth-first
	# 1+1+8
	cache_lock.acquire()
	cache_list[workload_no].put(wid)
	tensor_ele = None
	if cache_list[workload_no].full():
		tensor_ele = torch.zeros(10, dtype = torch.int32)
		tensor_ele[0]= workload_no
		tensor_ele[1] = FLEX_DEGREE[workload_no]
		cnt  = 2
		while cache_list[workload_no].empty() == False:
			tensor_ele[cnt] = cache_list[workload_no].get()
			cnt += 1
	cache_lock.release()

	if tensor_ele is not None:
		qu_lock.acquire()
		# add
		qu_list[workload_no].put(tensor_ele)
		# fetch
		workload_tensor = None
		for i in range(QU_LIST_LEN-1, -1, -1):
			if qu_list[i].empty():
				continue
			else:
				workload_tensor = qu_list[i].get()
				break 		
		qu_lock.release()
		return workload_tensor
	else:
		return None




'''
recv_tensor  
workload_no(int32)

send_tensor
model_no(int32)|parallelism_degree(int32)|wid_0,wid_1,...,wid_n (num=parallelism_degree)
'''
def ps_process(channel_id):
	init_processes(channel_id, SIZE, 'gloo')
	recv_tensor = torch.zeros(1, type=torch.int32)
	my_wid = channel_id + args.wn
	dumb_tensor = torch.ones(10, dtype = torch.int32) * (-1)
	while True:
		dist.recv(tensor = recv_tensor, src = my_wid)
		#fetch another model component and send to the worker
		send_tensor = add_and_fetch_workload(recv_tensor[0], channel_id)
		if send_tensor is not None:
			dist.send(tensor = send_tensor, dst = my_wid)
		else:
			dist.send(tensor = dumb_tensor, dst = my_wid)