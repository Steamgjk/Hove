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
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--replica', default="1", type=int, help='Master Port')
args = parser.parse_args()


TOKEN_LAYERS = 5
TOKEN_CAPACITY = 32 * args.replica
WORKER_CAPCITY = 8
HOLD_MAP = torch.zeros([TOKEN_LAYERS,TOKEN_CAPACITY], dtype=torch.int32)
COMPLETE_MAP = torch.zeros([TOKEN_LAYERS,TOKEN_CAPACITY], dtype=torch.int32)
TOKEN_WEIGHT = [1,4,1,4,1]
TOKEN_NUMBER = [ int(TOKEN_CAPACITY/val) for val in TOKEN_WEIGHT]
HOLD_MAP = HOLD_MAP.share_memory_()
COMPLETE_MAP = COMPLETE_MAP.share_memory_()
COMPLETE_MAP.zero_()
TOKEN_COMPLETE_CNTERS = torch.zeros(TOKEN_LAYERS, dtype=torch.int32)
TOKEN_COMPLETE_CNTERS = TOKEN_COMPLETE_CNTERS.share_memory_()
HOLD_CNTERS = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
HOLD_CNTERS = HOLD_CNTERS.share_memory_()
HOLD_MAP_LOCK = [mp.Lock() for i in range(TOKEN_LAYERS)]
COMPLETE_MAP_LOCK = [mp.Lock() for i in range(TOKEN_LAYERS)]
WORKER_LOCK = [mp.Lock() for i in range(args.wn)]
QUEUE_LEN = 1000
#depth, token_no
TENSOR_QUEUES = torch.zeros([args.wn, QUEUE_LEN, 2], dtype=torch.int32)
TENSOR_QUEUES = TENSOR_QUEUES.share_memory_()
CHUNK_HOLD_MAP = torch.zeros([TOKEN_LAYERS,TOKEN_CAPACITY], dtype=torch.int32)
CHUNK_HOLD_MAP = CHUNK_HOLD_MAP.share_memory_()
# head and tail
QUEUE_PTRS = torch.zeros([args.wn, 2], dtype=torch.int32)
QUEUE_PTRS = QUEUE_PTRS.share_memory_()

SYNC_CNTERS = torch.zeros(WORKER_CAPCITY+1, dtype=torch.int32)
SYNC_CNTERS = SYNC_CNTERS.share_memory_()


WK_BASE = 0
TS_BASE = args.wn
WC_BASE = TS_BASE + args.wn
TC_BASE = WC_BASE + args.wn
SY_BASE = TC_BASE + args.wn
WORLD_SIZE = 5 * args.wn

W2TS_MSG_SIZE = 1+2
TS2W_MSG_SIZE = 1+5
NEW_REQUEST = 0
REPORT_PROGRESS = 1
SYNC_FIN = 4
DISTRIBUTE_TOKEN = 2
SYNC_CMD = 3
NO_AVAILABLE = 5

def is_fc_depth(depth):
	if depth == 2:
		return True
	else:
		return False
def is_fc_worker(wid):
	if wid == args.wn-1:
		return True
	else:
		return False

def init():
	#depth first
	replica_num = args.replica
	QUEUE_PTRS.zero_()
	fc_worker = args.wn - 1
	'''
	for j in range(TOKEN_LAYERS):
		for i in range(replica_num):
			replica_height = TOKEN_NUMBER[j]/replica_num
			height = int(replica_height/args.wn)
			for w in range(args.wn):
				base_offset = w*height +i * replica_height
				for k in range(height):
					tail_ptr = QUEUE_PTRS[w][1]
					TENSOR_QUEUES[w][tail_ptr][0] = j
					TENSOR_QUEUES[w][tail_ptr][1] = k + base_offset
					QUEUE_PTRS[w][1] += 1
					if is_fc_depth(j) and (not is_fc_worker(w)):
						tail_ptr = QUEUE_PTRS[fc_worker][1]
						TENSOR_QUEUES[fc_worker][tail_ptr][0] = j
						TENSOR_QUEUES[fc_worker][tail_ptr][1] = k + base_offset
						QUEUE_PTRS[fc_worker][1] += 1
	'''
	for i in range(replica_num):
		for j in range(TOKEN_LAYERS):
			replica_height = TOKEN_NUMBER[j]/replica_num
			height = int(replica_height/args.wn)
			for w in range(args.wn):
				base_offset = w*height +i * replica_height
				for k in range(height):
					tail_ptr = QUEUE_PTRS[w][1]
					TENSOR_QUEUES[w][tail_ptr][0] = j
					TENSOR_QUEUES[w][tail_ptr][1] = k + base_offset
					QUEUE_PTRS[w][1] += 1
					if is_fc_depth(j) and (not is_fc_worker(w)):
						tail_ptr = QUEUE_PTRS[fc_worker][1]
						TENSOR_QUEUES[fc_worker][tail_ptr][0] = j
						TENSOR_QUEUES[fc_worker][tail_ptr][1] = k + base_offset
						QUEUE_PTRS[fc_worker][1] += 1
	



def reset():
	HOLD_MAP.zero_().add(-1)
	for i in range(args.wn):
		QUEUE_PTRS[i][0] = 0
	COMPLETE_MAP.zero_()
	CHUNK_HOLD_MAP.zero_().add(-1)
	HOLD_MAP[2].add(8) #FC worker

def is_available(requester_wid, depth, token_no):
	if (HOLD_MAP[depth][token_no] > -1) and (not (HOLD_MAP[depth][token_no] == requester_wid)):
		return False
	elif depth == 0 and HOLD_MAP[depth][token_no] == -1:
		return True
	else:
		pre_depth = depth-1
		if TOKEN_NUMBER[pre_depth] > TOKEN_NUMBER[depth]:
			#many to one
			unit_size = int(TOKEN_NUMBER[pre_depth]/TOKEN_NUMBER[depth])
			sta_no = unit_size *token_no
			ed_no = sta_no + unit_size
			for pre_token_no in range(sta_no, ed_no):
				if (COMPLETE_MAP[pre_depth][pre_token_no] == 0) and (not (HOLD_MAP[depth][token_no] == requester_wid)):
					return False
			return True
		else:
			#one to many
			unit_size = int(TOKEN_NUMBER[depth]/TOKEN_NUMBER[pre_depth])
			pre_token_no = int(token_no/unit_size)
			if COMPLETE_MAP[pre_depth][pre_token_no] == 0 and (not (HOLD_MAP[depth][token_no] == requester_wid)):
				return False
			else:
				return True

def update_token_state(wid, depth, token_no):
	COMPLETE_MAP[depth][token_no] = 1
	sta = token_no * TOKEN_WEIGHT[depth]
	CHUNK_HOLD_MAP[depth][sta:(sta+TOKEN_WEIGHT[depth])].zero_().add_(wid)

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("GLOO: ", os.environ['MASTER_ADDR'], " ",  os.environ['MASTER_PORT'])
    dist.init_process_group(backend, rank=rank, world_size=size)
    train_sync_ranks = [WK_BASE]*(args.wn)
    train_sync_ranks_fc = [] #TODO
    for i in range(args.wn):
    	train_sync_ranks[i] += i
    train_sync_group = dist.new_group(ranks=train_sync_ranks, backend=backend)
    train_sync_fc_group = None
    return train_sync_group, train_sync_fc_group


def ts_process(channel_id):
	my_rank = channel_id + TS_BASE
	print("Starting init_processes rank=",my_rank)
	init_processes(my_rank, WORLD_SIZE, 'gloo')
	print("Fin init_processes rank=",my_rank)
	worker_rank = channel_id + WK_BASE
	worker2ts_tensor = torch.zeros(W2TS_MSG_SIZE, dtype = torch.int32)
	ts2worker_tensor = torch.zeros(TS2W_MSG_SIZE, dtype = torch.int32)
	worker2ts_tensor[0] = NEW_REQUEST
	requester_wid = channel_id
	iter_n = 0
	local_itern = 0
	cnt = 0
	time_list = []
	while True:
		dist.recv(tensor = worker2ts_tensor, src = worker_rank)
		if worker2ts_tensor[0] == NEW_REQUEST:			
			if QUEUE_PTRS[requester_wid][0]<QUEUE_PTRS[requester_wid][1]:
				front = int(QUEUE_PTRS[requester_wid][0])
				depth = TENSOR_QUEUES[requester_wid][front][0]
				token_no = TENSOR_QUEUES[requester_wid][front][1]
				HOLD_MAP[depth][token_no] = requester_wid
				cnt += 1
				QUEUE_PTRS[requester_wid][0] += 1
				ts2worker_tensor[0] = DISTRIBUTE_TOKEN
				ts2worker_tensor[1] = depth
				ts2worker_tensor[2] = token_no
				dist.send(tensor=ts2worker_tensor, dst = worker_rank)
			else:
				# no available tokens now
				if COMPLETE_MAP[TOKEN_LAYERS-1].sum()== TOKEN_CAPACITY:
					#all have been sent, now need to sync
					ts2worker_tensor[0] = SYNC_CMD
					#after it return, will swtich to  SYNC_FIN
				else:
					ts2worker_tensor[0] = NO_AVAILABLE
				for j in range(TOKEN_LAYERS):
					ts2worker_tensor[j+1] = COMPLETE_MAP[j].sum()
				dist.send(tensor=ts2worker_tensor, dst = worker_rank)
		elif worker2ts_tensor[0] == REPORT_PROGRESS:
			depth = worker2ts_tensor[1]
			token_no = worker2ts_tensor[2]
			#print(int(channel_id),"depth=",int(depth), " token_no=",int(token_no))
			update_token_state(requester_wid,depth,token_no)
		elif worker2ts_tensor[0] == SYNC_FIN:
			SYNC_CNTERS[channel_id] += 1
			while SYNC_CNTERS[channel_id] == SYNC_CNTERS[-1]:
				continue
			'''
			iter_n += 1
			time_list.append(time.time())
			print("iter_n = ", iter_n, "\t", time_list[-1]-time_list[0])
			'''



def rq_process(channel_id):
	#head MSG_PTR[channel_id][0]
	# tail MSG_PTR[channel_id][1]
	my_rank = channel_id + TC_BASE
	cq_rank = channel_id + WC_BASE
	print("starting rq_process rank=",my_rank)
	init_processes(my_rank, WORLD_SIZE, 'gloo')
	print("started rq_process rank=",my_rank)
	while True:
		time.sleep(1)
	



if __name__ == '__main__':
	init()
	'''
	len = int(QUEUE_PTRS[0][1])
	print("len = ", len)
	for i in range(len):
		print(int(TENSOR_QUEUES[0][i][0]),"\t", int(TENSOR_QUEUES[0][i][1]))
	len = int(QUEUE_PTRS[0][1])
	len = int(QUEUE_PTRS[7][1])
	print("len = ", len)
	for i in range(len):
		print(int(TENSOR_QUEUES[7][i][0]),"\t", int(TENSOR_QUEUES[7][i][1]))
	exit(0)
	'''
	ts_proc_list = []
	rq_proc_list = []
	for channel_id in range(args.wn):
		ts_p = mp.Process(target=ts_process, kwargs={"channel_id":channel_id})
		ts_p.start()
		ts_proc_list.append(ts_p)
	for channel_id in range(args.wn):
		rq_p = mp.Process(target = rq_process, kwargs={"channel_id":channel_id})
		rq_p.start()
		rq_proc_list.append(rq_p)

	SYNC_CNTERS[-1]+=1
	time_list = []
	while True:
		if int(SYNC_CNTERS[0:args.wn].sum()) == int(SYNC_CNTERS[-1]*args.wn):
			time_list.append(time.time())
			iter_num = len(time_list)-1
			if iter_num > 0:
				print("Iter : ", int(iter_num),"\t", float(time_list[-1]*1.0 - time_list[0])/iter_num)
			reset()
			SYNC_CNTERS[-1]+=1
			


	for ts_p in ts_proc_list:
		ts_p.join()
	for rs_p in rq_proc_list:
		rs_p.join()