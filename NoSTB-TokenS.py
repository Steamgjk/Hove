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
#parser.add_argument('--replica', default="1", type=int, help='Master Port')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--tokencap', default="32", type=int, help='token capacity')
parser.add_argument('--weight', default=[1,4,4,4,1], nargs='+', type=int)
args = parser.parse_args()


TOKEN_LAYERS = 5
TOKEN_CAPACITY = args.tokencap 
BASE_TOKEN_NUMBER = int(args.wn* args.subbs/TOKEN_CAPACITY)
WORKER_CAPCITY = args.wn

OCCUPY_MAP = torch.zeros([TOKEN_LAYERS,BASE_TOKEN_NUMBER], dtype=torch.int32)
HOLD_MAP = torch.zeros([TOKEN_LAYERS,BASE_TOKEN_NUMBER], dtype=torch.int32)
COMPLETE_MAP = torch.zeros([TOKEN_LAYERS,BASE_TOKEN_NUMBER], dtype=torch.int32)

TOKEN_WEIGHT = args.weight
TOKEN_NUMBER = [ int(BASE_TOKEN_NUMBER/val) for val in TOKEN_WEIGHT]

HOLD_MAP = HOLD_MAP.share_memory_()
OCCUPY_MAP= OCCUPY_MAP.share_memory_()
COMPLETE_MAP = COMPLETE_MAP.share_memory_()
COMPLETE_MAP.zero_()
HOLD_MAP_LOCK = [mp.Lock() for i in range(TOKEN_LAYERS)]


QUEUE_LEN = 1000
#depth, token_no
TENSOR_QUEUES = torch.zeros([args.wn, QUEUE_LEN, 2], dtype=torch.int32)
TENSOR_QUEUES = TENSOR_QUEUES.share_memory_()
#CHUNK_HOLD_MAP = torch.zeros([TOKEN_LAYERS,TOKEN_CAPACITY], dtype=torch.int32)
CHUNK_HOLD_MAP = torch.zeros([TOKEN_LAYERS,BASE_TOKEN_NUMBER], dtype=torch.int32)
CHUNK_HOLD_MAP = CHUNK_HOLD_MAP.share_memory_()
# head and tail
QUEUE_PTRS = torch.zeros([args.wn, 2], dtype=torch.int32)
QUEUE_PTRS = QUEUE_PTRS.share_memory_()
QUEUE_LOCKS = [mp.Lock() for i in range(args.wn)]

NEED_SYNC = torch.ones([args.wn, TOKEN_LAYERS], dtype=torch.int32)
NEED_SYNC = NEED_SYNC.share_memory_()

SYNC_CNTERS = torch.zeros(WORKER_CAPCITY+1, dtype=torch.int32)
SYNC_CNTERS = SYNC_CNTERS.share_memory_()

READY_RST = torch.ones(WORKER_CAPCITY, dtype=torch.int32)
READY_RST = READY_RST.share_memory_()


ESTABLISHED_CONN = torch.zeros(1, dtype=torch.int32)
ESTABLISHED_CONN = ESTABLISHED_CONN.share_memory_()
connection_lock = mp.Lock()

CONNECTION_ESTABLISHED = torch.zeros(WORKER_CAPCITY, dtype=torch.int32)
CONNECTION_ESTABLISHED = CONNECTION_ESTABLISHED.share_memory_()

GLOBAL_STEP = torch.zeros(1, dtype=torch.int32)
GLOBAL_STEP = GLOBAL_STEP.share_memory_()
#depth|chunk_no|sender/receiver

WK_BASE = 0
TS_BASE = args.wn
WCR_BASE = TS_BASE + args.wn
WCS_BASE = WCR_BASE + args.wn
TC_BASE = WCS_BASE + args.wn
SY_BASE = TC_BASE + args.wn
TSY_BASE = SY_BASE + args.wn
WORLD_SIZE = 7 * args.wn

W2TS_MSG_SIZE = 1+0
TS2W_MSG_SIZE = 1+2
TS2C_MSG_SIZE = 1+3
TS2S_MSG_SIZE = 1+1
S2TS_MSG_SIZE = 1+1
NEW_REQUEST = 0
REPORT_PROGRESS = 1
SYNC_FIN = 4
DISTRIBUTE_TOKEN = 2
SYNC_CMD = 3
NO_AVAILABLE = 5
OTHER_TOKENS = 6
CHUNK_REQUEST = 7
CHUNK_RESPONSE = 8
SYNC_REQUEST = 9
SYNC_RESPONSE = 10
CONN_ESTABLISH = 11
CONNECTION_RST = 12
CONNECTION_REQUEST = 13
RST_FIN = 14

TS2C_MSG_QUEUES = torch.zeros([args.wn, QUEUE_LEN, TS2C_MSG_SIZE], dtype= torch.int32)
TS2C_MSG_QUEUES = TS2C_MSG_QUEUES.share_memory_()
TS2C_MSG_PTRS = torch.zeros([args.wn, 2], dtype= torch.int32)
TS2C_MSG_PTRS = TS2C_MSG_PTRS.share_memory_()
TS2C_MSG_QUEUE_LOCKS = [mp.Lock() for i in range(args.wn)]

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
	QUEUE_PTRS.zero_()
	for j in range(TOKEN_LAYERS):
		UNIT_TOKEN_NO = int(TOKEN_NUMBER[j]/args.wn)
		if UNIT_TOKEN_NO > 0:
			for w in range(args.wn):
				token_base_offset = w * UNIT_TOKEN_NO
				for i  in range(UNIT_TOKEN_NO):
					tail_ptr = QUEUE_PTRS[w][1]
					TENSOR_QUEUES[w][tail_ptr][0] = j    #depth
					TENSOR_QUEUES[w][tail_ptr][1] =token_base_offset+i   #token_no
					QUEUE_PTRS[w][1] += 1
		else:
			UNIT_WK_NUM = int(args.wn/TOKEN_NUMBER[j])
			for i in range(TOKEN_NUMBER[j]):
				w_no = i*UNIT_WK_NUM
				tail_ptr = QUEUE_PTRS[w_no][1]
				TENSOR_QUEUES[w_no][tail_ptr][0] = j    #depth
				TENSOR_QUEUES[w_no][tail_ptr][1] = i   #token_no
				QUEUE_PTRS[w_no][1] += 1

	HOLD_MAP.zero_().add_(-1)
	OCCUPY_MAP.zero_().add_(-1)
	CHUNK_HOLD_MAP.zero_().add_(-2)
	for i in range(args.wn):
		QUEUE_PTRS[i][0] = 0
	COMPLETE_MAP.zero_()
	NEED_SYNC.zero_().add_(1)
	GLOBAL_STEP.zero_()
	READY_RST.zero_()
	print("QUEUE ptrs ", QUEUE_PTRS)
	print("TOKEN_NUMBER:",TOKEN_NUMBER)
	#exit(0)

def reset():
	
	connection_lock.acquire()
	while READY_RST.sum() != int(ESTABLISHED_CONN):
		#print(READY_RST)
		#print("ests=",int(ESTABLISHED_CONN))
		continue

	HOLD_MAP.zero_().add_(-1)
	OCCUPY_MAP.zero_().add_(-1)
	CHUNK_HOLD_MAP.zero_().add_(-2)
	for i in range(args.wn):
		QUEUE_PTRS[i][0] = 0
	COMPLETE_MAP.zero_()
	NEED_SYNC.zero_().add_(1)
	GLOBAL_STEP.add_(1)
	READY_RST.zero_()
	ESTABLISHED_CONN.zero_()
	connection_lock.release()
	
def get_token(wid):
	depth = None
	token_no = None
	dependency_list = None
	
	QUEUE_LOCKS[0].acquire()

	for i in range(TOKEN_LAYERS):
		for j in range(TOKEN_NUMBER[i]):
			if OCCUPY_MAP[i][j] < 0:
				dependency_list= check_dependency(wid, j, j)
				if dependency_list is not None:
					depth = i 
					token_no = j 
					OCCUPY_MAP[depth][token_no]=wid
					break 
		if depth is not None:
			break
	QUEUE_LOCKS[0].release()
	#print("get_token\t",wid,"\t depth=", (depth),"\ttoken_no=", (token_no))
	return depth, token_no,dependency_list

def get_token1(wid):
	depth = None
	token_no = None
	dependency_list = None
	
	QUEUE_LOCKS[0].acquire()
	for i in range(TOKEN_LAYERS-1,-1,-1):
		for j in range(TOKEN_NUMBER[i]):
			min_dependency = 99999
			dependency_list = check_dependency(wid, i, j)
			#print("three-1  ", i,"\t", j, "\t",dependency_list )
			if HOLD_MAP[i][j] <0 and OCCUPY_MAP[i][j]<0 and dependency_list is not None:
				if min_dependency > len(dependency_list):
					min_dependency = len(dependency_list)
					depth= i 
					token_no = j
				break
		if depth is not None:
			break
	if depth is not None:
		OCCUPY_MAP[depth][token_no]=wid
	QUEUE_LOCKS[0].release()
	#print("get_token\t",wid,"\t depth=", (depth),"\ttoken_no=", (token_no))
	return depth, token_no,dependency_list

def update_token_state(wid, depth, token_no):
	sta = token_no * TOKEN_WEIGHT[depth]
	CHUNK_HOLD_MAP[depth][sta:(sta+TOKEN_WEIGHT[depth])] = wid 
	HOLD_MAP[depth][token_no] = wid
	COMPLETE_MAP[depth][token_no] = 1


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


def is_bp_depth(depth):
	if depth == 3 or depth == 4:
		return True
	else:
		return False
def check_dependency(request_wid, request_depth, request_token_no):
	dependency_list = []
	if request_depth == 0:
		return dependency_list
	else:
		pre_depth = request_depth -1
		sta = request_token_no * TOKEN_WEIGHT[request_depth]
		for chunk_id in range(sta, sta+TOKEN_WEIGHT[request_depth]):
			#print("chunk_id=",int(chunk_id), "\t pre_depth=",int(pre_depth))
			if request_wid == CHUNK_HOLD_MAP[pre_depth][chunk_id]:
				continue
			else:
				dependency_item = [0 for row in range(3)]
				dependency_item[0] = int(CHUNK_HOLD_MAP[pre_depth][chunk_id])
				if dependency_item[0]<0:
					#print("Intended ", "request_depth=",int(request_depth),"\t request_token_no=",int(request_token_no), "pre_depth=", int(pre_depth), " chunk_id=", chunk_id)
					return None

				dependency_item[1] = pre_depth 
				dependency_item[2]=  chunk_id
				dependency_list.append(dependency_item)
				
		if is_bp_depth(request_depth):
			fp_depth = TOKEN_LAYERS-1 - request_depth
			sta = request_token_no * TOKEN_WEIGHT[request_depth]
			for chunk_id in range(sta, sta+TOKEN_WEIGHT[request_depth]):
				if request_wid == CHUNK_HOLD_MAP[fp_depth][chunk_id]:
					continue
				else:
					dependency_item = [0 for row in range(3)]
					dependency_item[0] = int(CHUNK_HOLD_MAP[fp_depth][chunk_id])
					dependency_item[1] = fp_depth 
					dependency_item[2]=  chunk_id
					dependency_list.append(dependency_item)	

	return dependency_list



def fill_cmd(my_wid, dependency_list):
	#print("my_wid=",int(my_wid),"\t len=",len(dependency_list))
	TS2C_MSG_QUEUE_LOCKS[my_wid].acquire()
	tail = TS2C_MSG_PTRS[my_wid][1]
	idx = 0
	for dependency_item in dependency_list:
		if tail >= QUEUE_LEN:
			idx = int(tail % QUEUE_LEN)
		else:
			idx = int(tail)
		TS2C_MSG_QUEUES[my_wid][idx][0] = CHUNK_REQUEST
		TS2C_MSG_QUEUES[my_wid][idx][1] = dependency_item[0] #work id
		TS2C_MSG_QUEUES[my_wid][idx][2] = dependency_item[1] # depth
		TS2C_MSG_QUEUES[my_wid][idx][3] = dependency_item[2] # chunk_id
		
		#print("CHUNK_REQUEST:",TS2C_MSG_QUEUES[my_wid][tail])
		tail += 1
	#print(dependency_list)
	TS2C_MSG_QUEUE_LOCKS[my_wid].release()
	for dependency_item in dependency_list:
		response_wid =  dependency_item[0]
		#print("response_wid=",int(response_wid))
		TS2C_MSG_QUEUE_LOCKS[response_wid].acquire()
		tail = TS2C_MSG_PTRS[response_wid][1]
		idx = 0
		if tail >= QUEUE_LEN:
			idx = int(tail%QUEUE_LEN)
		else:
			idx = int(tail)

		TS2C_MSG_QUEUES[response_wid][idx][0] = CHUNK_RESPONSE
		TS2C_MSG_QUEUES[response_wid][idx][1] = my_wid #work id
		TS2C_MSG_QUEUES[response_wid][idx][2] = dependency_item[1] # depth
		TS2C_MSG_QUEUES[response_wid][idx][3] = dependency_item[2] # chunk_id
		#print("CHUNK_RESPONSE:",TS2C_MSG_QUEUES[response_wid][tail])
		tail += 1
		TS2C_MSG_QUEUE_LOCKS[response_wid].release()
		#print("add dependency_list ", len(dependency_list))


def get_sync_layer(wid):
	if (NEED_SYNC[wid][2]==1) and  (COMPLETE_MAP[2].sum() == TOKEN_NUMBER[2]):
		return 2
	if (NEED_SYNC[wid][3]==1) and  (COMPLETE_MAP[3].sum() == TOKEN_NUMBER[3]):
		return 3
	if (NEED_SYNC[wid][4]==1) and  (COMPLETE_MAP[4].sum() == TOKEN_NUMBER[4]):
		return 4
	return None
	

def ts_process(channel_id):
	#reset()
	#print(HOLD_MAP)
	my_rank = channel_id + TS_BASE
	print("Starting init_processes rank=",my_rank)
	init_processes(my_rank, WORLD_SIZE, 'gloo')
	print("Fin init_processes rank=",my_rank)
	worker_rank = channel_id + WK_BASE
	worker2ts_tensor = torch.zeros(W2TS_MSG_SIZE, dtype = torch.int32)
	ts2worker_tensor = torch.zeros(TS2W_MSG_SIZE, dtype = torch.int32)
	worker2ts_tensor[0] = NEW_REQUEST
	requester_wid = channel_id
	local_cnt = 0
	#print(OCCUPY_MAP)
	#exit(0)
	while True:
		#print(int(channel_id)," Recvinb")
		dist.recv(tensor = worker2ts_tensor, src = worker_rank)
		#print(int(channel_id)," Recved ", worker2ts_tensor)
		if worker2ts_tensor[0] == CONNECTION_REQUEST:
			connection_lock.acquire()
			CONNECTION_ESTABLISHED[channel_id] =1
			ESTABLISHED_CONN.add_(1)
			ts2worker_tensor[0] = CONN_ESTABLISH
			ts2worker_tensor[1] = int(GLOBAL_STEP)
			connection_lock.release()
			dist.send(ts2worker_tensor, dst = worker_rank)
		#print(int(channel_id)," Recved ", worker2ts_tensor)
		elif worker2ts_tensor[0] == NEW_REQUEST:	
			#print("requester_wid=",requester_wid,"\t",QUEUE_PTRS[requester_wid][0], "\t", QUEUE_PTRS[requester_wid][1])	
			to_sync_layer = get_sync_layer(channel_id)
			if to_sync_layer is not None :
				ts2worker_tensor[0] = SYNC_CMD
				ts2worker_tensor[1] = to_sync_layer
				dist.send(ts2worker_tensor, dst = worker_rank)
				#sync response
				dist.recv(worker2ts_tensor, src = worker_rank)
				print(int(channel_id),"\t","FIN to_sync_layer=",int(to_sync_layer) )
				NEED_SYNC[channel_id][to_sync_layer] = 0
				if  to_sync_layer == TOKEN_LAYERS-1:
					dist.recv(worker2ts_tensor, src = worker_rank)
					ts2worker_tensor[0] = CONNECTION_RST
					dist.send(ts2worker_tensor, dst = worker_rank)
					SYNC_CNTERS[channel_id] += 1
					#this iteration has finned
					READY_RST[channel_id] =1
					while READY_RST[channel_id] == 1:
						continue 
				continue
			depth, token_no,dependency_list = get_token(channel_id)
			#print(int(channel_id),"\t New Request  depth=",(depth),"\t token_no=",(token_no))
			if depth is None:
				ts2worker_tensor[0] = NO_AVAILABLE
				dist.send(tensor = ts2worker_tensor, dst = worker_rank)	
			else:
				#print(int(channel_id),"\t New Request  depth=",(depth),"\t token_no=",(token_no))
				ts2worker_tensor[0] = DISTRIBUTE_TOKEN
				ts2worker_tensor[1] = depth
				ts2worker_tensor[2] = token_no
				if len(dependency_list) == 0:
					dist.send(tensor=ts2worker_tensor, dst = worker_rank)
				else:
					ts2worker_tensor[0] = OTHER_TOKENS
					fill_cmd(channel_id,dependency_list)
					dist.send(tensor=ts2worker_tensor, dst = worker_rank)

				#wait for report progress
				dist.recv(tensor = worker2ts_tensor, src = worker_rank)
				update_token_state(channel_id, depth, token_no)




def rq_process(channel_id):
	#head MSG_PTR[channel_id][0]
	# tail MSG_PTR[channel_id][1]
	my_rank = channel_id + TC_BASE
	wcr_rank = channel_id + WCR_BASE
	wcs_rank = channel_id + WCS_BASE
	print("starting rq_process rank=",my_rank)
	init_processes(my_rank, WORLD_SIZE, 'gloo')
	print("started rq_process rank=",my_rank)
	front = TS2C_MSG_PTRS[channel_id][0]
	tail = TS2C_MSG_PTRS[channel_id][1]
	while True:
		if front < tail:
			idx = int(front% QUEUE_LEN)
			rc2wc_tensor = TS2C_MSG_QUEUES[channel_id][idx]
			#print("sending...", int(channel_id),"\t",rc2wc_tensor)
			if rc2wc_tensor[0] == CHUNK_REQUEST:
				dist.send(tensor = rc2wc_tensor, dst = wcr_rank)
			elif rc2wc_tensor[0] == CHUNK_RESPONSE:
				dist.send(tensor = rc2wc_tensor, dst = wcs_rank)
			#print("sended...", int(channel_id),"\t",rc2wc_tensor)
			front += 1

def ms_process(channel_id):
	my_rank = channel_id + TSY_BASE
	ms_rank = channel_id + SY_BASE
	print("starting ms_process rank=",my_rank)
	init_processes(my_rank, WORLD_SIZE, 'gloo')
	ms2ts_tensor = torch.zeros(S2TS_MSG_SIZE, dtype=torch.int32)
	while True:
		continue
		dist.recv(tensor = ms2ts_tensor, src = ms_rank)
		SYNC_CNTERS[channel_id] += 1
		#print(int(channel_id),"\tfin syncing layer ", int(SYNC_CNTERS[0:args.wn].sum()), "\t", int(SYNC_CNTERS[-1]*args.wn))



if __name__ == '__main__':
	init()
	
	print("init....")
	ts_proc_list = []
	rq_proc_list = []
	ms_proc_list = []
	for channel_id in range(args.wn):
		ts_p = mp.Process(target=ts_process, kwargs={"channel_id":channel_id})
		ts_p.start()
		ts_proc_list.append(ts_p)
	for channel_id in range(args.wn):
		rq_p = mp.Process(target = rq_process, kwargs={"channel_id":channel_id})
		rq_p.start()
		rq_proc_list.append(rq_p)
	for channel_id in range(args.wn):
		ms_p = mp.Process(target = ms_process, kwargs={"channel_id":channel_id})
		ms_p.start()
		ms_proc_list.append(ms_p)


	SYNC_CNTERS[-1]+=1
	time_list = []
	while True:
		if int(SYNC_CNTERS[0:args.wn].sum()) == int(SYNC_CNTERS[-1]*args.wn):
			time_list.append(time.time())
			iter_num = len(time_list)-1
			print("iter_num=",iter_num)
			if iter_num > 0:
				print("Iter : ", int(iter_num),"\t", float(time_list[-1]*1.0 - time_list[0])/iter_num)
			reset()
			SYNC_CNTERS[-1]+=1
			

	for ts_p in ts_proc_list:
		ts_p.join()
	for rs_p in rq_proc_list:
		rs_p.join()
	for ms_p in ms_proc_list:
		ms_p.join()