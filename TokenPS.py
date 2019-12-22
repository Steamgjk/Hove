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
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
args = parser.parse_args()

'''
if args.wid == 0:
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
'''
'''
TS: wn<-> W: wn; TS: wn-> W: wn
TS；wn master_channel
Worker: 2 channel: worker_channel and coordination_channel
WC->MC  MC->WC MC->CC' CC'->WC
'''


TOKEN_LAYERS = 5
TOKEN_CAPACITY = 32
WORKER_CAPCITY = 32
RECV_TENSOR_SIZE = 3
REQUEST_TENSOR_SIZE = 3
TS2W_HEADER=4
SEND_TENSOT_SIZE = TOKEN_CAPACITY * 2 + TS2W_HEADER
TS_BASE = 0
RQ_BASE = args.wn * 3
TQ_BASE = args.wn 
CQ_BASE = args.wn * 2
TOKEN_WEIGHT = [1,8,32,8,1]
WORLD_SIZE = args.wn * 4
MAX_DEGREE = 8


#### Shared memory
'''
TOKEN_SLICES[i][j]: bitmap, Which tokens (bitmap) are needed to compute token_j in layer_i 
TOKEN_SUCCS[i][j]: bitmap, which tokens need token_j in workload_i to compute them 
WORKER_DATA_INFO[i][j]: bitmap, As for workload_i, which tokens (bitmap) do the worker_j have undertaken
TOKEN_WORKER_INFO[i][j]: int32 As for token_j in workload_i, which worker has fetched it for training? if the value is -1, then no worker is holding it, then it can be fetched
TOKEN_CNTERS[i]: for workload_i, how many tokens have been generated (and possibly can be fetched)
CACHE_WEIGHT[i]: for workload_i, how many weights have been cached. if it reaches the weigth threshold, it can generate a new token for workload_{i+1}, and then the CACHE_WEIGHT[i] will be zeroed
FLAG_BITS[i]: to be abandoned
REQUEST_TENSORS[i][j][k]: to faciliate asynchronous send to coordiante thread. worker_i is requesting the workload_j for token_k, to another worker
SYNC_FLAG: bitmap, 1 represents that TS has sent sync signal to that worker, when SYNC_FLAG become all 1s, reset the env and start the next iteration
TOKEN_LOCATION[i][j]: for those one-to-many dependencies: for token_j in workerload_i, which portion of data does it depend on in its predecessor?

MSG_QUEUE[i][j][0]: the workload_no requested by  the worker_i 
MSG_QUEUE[i][j][1]: the token_idx requested by  the worker_i 
MSG_PTR[i][0] the front prt requested by worker_i
MSG_PTR[i][1] the tail prt requested by worker_i
'''
TOKEN_PRE = torch.zeros([TOKEN_LAYERS, TOKEN_CAPACITY, MAX_DEGREE], dtype = torch.int32)
TOKEN_SUC = torch.zeros([TOKEN_LAYERS, TOKEN_CAPACITY, MAX_DEGREE], dtype = torch.int32)
TOKEN_PRE_CNTER = torch.zeros([TOKEN_LAYERS, TOKEN_CAPACITY], dtype = torch.int32)
TOKEN_SUC_CNTER = torch.zeros([TOKEN_LAYERS, TOKEN_CAPACITY], dtype = torch.int32)
TOKEN_WORKER_INFO = torch.ones([TOKEN_LAYERS, TOKEN_CAPACITY], dtype = torch.int32).mul_(-1) 
TOKEN_CNTERS = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
CACHE_WEIGHT = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
MSG_QUEUE = torch.zeros([WORKER_CAPCITY, TOKEN_LAYERS*TOKEN_CAPACITY, REQUEST_TENSOR_SIZE], dtype = torch.int32)
MSG_PTR = torch.zeros([WORKER_CAPCITY,2], dtype = torch.int32)
SYNC_CNTER = torch.zeros(1, dtype=torch.int32)
SYNCED_FLAG = torch.zeros(WORKER_CAPCITY, dtype= torch.int32)
LAST_UPLOADED_CNTER = torch.zeros(1, dtype=torch.int32)
UPLOADED_CNTERS = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
READY_WORKLOAD = torch.zeros(1, dtype=torch.int32).add_(-1)
ITER_CNT = torch.zeros(1, dtype=torch.int32)
TOKEN_WORKER_INFO = TOKEN_WORKER_INFO.share_memory_()
TOKEN_CNTERS = TOKEN_CNTERS.share_memory_()
CACHE_WEIGHT = CACHE_WEIGHT.share_memory_()
SYNC_CNTER = SYNC_CNTER.share_memory_()
SYNCED_FLAG = SYNCED_FLAG.share_memory_()
ITER_CNT = ITER_CNT.share_memory_()
LAST_UPLOADED_CNTER = LAST_UPLOADED_CNTER.share_memory_()
UPLOADED_CNTERS = UPLOADED_CNTERS.share_memory_()
READY_WORKLOAD = READY_WORKLOAD.share_memory_()
MSG_QUEUE = MSG_QUEUE.share_memory_()
MSG_PTR = MSG_PTR.share_memory_()
TOKEN_PRE = TOKEN_PRE.share_memory_()
TOKEN_SUC = TOKEN_SUC.share_memory_()
TOKEN_PRE_CNTER = TOKEN_PRE_CNTER.share_memory_()
TOKEN_SUC_CNTER = TOKEN_SUC_CNTER.share_memory_()
FLOW_MAP = torch.zeros([TOKEN_LAYERS,TOKEN_CAPACITY], dtype=torch.int32)
FLOW_MAP = FLOW_MAP.share_memory_()
#####
CACHE_LOCKS = [mp.Lock() for i in range(TOKEN_LAYERS)]
TOKEN_LOCKS = [mp.Lock() for i in range(TOKEN_LAYERS)]
REQUEST_LOCKS = [mp.Lock() for i in range(WORKER_CAPCITY)]



def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("GLOO: ", os.environ['MASTER_ADDR'], " ",  os.environ['MASTER_PORT'])
    dist.init_process_group(backend, rank=rank, world_size=size)
    train_sync_ranks = [args.wn]*(args.wn)
    train_sync_ranks_fc = [] #TODO
    for i in range(args.wn):
    	train_sync_ranks[i] += i
    train_sync_group = dist.new_group(ranks=train_sync_ranks, backend=backend)
    train_sync_fc_group = None
    return train_sync_group, train_sync_fc_group
	

def reset_env():
	rest_sta = time.time()
	CACHE_WEIGHT.zero_()
	LAST_UPLOADED_CNTER.zero_()
	UPLOADED_CNTERS.zero_()
	TOKEN_WORKER_INFO.zero_().add_(-1)
	#enque initialization
	#TOKEN_LOCKS[0].acquire()
	TOKEN_CNTERS.zero_()
	for i in range(TOKEN_CAPACITY):
		FLOW_MAP[0][i] = i
	MSG_PTR.zero_()

	TOKEN_CNTERS[0] = TOKEN_CAPACITY
	TOKEN_PRE_CNTER.zero_()
	TOKEN_SUC_CNTER.zero_()
	ITER_CNT[0] += 1
	READY_WORKLOAD[0] = int(ITER_CNT * TOKEN_LAYERS)-1
	SYNCED_FLAG.zero_()
	SYNC_CNTER.zero_()
	rest_ed = time.time()
	print("rese11t = ", float(rest_ed-rest_sta))
	#print("reset env:", TOKEN_CNTERS)

def add_token(workload_no, token_id, wid):
	global LAST_UPLOADED_CNTER
	if workload_no == TOKEN_LAYERS-1:
		LAST_UPLOADED_CNTER += 1
		#print("LAST_UPLOADED_CNTER=",LAST_UPLOADED_CNTER)
		return
	
	next_workload_no = workload_no + 1
	CACHE_LOCKS[workload_no].acquire()

	CACHE_WEIGHT[workload_no] += TOKEN_WEIGHT[workload_no]
	#print("added workload_no=",int(workload_no),"\t token_id=",int(token_id),"\t wid=",int(wid)," CACHE_WEIGHT[workload_no]=",int(CACHE_WEIGHT[workload_no]))

	while True:
		token_candidate_id = TOKEN_CNTERS[next_workload_no]
		#print("precnter=", TOKEN_PRE_CNTER[next_workload_no][token_candidate_id])
		pre_cnt = int(TOKEN_PRE_CNTER[next_workload_no][token_candidate_id])
		TOKEN_PRE[next_workload_no][token_candidate_id][pre_cnt] = token_id 
		TOKEN_PRE_CNTER[next_workload_no][token_candidate_id] += 1
		suc_cnt = int(TOKEN_SUC_CNTER[workload_no][token_id])
		TOKEN_SUC[workload_no][token_id][suc_cnt] = token_candidate_id
		TOKEN_SUC_CNTER[workload_no][token_id] += 1
		#if workload_no ==2:
		#	print("workload_no=",int(workload_no),"\ttoken_id=",int(token_id),"\tsucc_cnt=",int(TOKEN_SUC_CNTER[workload_no][token_id]),"\tcachew=",int(CACHE_WEIGHT[workload_no]),"\ttoken_w=",int(TOKEN_WEIGHT[next_workload_no]), "\twid=",int(wid))
		if CACHE_WEIGHT[workload_no] < TOKEN_WEIGHT[next_workload_no]:
			#print("Less: CACHE_WEIGHT[workload_no]=",int(CACHE_WEIGHT[workload_no]),"\t TOKEN_WEIGHT[next_workload_no]=",int(TOKEN_WEIGHT[next_workload_no]))
			#if workload_no == 2:
			#	print("00cachew=",int(CACHE_WEIGHT[workload_no]),"\ttoken_w=",int(TOKEN_WEIGHT[next_workload_no]))
			break
		else:
			CACHE_WEIGHT[workload_no] -= TOKEN_WEIGHT[next_workload_no]
			#if workload_no == 2:
			#	print("cachew=",int(CACHE_WEIGHT[workload_no]),"\ttoken_w=",int(TOKEN_WEIGHT[next_workload_no]))
			TOKEN_CNTERS[next_workload_no] += 1
			#print("TOKEN_CNTERS add one=",TOKEN_CNTERS[next_workload_no])
			if CACHE_WEIGHT[workload_no] ==0:
				break

	if TOKEN_WEIGHT[workload_no] < TOKEN_WEIGHT[next_workload_no]:
		#only one succ
		succ_token_id = TOKEN_SUC[workload_no][token_id][0]
		for j in range(TOKEN_CAPACITY):
			if FLOW_MAP[workload_no][j] == token_id:
				FLOW_MAP[next_workload_no][j] = succ_token_id
	else:
		#many succs
		suc_cnt = int(TOKEN_SUC_CNTER[workload_no][token_id])
		allocate_num = suc_cnt * TOKEN_WEIGHT[next_workload_no]
		my_cnt = 0
		for j in range(TOKEN_CAPACITY):
			if FLOW_MAP[workload_no][j] == token_id:
				succ_location = int(my_cnt/TOKEN_WEIGHT[next_workload_no])
				FLOW_MAP[next_workload_no][j] = TOKEN_SUC[workload_no][token_id][succ_location]
				my_cnt += 1
				if my_cnt == allocate_num:
					break

	
	CACHE_LOCKS[workload_no].release()

	UPLOADED_CNTERS[workload_no] += TOKEN_WEIGHT[workload_no]
	to_complete_wno = int(READY_WORKLOAD+1)% TOKEN_LAYERS
	if UPLOADED_CNTERS[to_complete_wno] == TOKEN_CAPACITY:
		READY_WORKLOAD[0] += 1
	#print("add workload_no=",int(workload_no), " token_id=",int(token_id), " wid=",int(wid), " ready=",int(READY_WORKLOAD[0]))
	
def calc_data_locality(workload_no,token_id,wid):
	pre_cnt = TOKEN_PRE_CNTER[workload_no][token_id]
	local_num = 0
	for i in range(pre_cnt):
		pre_wno = workload_no - 1
		pre_token = TOKEN_PRE[workload_no][token_id][i]
		pre_wid = TOKEN_WORKER_INFO[pre_wno][pre_token]
		if pre_wid == wid:
			local_num += 1
	return local_num

def is_fc_worker(wid):
	if wid == 0:
		return True
	else:
		return False
def is_fc_layer(workload_no):
	if workload_no == 2:
		return True
	else:
		return False
def fetch_token(synced_wokload_cnter,wid):
	global SYNC_CNTER
	#The last layer has been completed, send the sync signal to each worker
	if SYNC_CNTER == args.wn:
		#has not been prepared ready for next iteration
		return -1, -1
	if LAST_UPLOADED_CNTER == TOKEN_CAPACITY:
		if SYNCED_FLAG[wid] == 0:
			SYNC_CNTER += 1
			SYNCED_FLAG[wid] = 1
			return -2, -2  # -2 represents sync
		else:
			return -1,-1
	token_to_fetch = -1
	workload_no_to_fetch = -1
	#TODO: FC should be distinguished from CONV

	available_wno = TOKEN_LAYERS-1
	if READY_WORKLOAD[0] - synced_wokload_cnter == TOKEN_LAYERS:
		available_wno = synced_wokload_cnter % TOKEN_LAYERS
	if is_fc_worker(wid) and available_wno > 2:
		#fetch from FC layer
		TOKEN_LOCKS[2].acquire()
		maxv = -1
		for j in range(TOKEN_CNTERS[2]):
			if TOKEN_WORKER_INFO[2][j] == -1:
				local_num = calc_data_locality(2,j, wid)
				if local_num > maxv:
					workload_no_to_fetch = 2 
					token_to_fetch = j
					maxv = local_num
				break
			else:
				continue
		if token_to_fetch > -1:
			TOKEN_WORKER_INFO[workload_no_to_fetch][token_to_fetch] = wid
		TOKEN_LOCKS[2].release()
		if token_to_fetch > -1:
			return workload_no_to_fetch, token_to_fetch


	for i in range(available_wno, -1, -1):
		#print("i = ", i, " cnter=", TOKEN_CNTERS[i])
		TOKEN_LOCKS[i].acquire()
		maxv = -1
		for j in range(TOKEN_CNTERS[i]):
			if TOKEN_WORKER_INFO[i][j] == -1:
				#can be fetched
				#TODO: considering data locality !!! PRE
				#print("can fetch ", i, " ", j)
				local_num = calc_data_locality(i,j, wid)
				if local_num > maxv:
					workload_no_to_fetch = i 
					token_to_fetch = j
					maxv = local_num
				break
			else:
				continue
		if token_to_fetch > -1:
			TOKEN_WORKER_INFO[workload_no_to_fetch][token_to_fetch] = wid
			#print("Locality:",  float(maxv))

		TOKEN_LOCKS[i].release()	
		if token_to_fetch == -1:
			continue
		else:
			break

	return workload_no_to_fetch,token_to_fetch



def add_and_fetch_workload(workload_no, token_id, synced_wokload_cnter, wid):

	#tricky depth-first
	if workload_no > -1: 
		#The request is valid, update the token state, and then return new workload; otherwise, directly return new workload
		add_token(workload_no, token_id, wid)
	
	#Fetch new workload for the worker wid, depth first
	workload_no_to_fetch,token_to_fetch = fetch_token(synced_wokload_cnter, wid)
	return workload_no_to_fetch, token_to_fetch

def get_fp_token(bp_workload_no, bp_token_no):
	fp_workload_no = TOKEN_LAYERS - 1 - bp_workload_no
	fp_token_no = -1
	for i in range(TOKEN_CAPACITY):
		if(FLOW_MAP[bp_workload_no][i] == bp_token_no):
			fp_token_no = FLOW_MAP[fp_workload_no][i]
			break
	'''
	if fp_token_no == -1:
		print("bp_workload_no=",bp_workload_no,"\t", FLOW_MAP[bp_workload_no], "\tbp_token_no=",int(bp_token_no))
		print("fp_workload_no=",fp_workload_no,"\t", FLOW_MAP[fp_workload_no])
		print("Map:", FLOW_MAP)
	'''
	return fp_workload_no,fp_token_no
def is_bp(workload_no):
	if workload_no == 3 or workload_no == 4:
		return True
	else:
		return False

def ts_process(channel_id):
	my_rank = channel_id + TS_BASE
	print("Starting init_processes rank=",my_rank)
	init_processes(my_rank, WORLD_SIZE, 'gloo')
	print("Fin init_processes rank=",my_rank)
	recv_tensor = torch.zeros(RECV_TENSOR_SIZE, dtype=torch.int32)
	send_tensor = torch.ones(SEND_TENSOT_SIZE, dtype = torch.int32) * (-1)
	work_rank = channel_id + TQ_BASE
	
	while True:
		#sta_time = time.time()
		#print("recving...")
		dist.recv(tensor = recv_tensor, src = work_rank)
		#fetch another model component and send to the worker
		workload_no_to_fetch, token_to_fetch = add_and_fetch_workload(recv_tensor[0], recv_tensor[1], recv_tensor[2], channel_id)

		#print("channel_id=",channel_id, " workload_no_to_fetch=", int(workload_no_to_fetch), " token_to_fetch=",int(token_to_fetch))
		if workload_no_to_fetch == -2:
			#send sync
			send_tensor[0] = -2
			send_tensor[1] = -2

		elif workload_no_to_fetch == -1:
			#send no available
			send_tensor[0] = -1
			send_tensor[1] = -1

		else:
			pre_workload_no = workload_no_to_fetch - 1
			send_tensor[0] = workload_no_to_fetch
			send_tensor[1] = token_to_fetch
			pre_chunk_num = 0
			base_offset = TS2W_HEADER
			if pre_workload_no < 0:
				#basic layer, no need from other workers
				#TODO: adaptive batching
				pass
			else:

				pre_chunk_num = TOKEN_WEIGHT[workload_no_to_fetch]
				pre_num = TOKEN_PRE_CNTER[workload_no_to_fetch][token_to_fetch]

				'''
				print("workload_no_to_fetch-1=", int(workload_no_to_fetch))
				print("token_to_fetch-1=", int(token_to_fetch))
				print("pre_num-1=",int(pre_num))
				print("pre_chunk_num-1=",int(pre_chunk_num))
				'''
				if  pre_num == 1:
					#one to many
					pre_token_id = TOKEN_PRE[workload_no_to_fetch][token_to_fetch][0]
					pre_worker_id = TOKEN_WORKER_INFO[pre_workload_no][pre_token_id]
					succ_num = TOKEN_SUC_CNTER[pre_workload_no][pre_token_id]
					location = 0
					for i in range(succ_num):
						if TOKEN_SUC[pre_workload_no][pre_token_id][i] == token_to_fetch:
							location = i 
							break
					base_chunk_offset = pre_token_id * TOKEN_WEIGHT[pre_workload_no]+location * pre_chunk_num
					'''
					#if workload_no_to_fetch == 3:
					print("pre_workload_no=",int(pre_workload_no))
					print("token_to_fetch=", int(token_to_fetch))
					print("workload_no_to_fetch=", int(workload_no_to_fetch))
					print("pre_token_id=",int(pre_token_id))
					print("pre_worker_id=",int(pre_worker_id))
					print("succ_num=",int(succ_num))
					print("location=",int(location))
					print("base_chunk_offset=",int(base_chunk_offset))
					print("pre_chunk_num=",int(pre_chunk_num))
					'''
					for j in range(pre_chunk_num):
						send_tensor[base_offset] = pre_worker_id
						send_tensor[base_offset+1] = base_chunk_offset + j	
						if not pre_worker_id == channel_id:
							REQUEST_LOCKS[pre_worker_id].acquire()
							tail_id = MSG_PTR[pre_worker_id][1]
							request_tensor = MSG_QUEUE[pre_worker_id][tail_id]
							request_tensor[0] = pre_workload_no
							request_tensor[1] = send_tensor[base_offset+1] #chunk_id
							request_tensor[2] = channel_id 
							MSG_PTR[pre_worker_id][1] += 1
							REQUEST_LOCKS[pre_worker_id].release()
						base_offset += 2
				else:
					#many to one
					#print("pre_num=",int(pre_num))
					for i in range(pre_num):
						pre_token_id = TOKEN_PRE[workload_no_to_fetch][token_to_fetch][i]
						pre_worker_id = TOKEN_WORKER_INFO[pre_workload_no][pre_token_id]
						base_chunk_offset = pre_token_id * TOKEN_WEIGHT[pre_workload_no]
						'''
						print(int(channel_id),"\tpre_token_id=",int(pre_token_id))
						print(int(channel_id),"\tpre_worker_id=",int(pre_worker_id))
						print(int(channel_id),"\tbase_chunk_offset=",int(base_chunk_offset))
						print(int(channel_id),"\tbase_offset=",int(base_offset))
						'''
						for j in range(TOKEN_WEIGHT[pre_workload_no]):
							#print(int(channel_id),"\tbase_offset=",int(base_offset), " j=",j)
							send_tensor[base_offset] = pre_worker_id
							send_tensor[base_offset+1] = base_chunk_offset + j
							if not pre_worker_id == channel_id:
								REQUEST_LOCKS[pre_worker_id].acquire()
								tail_id = MSG_PTR[pre_worker_id][1]
								request_tensor = MSG_QUEUE[pre_worker_id][tail_id]
								request_tensor[0] = pre_workload_no
								request_tensor[1] = send_tensor[base_offset+1] #chunk_no
								request_tensor[2] = channel_id 
								MSG_PTR[pre_worker_id][1] += 1
								REQUEST_LOCKS[pre_worker_id].release()
							base_offset += 2
				#if it is bp, double 
				if is_bp(workload_no_to_fetch):
					#!!!!!Wait, I will create a map for you
					#FLOW_MAP中同一行的是一条流，看他里面填的token_idx, 再根据token_idex去找worker_id
					fp_workload_no, fp_token_id = get_fp_token(workload_no_to_fetch, token_to_fetch)
					fp_wid = TOKEN_WORKER_INFO[fp_workload_no][fp_token_id]
					base_chunk_offset = fp_token_id * TOKEN_WEIGHT[fp_workload_no]
					#print("fp_workload_no=",int(fp_workload_no)," fp_token_id=",int(fp_token_id)," base_offset=",int(base_offset)," fp_wid=",int(fp_wid))
					for j in range(pre_chunk_num):
						send_tensor[base_offset] = fp_wid
						send_tensor[base_offset+1] = base_chunk_offset + j
						if not (fp_wid == channel_id):
							REQUEST_LOCKS[fp_wid].acquire()
							tail_id = MSG_PTR[fp_wid][1]
							request_tensor = MSG_QUEUE[fp_wid][tail_id]
							request_tensor[0] = fp_workload_no
							request_tensor[1] = send_tensor[base_offset+1]#chunk_no
							request_tensor[2] = channel_id 
							MSG_PTR[fp_wid][1] += 1
							REQUEST_LOCKS[fp_wid].release()		
						base_offset +=2
					send_tensor[2] = pre_chunk_num + pre_chunk_num
					send_tensor[-1] = fp_token_id	
					#print("bp fp_token_id=",int(fp_token_id))
					#print("->",int(channel_id),"\tbp tensor=",send_tensor)			
				else:
					send_tensor[2] = pre_chunk_num

				send_tensor[3] = int(READY_WORKLOAD[0])
		dist.send(tensor = send_tensor, dst = work_rank)
		#ed_time = time.time()
		#print(channel_id,":",float(ed_time -sta_time))
		#print("TS2W... ",my_rank,"->",work_rank, " workload_no=",int(send_tensor[0])," token_no=",int(send_tensor[1]))

def rq_process(channel_id):
	#head MSG_PTR[channel_id][0]
	# tail MSG_PTR[channel_id][1]
	my_rank = channel_id + RQ_BASE
	cq_rank = channel_id + CQ_BASE
	print("starting rq_process rank=",my_rank)
	init_processes(my_rank, WORLD_SIZE, 'gloo')
	print("started rq_process rank=",my_rank)
	while True:
		if MSG_PTR[channel_id][0] < MSG_PTR[channel_id][1]:
			idx = MSG_PTR[channel_id][0]
			request_tensor = MSG_QUEUE[channel_id][idx]
			#print("RQ2C... ", int(my_rank),"->",int(cq_rank), " ", request_tensor)
			dist.send(tensor=request_tensor, dst=cq_rank)
			MSG_PTR[channel_id][0] += 1


if __name__ == '__main__':
	reset_env()
	print("Launching...")
	
	#Launch process
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

	print("Start All")

	while True:
		if  SYNC_CNTER == args.wn:
			#each work has entered sync phase, so reset the env for new iterations

			reset_env()
			'''
			while True:
				print("REseting..")
				time.sleep(1)
			'''
			

	for ts_p in ts_proc_list:
		ts_p.join()
	for rq_p in rq_proc_list:
		rq_p.join()

	

