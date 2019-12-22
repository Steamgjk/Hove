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
from multiprocessing import Queue
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
parser.add_argument('--partition', default=[0,2,2,4,4,6,6,8,8,10,10,11,11,12,12,-1], nargs='+', type=int)

args = parser.parse_args()


WK_BASE = 0
TS_BASE = args.wn
WC_BASE = TS_BASE + args.wn
TC_BASE = WC_BASE + args.wn
SY_BASE = TC_BASE + args.wn
WORLD_SIZE = 5 * args.wn

criterion = nn.CrossEntropyLoss()
fake_input = torch.randn([args.subbs,3,32,32], dtype=torch.float)
fake_target = torch.from_numpy(np.random.randint(0,999,size=int(args.subbs)))
fake_input = fake_input.share_memory_()
fake_target = fake_target.share_memory_()

FP_RECV_TENSORS = None
BP_RECV_TENSORS = None

def ini_data_storage(wid):
    global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE, FP_RECV_TENSORS, BP_RECV_TENSORS
    f= open("./googlenet_info.dump", "rb")
    profile_list = pickle.load(f)
    f.close()
    sta_lidx = args.partition[wid+wid]
    end_lidx = args.partition[wid+wid+1]
    SUB_MODEL = GoogLeNet(sta_lidx = sta_lidx, ed_lidx = end_lidx)
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
    fp_mem_sz = [args.subitern]+INPUT_SIZE
    bp_mem_sz = [args.subitern] + OUTPUT_SIZE
    FP_RECV_TENSORS = torch.zeros(fp_mem_sz)
    FP_RECV_TENSORS = FP_RECV_TENSORS.share_memory_()
    BP_RECV_TENSORS = torch.zeros(bp_mem_sz)
    BP_RECV_TENSORS = BP_RECV_TENSORS.share_memory_()
    INPUT_PLACEHOLDER = torch.zeros(INPUT_SIZE)
    INPUT_PLACEHOLDER.requires_grad = True
    OUTPUT_PLACEHOLDER = torch.zeros(OUTPUT_SIZE)
    INPUT_PLACEHOLDER = INPUT_PLACEHOLDER.cuda()
    OUTPUT_PLACEHOLDER = OUTPUT_PLACEHOLDER.cuda()
    INPUT_PLACEHOLDER = INPUT_PLACEHOLDER.share_memory_()
    OUTPUT_PLACEHOLDER = OUTPUT_PLACEHOLDER.share_memory_()

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
#fp_send 0 | fp_recv 1 | bp_send 2| bp_recv 3
def get_fp_succ_rank(wid):
	return (wid+1)* 4 + 1
def get_fp_pre_rank(wid):
	return (wid-1)* 4 + 0
def get_bp_succ_rank(wid):
	return (wid+1)* 4 + 2
def get_bp_pre_rank(wid):
	return (wid-1)* 4 + 3


def fp_send_proc(wid, fp_senq):
	my_rank = wid* 4 + 0
	world_size = args.wn * 4
	init_processes(my_rank, world_size, backend='gloo')
	succ_rank = get_fp_succ_rank(wid)
	while True:
		if wid + 1 >args.wn-1:
			time.sleep(1)
			continue
		#print("fp_send_proc:", my_rank,"->",succ_rank)
		if fp_senq.empty()== False:
			send_tensor = fp_senq.get()
			#print("fp send_tensor sz=",send_tensor.size(), my_rank, "->", succ_rank)
			dist.send(tensor = send_tensor, dst = succ_rank)

def bp_send_proc(wid, bp_senq):
	my_rank = wid * 4 + 2
	world_size = args.wn * 4
	init_processes(my_rank, world_size, backend='gloo')
	pre_rank = get_bp_pre_rank(wid)
	while True:
		if wid - 1 < 0:
			time.sleep(1)
			continue
		
		if bp_senq.empty()==False:
			#print("bp_send_proc:", my_rank,"->",pre_rank)
			send_tensor = bp_senq.get()
			#print("bp send_tensor sz=",send_tensor.size())
			dist.send(tensor = send_tensor, dst = pre_rank)
			#print("bp send success")
def fp_recv_proc(wid, fp_recq):
	my_rank = wid * 4 + 1
	world_size = args.wn * 4
	init_processes(my_rank, world_size, backend='gloo')
	pre_rank = get_fp_pre_rank(wid)
	cnt = 0
	while True:
		if wid - 1 < 0:
			time.sleep(1)
			continue
		#print("fp_recv_proc:", pre_rank,"->",my_rank)
		recv_tensor = FP_RECV_TENSORS[cnt]
		#print("recv from ",pre_rank,"->",my_rank)
		dist.recv(tensor = recv_tensor, src = pre_rank)
		#print("fp recv_tensor sz=",recv_tensor.size())
		fp_recq.put(recv_tensor)
		cnt += 1
		if cnt == args.subitern:
			cnt = 0
def bp_recv_proc(wid, bp_recq):
	my_rank = wid * 4 + 3
	world_size = args.wn * 4
	init_processes(my_rank, world_size, backend='gloo')
	succ_rank = get_bp_succ_rank(wid)
	cnt = 0
	while True:
		if wid+1 > args.wn-1:
			time.sleep(1)
			continue
		#print("bp_recv_proc:", succ_rank,"->",my_rank)
		recv_tensor = BP_RECV_TENSORS[cnt]
		#print("1-bp recv_tensor sz=",recv_tensor.size())
		dist.recv(tensor = recv_tensor, src = succ_rank)
		#print("bp recv_tensor sz=",recv_tensor.size())
		bp_recq.put(recv_tensor)
		cnt += 1
		if cnt == args.subitern:
			cnt = 0
def train(wid, fp_senq, fp_recq, bp_senq, bp_recq):
	global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE
	SUB_OPTIMIZER.zero_grad()
	iter_num = 0
	time_list = []
	output_list = [None for j in range(args.subitern)]
	storage_sz = [args.subitern]+OUTPUT_SIZE
	output_data_storage = torch.zeros(storage_sz)
	while True:
		if is_head(wid):
			for j in range(args.subitern):
				INPUT_PLACEHOLDER.copy_(fake_input)
				OUTPUT_PLACEHOLDER = SUB_MODEL(INPUT_PLACEHOLDER)
				#OUTPUT_PLACEHOLDER = OUTPUT_PLACEHOLDER.cpu()
				output_data_storage[j].copy_(OUTPUT_PLACEHOLDER.data.cpu())
				fp_senq.put(output_data_storage[j])
			
			for j in range(args.subitern):
				back_ctx = bp_recq.get()
				OUTPUT_PLACEHOLDER.copy_(output_data_storage[j])
				back_ctx = back_ctx.cuda()
				OUTPUT_PLACEHOLDER.backward(back_ctx, retain_graph=True)
				back_ctx = back_ctx.cpu()
		elif is_tail(wid):
			for j in range(args.subitern):
				input_data = fp_recq.get()
				#print("fp_recq get... ", j)
				INPUT_PLACEHOLDER.copy_(input_data)
				OUTPUT_PLACEHOLDER = SUB_MODEL(INPUT_PLACEHOLDER)
				loss = criterion(OUTPUT_PLACEHOLDER, fake_target.cuda())
				loss.backward()
				back_ctx = GoogleNetHookFunc.backward_ctx.cpu()
				#print("bp_senq put... ", j)
				bp_senq.put(back_ctx.data)
		else:
			fp_recv_cnt = 0
			bp_recv_cnt = 0
			while True:
				if fp_recq.empty()==False and fp_recv_cnt < args.subitern:
					input_data = fp_recq.get()
					INPUT_PLACEHOLDER.copy_(input_data.data)
					OUTPUT_PLACEHOLDER = SUB_MODEL(INPUT_PLACEHOLDER)
					output_data_storage[fp_recv_cnt].copy_(OUTPUT_PLACEHOLDER.data.cpu())
					fp_senq.put(output_data_storage[fp_recv_cnt])
					fp_recv_cnt += 1
					#print("fp_recv_cnt=",fp_recv_cnt)
				if bp_recq.empty() == False and bp_recv_cnt < args.subitern:
					back_ctx = bp_recq.get()
					OUTPUT_PLACEHOLDER.data.copy_(output_data_storage[bp_recv_cnt])
					OUTPUT_PLACEHOLDER.backward(back_ctx.cuda(), retain_graph=True)
					back_ctx = GoogleNetHookFunc.backward_ctx.cpu()
					bp_senq.put(back_ctx.data)
					bp_recv_cnt += 1
					#print("bp_recv_cnt=",bp_recv_cnt)
				if fp_recv_cnt == args.subitern and bp_recv_cnt == args.subitern:
					break 

		SUB_OPTIMIZER.step()
		time_list.append(time.time())
		iter_num = len(time_list)-1
		if iter_num>0:
			print("iter_num=",iter_num,"\t",float(time_list[-1]-time_list[0]*1.0)/iter_num)


def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("GLOO: ", os.environ['MASTER_ADDR'], " ",  os.environ['MASTER_PORT'])
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("completed rank=", rank, " size =", size)

if __name__ == '__main__':
	global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE
	ini_data_storage(args.wid)
	SUB_OPTIMIZER.zero_grad()
	time_list = []
	fp_senq = mp.Queue()
	bp_senq = mp.Queue()
	fp_recq = mp.Queue()
	bp_recq = mp.Queue()

	fp_send_p = mp.Process(target=fp_send_proc, kwargs={"wid":args.wid, "fp_senq":fp_senq})
	bp_send_p = mp.Process(target=bp_send_proc, kwargs={"wid":args.wid, "bp_senq":bp_senq})
	fp_recv_p = mp.Process(target=fp_recv_proc, kwargs={"wid":args.wid, "fp_recq":fp_recq})
	bp_recv_p = mp.Process(target=bp_recv_proc, kwargs={"wid":args.wid, "bp_recq":bp_recq})

	
	#train_p = mp.Process(target=train, kwargs={"wid":args.wid, "fp_senq":fp_senq, "fp_recq":fp_recq, "bp_senq":bp_senq, "bp_recq":bp_recq})

	fp_send_p.start()
	bp_send_p.start()
	fp_recv_p.start()
	bp_recv_p.start()
	

	train(args.wid,fp_senq, fp_recq, bp_senq, bp_recq )

	fp_send_p.join()
	bp_send_p.join()
	fp_recv_p.join()
	bp_recv_p.join()


		



    	

