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
parser.add_argument('--adjust_epoch', default=3, type=int, help='adjust epoch')
parser.add_argument('--partition', default=[0,4,4,7,7,14,14,23,23,33,33,40,40,53,53,-1], nargs='+', type=int)
parser.add_argument('--sleepn', default=1, type=int, help='sleep time')
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

#FP_MEMS = [None for i in range(args.subitern)]
#BP_MEMS = [None for i in range(args.subitern)]
FP_MEMS = None
BP_MEMS = None

FP_RECV_TENSOR_SIZE = torch.zeros(4,dtype=torch.int32)
FP_RECV_TENSOR_SIZE = FP_RECV_TENSOR_SIZE.share_memory_()
BP_RECV_TENSOR_SIZE = torch.zeros(4,dtype=torch.int32)
BP_RECV_TENSOR_SIZE = BP_RECV_TENSOR_SIZE.share_memory_()

DATA_CHUNK_STORAGE = None

global fp_mem_idx, bp_mem_idx
fp_mem_idx = -1
bp_mem_idx = -1

def get_flops(layer, Batchsz, Cin, Cout, Hin, Hout, Win, Wout):
    if isinstance(layer, torch.nn.Conv2d):
        kernel_sz = layer.kernel_size
        kk = torch.prod(torch.tensor(kernel_sz)).item()
        flops1 = Cout * Cin *kk*Hout * Wout * Batchsz
        flops2 = Cout * Hout * Wout * Batchsz
        flops3 = Cin * Cout * kk * Hin *Win * Batchsz
        flops4 = Cout * Hout  * Wout * Batchsz
        flops = flops1 + flops2 +flops3 + flops4
        return flops
    elif isinstance(layer, torch.nn.Linear):
        flops1 = Batchsz * layer.in_features * layer.out_features
        flops2 = Batchsz * layer.out_features
        flops = flops1 + flops2
        return flops
    elif isinstance(layer, torch.nn.MaxPool2d):
        flops = Batchsz * Cin * Hin * Win
        return flops
    else:
        return 0
'''
def allocate_mem():
	print("allocate_mem")
	input_sz = [args.subitern, args.subbs, 64, 224, 224]
	FP_MEMS[0] = torch.zeros(input_sz)
	FP_MEMS[0] = FP_MEMS[0].share_memory_()
	input_sz = [args.subitern, args.subbs, 128, 112, 112]
	FP_MEMS[1] = torch.zeros(input_sz)
	FP_MEMS[1] = FP_MEMS[1].share_memory_()
	input_sz = [args.subitern,args.subbs, 128, 56, 56]
	FP_MEMS[2] = torch.zeros(input_sz)
	FP_MEMS[2] = FP_MEMS[2].share_memory_()
	input_sz = [args.subitern,args.subbs, 256, 56, 56]
	FP_MEMS[3] = torch.zeros(input_sz)
	FP_MEMS[3] = FP_MEMS[3].share_memory_()
	input_sz = [args.subitern, args.subbs, 512, 28, 28]
	FP_MEMS[4] = torch.zeros(input_sz)
	FP_MEMS[4] = FP_MEMS[4].share_memory_()
	input_sz = [args.subitern, args.subbs, 512, 14, 14]
	FP_MEMS[5] = torch.zeros(input_sz)
	FP_MEMS[5] = FP_MEMS[5].share_memory_()
	input_sz = [args.subitern, args.subbs, 512, 7, 7]
	FP_MEMS[6] = torch.zeros(input_sz)
	FP_MEMS[6] = FP_MEMS[6].share_memory_()
	input_sz = [args.subitern, args.subbs, 4096]
	FP_MEMS[7] = torch.zeros(input_sz)
	FP_MEMS[7] = FP_MEMS[7].share_memory_()  
	print("allocate_mem mid")
	for i in range(8):
		BP_MEMS[i] = torch.zeros(FP_MEMS[i].size())
		BP_MEMS[i] = BP_MEMS[i].share_memory_()
	print("allocate FIN")
	sta_lidx = args.partition[wid+wid]
	end_lidx = args.partition[wid+wid+1]
	fp_mem_idx = -1
	bp_mem_idx = -1
	for i in range(8):
		sz = list(FP_MEMS[i][0].size())
		mem_sz = profile_shape[sta_lidx]["shape"]
		if sz_eq(sz, mem_sz):
			fp_mem_idx = i
			break
		else:
			continue
	for i in range(8):
		sz = list(BP_MEMS[i][0].size())
		mem_sz = profile_shape[end_lidx]["shape"]
		if sz_eq(sz, mem_sz):
			bp_mem_idx = i
			break
		else:
			continue
'''

def ini_data_storage(wid):
    global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE, FP_RECV_TENSORS, BP_RECV_TENSORS, DATA_CHUNK_STORAGE,FP_MEMS,BP_MEMS
    f= open("./vgg_info.dump", "rb")
    profile_list = pickle.load(f)
    f.close()
    sta_lidx = args.partition[wid+wid]
    end_lidx = args.partition[wid+wid+1]
    SUB_MODEL = myVGG("VGG19", sta_lidx = sta_lidx, end_lidx = end_lidx)
    #print("sta_lidx=",sta_lidx," ", "end_lidx=", end_lidx)
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
    INPUT_PLACEHOLDER = INPUT_PLACEHOLDER.share_memory_()
    OUTPUT_PLACEHOLDER = OUTPUT_PLACEHOLDER.share_memory_()
    storage_sz = [args.subitern]+OUTPUT_SIZE
    DATA_CHUNK_STORAGE = torch.zeros(storage_sz)
    '''
    for i in range(args.subitern):
    	FP_MEMS[i] = torch.zeros(INPUT_SIZE)
    	FP_MEMS[i] = FP_MEMS[i].share_memory_()
    	BP_MEMS[i] = torch.zeros(OUTPUT_SIZE)
    	BP_MEMS[i] - BP_MEMS[i].share_memory_()
    
    fp_sz = [args.subitern] + INPUT_SIZE
    bp_sz = [args.subitern] + OUTPUT_SIZE
    '''
    fp_sz = INPUT_SIZE
    bp_sz = OUTPUT_SIZE    
    FP_MEMS = torch.zeros(fp_sz)
    FP_MEMS = FP_MEMS.share_memory_()
    BP_MEMS = torch.zeros(bp_sz)
    BP_MEMS = BP_MEMS.share_memory_()

    #print("FP 0 size = ", FP_MEMS.size())
    #print("BP 0 size = ", BP_MEMS.size())

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
	return (wid+1)* 5 + 1
def get_fp_pre_rank(wid):
	return (wid-1)* 5 + 0
def get_bp_succ_rank(wid):
	return (wid+1)* 5 + 2
def get_bp_pre_rank(wid):
	return (wid-1)* 5 + 3


def fp_send_proc(wid, fp_senq):
	my_rank = wid* 5 + 0
	world_size = args.wn * 5
	init_processes(my_rank, world_size, backend='gloo')
	succ_rank = get_fp_succ_rank(wid)
	sz_tensor = torch.zeros(4, dtype=torch.int32)
	while True:
		if wid + 1 >args.wn-1:
			#time.sleep(1)
			continue
		#print("fp_send_proc:", my_rank,"->",succ_rank)
		if fp_senq.empty()== False:
			send_tensor = fp_senq.get()
			dist.send(tensor = send_tensor, dst = succ_rank)

def bp_send_proc(wid, bp_senq):
	my_rank = wid * 5 + 2
	world_size = args.wn * 5
	init_processes(my_rank, world_size, backend='gloo')
	pre_rank = get_bp_pre_rank(wid)
	sz_tensor = torch.zeros(4, dtype=torch.int32)
	while True:
		if wid - 1 < 0:
			#time.sleep(1)
			continue
		if bp_senq.empty()==False:
			#print("bp_send_proc:", my_rank,"->",pre_rank)
			send_tensor = bp_senq.get()
			dist.send(tensor = send_tensor, dst = pre_rank)
			#print("bp send success")
def fp_recv_proc(wid, fp_recq, FP_MEMS, fp_recv_assistq):
	my_rank = wid * 5 + 1
	world_size = args.wn * 5
	init_processes(my_rank, world_size, backend='gloo')
	pre_rank = get_fp_pre_rank(wid)
	cnt = 0
	sz_tensor = torch.zeros(4, dtype=torch.int32)
	iter_n = -1 # to align
	fp_mem_idx = -1

	while True:
		if wid - 1 < 0:
			#time.sleep(1)
			continue
		#print("fp_recv_proc:", pre_rank,"->",my_rank)
		#recv_tensor = FP_MEMS[fp_mem_idx][cnt]
		#recv_tensor = FP_MEMS[cnt]
		recv_tensor = FP_MEMS
		#print("recv from ",pre_rank,"->",my_rank)
		#print("fp recv sz = ", recv_tensor.size())
		dist.recv(tensor = recv_tensor, src = pre_rank)
		fp_recq.put(recv_tensor)
		cnt += 1
		if cnt == args.subitern:
			cnt = 0
			iter_n += 1
			#print("uter n =", iter_n)
			if iter_n >0 and iter_n % args.adjust_epoch == 0:
				#should update FP_RECV_TENSORS
				FP_MEMS= fp_recv_assistq.get()

			

def bp_recv_proc(wid, bp_recq, BP_MEMS,bp_recv_assistq):
	my_rank = wid * 5 + 3
	world_size = args.wn * 5
	init_processes(my_rank, world_size, backend='gloo')
	succ_rank = get_bp_succ_rank(wid)
	cnt = 0
	iter_n = -1 # to align
	sz_tensor = torch.zeros(4, dtype=torch.int32)
	while True:
		if wid+1 > args.wn-1:
			#time.sleep(1)
			continue
		#print("bp_recv_proc:", succ_rank,"->",my_rank)
		#recv_tensor = BP_RECV_TENSORS[cnt]
		#recv_tensor = BP_MEMS[bp_mem_idx][cnt]
		#recv_tensor = BP_MEMS[cnt]
		recv_tensor = BP_MEMS
		#print("bp recv sz = ", recv_tensor.size())
		dist.recv(tensor = recv_tensor, src = succ_rank)
		#print("bp recv_tensor sz=",recv_tensor.size())
		bp_recq.put(recv_tensor)
		cnt += 1
		if cnt == args.subitern:
			cnt = 0
			iter_n += 1
			if iter_n > 0 and iter_n % args.adjust_epoch == 0:
				BP_MEMS = bp_recv_assistq.get()


def profile_tensor_shape(batch_size):
    profile_shape = []
    break_points = []
    test_net = VGG("VGG19")
    out = fake_input
    cnt = 0
    for layer in test_net.features:
        if isinstance(layer, HookLayer):
            print("HookLayer jump off")
            continue
        in_sz = out.size()
        Cin = in_sz[1]
        Hin = in_sz[2]
        Win = in_sz[3]
        out = layer(out) 
        shp = out.size()
        Cout = shp[1]
        Hout = shp[2]
        Wout = shp[3]
        num = torch.tensor(shp).prod().item()
        layer_type = type(layer)
        flops = get_flops(layer, batch_size, Cin, Cout, Hin, Hout, Win, Wout)
        weight_size = 0
        bias_size = 0
        if(hasattr(layer, "weight")):
            weight_shape = layer.weight.data.size()
            weight_size = torch.tensor(weight_shape).prod().item()
        if(hasattr(layer, "bias")):
            bias_shape = layer.bias.data.size()
            bias_size = torch.tensor(bias_shape).prod().item()
        dict_item = {"shape":shp,"size":num,"type":layer_type, "flops": flops,"weight_shape":weight_shape, "weight_size":weight_size,"bias_shape":bias_shape, "bias_size":bias_size, "bias_size":bias_size}
        profile_shape.append(dict_item)
        if isinstance(layer, torch.nn.Conv2d) and cnt >0 :
            break_points.append(cnt)
        cnt = cnt + 1
    out = out.view(out.size(0), -1)
    for layer in test_net.fc_layers:
        out = layer(out)
        shp = out.size()
        num = torch.tensor(shp).prod().item()
        layer_type = type(layer)
        flops = get_flops(layer, batch_size, -1, -1, -1, -1, -1, -1)
        if(hasattr(layer, "weight")):
            weight_shape = layer.weight.data.size()
            weight_size = torch.tensor(weight_shape).prod().item()
        if(hasattr(layer, "bias")):
            bias_shape = layer.bias.data.size()
            bias_size = torch.tensor(bias_shape).prod().item()
        dict_item = {"shape":shp,"size":num,"type":layer_type, "flops": flops,"weight_shape":weight_shape, "weight_size":weight_size,"bias_shape":bias_shape, "bias_size":bias_size, "bias_size":bias_size}
        profile_shape.append(dict_item)

    layer = test_net.classifier
    out = layer(out)
    shp = out.size()
    num = torch.tensor(shp).prod()
    layer_type = type(layer)
    flops = get_flops(layer, batch_size, -1, -1, -1, -1, -1, -1)
    if(hasattr(layer, "weight")):
        weight_shape = layer.weight.data.size()
        weight_size = torch.tensor(weight_shape).prod().item()
    if(hasattr(layer, "bias")):
        bias_shape = layer.bias.data.size()
        bias_size = torch.tensor(bias_shape).prod().item()
    dict_item = {"shape":shp,"size":num,"type":layer_type, "flops": flops,"weight_shape":weight_shape, "weight_size":weight_size,"bias_shape":bias_shape, "bias_size":bias_size, "bias_size":bias_size}
    profile_shape.append(dict_item)
    return profile_shape,break_points

def auto_tune(wid, profile_shape,break_points, speed, total_flops):
	global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE, DATA_CHUNK_STORAGE
	print("auto_tune ", speed, " ", total_flops)
	if wid == 0:
		req_list = []
		speed_tensor = torch.zeros([args.wn, 1], dtype=torch.float)
		speed_tensor[0][0] = speed
		req_list.append(None)
		#print("recving...")
		for i in range(1, args.wn):
			#print("i=",i)
			src_rank = i * 5 + 4
			req = dist.irecv(tensor=speed_tensor[i], src = src_rank)
			req_list.append(req)
		#print("req....")
		for i in range(1, args.wn):
			req_list[i].wait()
		#print("speed_tensor ", speed_tensor)
		average_workload = total_flops / speed_tensor.sum()
		break_points_num = len(break_points)
		partition_tensor = torch.zeros(args.wn*2, dtype=torch.int)
		sta_lidx = 0
		end_lidx = -1
		for i in range(args.wn):
			expected_workload = average_workload * speed_tensor[i]
			rest_wn = args.wn - 1 - i
			acc_workload = 0
			for j in range(break_points_num - rest_wn):
				acc_workload = 0
				for k in range(sta_lidx, break_points[j]):
					acc_workload += profile_shape[k]["flops"]
				if acc_workload > expected_workload:
					end_lidx = break_points[j]
					break
			if end_lidx == -1:
				end_lidx = break_points[break_points_num - rest_wn-1]
			partition_tensor[i+i] = sta_lidx
			partition_tensor[i+i+1] = end_lidx
			sta_lidx = end_lidx
			end_lidx = -1
			partition_tensor[-1] = -1
		print("new parititon ", partition_tensor)

		seq_list = [None]
		for i in range(1, args.wn):
			seq = dist.isend(tensor = partition_tensor, dst = i*5+4)
			seq_list.append(seq)
		for i in range(1, args.wn):
			seq_list[i].wait()
		#omit data transfer, directly repartition
		for i in range(args.wn+args.wn):
			args.partition[i]=partition_tensor[i]
		ini_data_storage(wid)
	else:
		speed_tensor = torch.zeros(1, dtype=torch.float)
		speed_tensor[0] = speed
		seq = dist.isend(tensor=speed_tensor,dst = 0*5+4)
		seq.wait()
		partition_tensor = torch.zeros(args.wn*2, dtype=torch.int)
		req = dist.irecv(tensor = partition_tensor, src = 0*5+4)
		req.wait()
		for i in range(args.wn+args.wn):
			args.partition[i]=partition_tensor[i]
		ini_data_storage(wid)		


def sz_eq(list1, list2):
	if len(list1)!= len(list2):
		return False
	else:
		for i in range(len(list1)):
			if list1[i] != list2[i]:
				return False
		return True

def train(wid, fp_senq, fp_recq, bp_senq, bp_recq, fp_recv_assistq, bp_recv_assistq ):
	global SUB_MODEL,SUB_OPTIMIZER,INPUT_PLACEHOLDER,OUTPUT_PLACEHOLDER,INPUT_SIZE, OUTPUT_SIZE, FP_MEMS,BP_MEMS

	SUB_OPTIMIZER.zero_grad()

	iter_num = 0
	time_list = []
	output_list = [None for j in range(args.subitern)]
	my_rank = wid * 5 + 4
	world_size = args.wn * 5
	init_processes(my_rank, world_size, backend='gloo')
	periodic_time = 0
	profile_shape,break_points = profile_tensor_shape(args.subbs*args.subitern)
	sta_lidx = args.partition[wid+wid]
	end_lidx = args.partition[wid+wid+1]
	total_flops = 0
	for i in range(len(profile_shape)):
		total_flops += profile_shape[i]["flops"]

	while True:
		sta_time = time.time()
		if iter_num % args.wn == wid:
			print("I need sleep {:d} s".format(args.sleepn))
			if args.sleepn > 0:
				time.sleep(args.sleepn)
		if is_head(wid):
			for j in range(args.subitern):
				INPUT_PLACEHOLDER.copy_(fake_input)
				OUTPUT_PLACEHOLDER = SUB_MODEL(INPUT_PLACEHOLDER)
				#print("OUTPUT_PLACEHOLDER size=", OUTPUT_PLACEHOLDER.size())
				#OUTPUT_PLACEHOLDER = OUTPUT_PLACEHOLDER.cpu()
				DATA_CHUNK_STORAGE[j].copy_(OUTPUT_PLACEHOLDER.data.cpu())
				fp_senq.put(DATA_CHUNK_STORAGE[j])
			
			for j in range(args.subitern):
				back_ctx = bp_recq.get()
				OUTPUT_PLACEHOLDER.copy_(DATA_CHUNK_STORAGE[j])
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
				back_ctx = HookFunc.backward_ctx.cpu()
				#print("bp_senq put... ", j)
				bp_senq.put(back_ctx.data)
		else:
			fp_recv_cnt = 0
			bp_recv_cnt = 0
			while True:
				if fp_recq.empty()==False and fp_recv_cnt < args.subitern:
					input_data = fp_recq.get()
					#print("ddd input_data size = ", input_data.size())
					#print("hh INPUT placeholder_SIZE=", INPUT_PLACEHOLDER.size())
					INPUT_PLACEHOLDER.copy_(input_data.data)
					OUTPUT_PLACEHOLDER = SUB_MODEL(INPUT_PLACEHOLDER)
					DATA_CHUNK_STORAGE[fp_recv_cnt].copy_(OUTPUT_PLACEHOLDER.data.cpu())
					fp_senq.put(DATA_CHUNK_STORAGE[fp_recv_cnt])
					fp_recv_cnt += 1
					#print("fp_recv_cnt=",fp_recv_cnt)
				if bp_recq.empty() == False and bp_recv_cnt < args.subitern:
					back_ctx = bp_recq.get()
					OUTPUT_PLACEHOLDER.data.copy_(DATA_CHUNK_STORAGE[bp_recv_cnt])
					OUTPUT_PLACEHOLDER.backward(back_ctx.cuda(), retain_graph=True)
					back_ctx = HookFunc.backward_ctx.cpu()
					bp_senq.put(back_ctx.data)
					bp_recv_cnt += 1
					#print("bp_recv_cnt=",bp_recv_cnt)
				if fp_recv_cnt == args.subitern and bp_recv_cnt == args.subitern:
					break 
		SUB_OPTIMIZER.step()
		ed_time = time.time()
		periodic_time += ed_time -sta_time
		time_list.append(time.time())
		iter_num = len(time_list)-1
		if iter_num>0:
			print("iter_num=",iter_num,"\t",float(time_list[-1]-time_list[0]*1.0)/iter_num)
			#peridically re-partition 
			if iter_num % args.adjust_epoch == 0:
				total_flops = 0
				for i in range(sta_lidx, end_lidx):
					total_flops += profile_shape[i]["flops"]
				speed = total_flops*1.0/periodic_time
				auto_tune(wid, profile_shape,break_points, speed, total_flops)
				#print("check FP 0 ", FP_MEMS.size())
				#print("check BP 0 ", BP_MEMS.size())
				fp_recv_assistq.put(FP_MEMS)
				bp_recv_assistq.put(BP_MEMS)
				periodic_time = 0
			




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

	fp_recv_assistq = mp.Queue()
	bp_recv_assistq = mp.Queue()

	fp_send_p = mp.Process(target=fp_send_proc, kwargs={"wid":args.wid, "fp_senq":fp_senq})
	bp_send_p = mp.Process(target=bp_send_proc, kwargs={"wid":args.wid, "bp_senq":bp_senq})
	'''
	fp_recv_p = mp.Process(target=fp_recv_proc, kwargs={"wid":args.wid, "fp_recq":fp_recq, "fp_recv_assistq":fp_recv_assistq})
	bp_recv_p = mp.Process(target=bp_recv_proc, kwargs={"wid":args.wid, "bp_recq":bp_recq, "bp_recv_assistq":bp_recv_assistq})
	'''
	
	'''
	fp_recv_p = mp.Process(target=fp_recv_proc, kwargs={"wid":args.wid, "fp_recq":fp_recq, "FP_MEMS":FP_MEMS})
	bp_recv_p = mp.Process(target=bp_recv_proc, kwargs={"wid":args.wid, "bp_recq":bp_recq, "BP_MEMS":BP_MEMS})
	'''
	#train_p = mp.Process(target=train, kwargs={"wid":args.wid, "fp_senq":fp_senq, "fp_recq":fp_recq, "bp_senq":bp_senq, "bp_recq":bp_recq})

	fp_recv_p = mp.Process(target=fp_recv_proc, kwargs={"wid":args.wid, "fp_recq":fp_recq, "FP_MEMS":FP_MEMS,"fp_recv_assistq":fp_recv_assistq})
	bp_recv_p = mp.Process(target=bp_recv_proc, kwargs={"wid":args.wid, "bp_recq":bp_recq, "BP_MEMS":BP_MEMS,"bp_recv_assistq":bp_recv_assistq})

	fp_send_p.start()
	bp_send_p.start()
	fp_recv_p.start()
	bp_recv_p.start()
	

	#train(args.wid,fp_senq, fp_recq, bp_senq, bp_recq, FP_MEMS, BP_MEMS )
	train(args.wid,fp_senq, fp_recq, bp_senq, bp_recq, fp_recv_assistq, bp_recv_assistq )

	fp_send_p.join()
	bp_send_p.join()
	fp_recv_p.join()
	bp_recv_p.join()


		



    	

