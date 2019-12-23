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
parser.add_argument('--convn', default=4, type=int, help='worker number')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--ip', default="12.12.12.101", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--partition', default=[0,26,1, 26,53,1, 53,-1,3, 26,53,2,0,26,2], nargs='+', type=int)
parser.add_argument('--subitern' , default=1, type=int, help='sub itern')
parser.add_argument('--itern', default=1000, type=int, help='itern')
parser.add_argument('--sleepn', default=0, type=int, help='sleep time')
args = parser.parse_args()

fake_input = torch.randn([args.subbs,3,224,224], dtype=torch.float)
fake_target = torch.from_numpy(np.random.randint(0,999,size=int(args.subbs*args.convn)))
fake_input = fake_input.share_memory_()
fake_target = fake_target.share_memory_()
fake_input = fake_input.cuda()
fake_target = fake_target.cuda()

def is_fc(wid):
	if wid < args.convn:
		return False
	else:
		return True
def ini_data_storage():
	pass

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("GLOO: ", os.environ['MASTER_ADDR'], " ",  os.environ['MASTER_PORT'])
    dist.init_process_group(backend, rank=rank, world_size=size)
    conv_ranks = []
    fc_ranks = []
    for i in range(args.convn):
    	conv_ranks.append(i)
    for i in range(args.convn, args.wn):
    	fc_ranks.append(i)

    conv_group = dist.new_group(ranks=conv_ranks, backend=backend)
    fc_group = dist.new_group(ranks=fc_ranks, backend=backend)
    return conv_group, fc_group

def model_sync(wid, mdl, optim, conv_group, fc_group):
	#print("sync wid=",wid)
	conv_n = args.convn
	fc_n = args.wn - args.convn
	if is_fc(wid):
		if fc_n>1:
			for name, parameters in mdl.named_parameters():
				if(parameters.grad is not None):
					#print("fc: ", name, "\t", parameters.grad.size())
					grad_content = parameters.grad.cpu()
					dist.all_reduce(tensor=grad_content, op = dist.ReduceOp.SUM, group=fc_group)
					grad_content = grad_content/fc_n
					parameters.grad.copy_(grad_content.cuda())			
		optim.step()
		optim.zero_grad()
	else:
		for name, parameters in mdl.named_parameters():
			if(parameters.grad is not None):
				#print("conv: ", name, "\t", parameters.grad.size())
				grad_content = parameters.grad.cpu()
				dist.all_reduce(tensor=grad_content, op = dist.ReduceOp.SUM, group=conv_group)
				grad_content = grad_content/conv_n
				parameters.grad.copy_(grad_content.cuda())	
		optim.step()
		optim.zero_grad()	

def train_sync_proc(wid):
	conv_n = args.convn
	fc_n = args.wn - conv_n
	subitern = args.subitern
	itern = args.itern
	conv_group, fc_group = init_processes(wid, args.wn, backend='gloo')
	sta = 0 
	ed = 0
	time_list = []
	if is_fc(wid):
		fc_model = myVGGfc("VGG19")
		fc_optim = optim.SGD(fc_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
		fc_optim.zero_grad()
		fc_model.to("cuda")
		criterion = nn.CrossEntropyLoss()
		for j in range(itern):
			if j % args.wn == wid:
				if args.sleepn > 0:
					print("I need sleep {:d} s".format(args.sleepn))
					time.sleep(args.sleepn)
			for i in range(subitern):
				req_list = []
				input_list = []
				for conv_id in range(conv_n):
					if conv_id % fc_n ==  wid - conv_n:
						input_tensor = torch.zeros([args.subbs,512,7,7])
						#print(int(conv_id), "->", int(wid))
						rq = dist.irecv(tensor = input_tensor, src = conv_id)
						req_list.append(rq)
						input_list.append(input_tensor)
				for rq in req_list:
					rq.wait()

				input_batch = int(args.subbs*conv_n/(fc_n))

				input_data = torch.cat(input_list,0)
				#print("input_data_sz = ",input_data.size())
				input_data = input_data.cuda()
				input_data.requires_grad = True
				output_data = fc_model(input_data)
				output_sz = output_data.size()
				fake_target = torch.from_numpy(np.random.randint(0,999,size=int(output_sz[0])))
				loss = criterion(output_data, fake_target.cuda())
				loss.backward(retain_graph=True)
				
				back_ctx = HookFunc.backward_ctx
				back_ctx = back_ctx.cpu()
				#print("back_ctx sz = ", back_ctx.size())
				seq_list = []
				cnt = 0
				for conv_id in range(conv_n):
					if conv_id % fc_n == wid - conv_n:
						#print("back: ",int(wid),"->",int(conv_id))
						sta = cnt*args.subbs
						send_te = back_ctx[sta:(sta+args.subbs)]
						send_te = send_te.cpu()
						seq = dist.isend(tensor=send_te, dst = conv_id)
						seq_list.append(seq)
						cnt += 1
				#for sq in seq_list:
				#	sq.wait()
				print("sub_iter_n=",i)
			print("iter=",j)
			model_sync(wid, fc_model, fc_optim, conv_group,fc_group)
			time_list.append(time.time())
			iter_num = len(time_list)-1
			if iter_num>0:
				iter_time = float(time_list[-1]*1.0 - time_list[0])/iter_num
				thput = args.subbs*conv_n*args.subitern/iter_time
				print("Iter : ", int(iter_num),"\t", iter_time, "\t", thput)

	else:
		conv_model = myVGGconv("VGG19")
		conv_optim = optim.SGD(conv_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
		conv_model.to("cuda")
		fake_input = torch.randn([args.subbs,3,224,224], dtype=torch.float)
		fake_input = fake_input.cuda()
		conv_optim.zero_grad()
		for j in range(itern):
			if j % args.wn == wid:
				if args.sleepn > 0:
					print("I need sleep {:d} s".format(args.sleepn))
					time.sleep(args.sleepn)
			for i in range(subitern):
				req_list = []
				input_list = []
				output_data = conv_model(fake_input)
				#print("output_data=",output_data.size())
				tosend_data = output_data.cpu()
				target_fc_wid = wid % fc_n + conv_n

				#print(int(wid), "->", int(target_fc_wid))

				seq = dist.isend(tensor =tosend_data, dst = target_fc_wid)

				torecv_data =torch.zeros(tosend_data.size())
				req = dist.irecv(tensor=torecv_data, src = target_fc_wid)

				seq.wait()
				req.wait()

				torecv_data = torecv_data.cuda()
				output_data.backward(torecv_data, retain_graph=True)

			model_sync(wid, conv_model, conv_optim, conv_group, fc_group)
			time_list.append(time.time())
			iter_num = len(time_list)-1
			if iter_num>0:
				iter_time =  float(time_list[-1]*1.0 - time_list[0])/iter_num
				thput = args.subbs*conv_n*args.subitern/iter_time
				print("Iter : ", int(iter_num),"\t", iter_time, "\t", thput)
			


if __name__ == '__main__':
    train_sync_proc(args.wid)

