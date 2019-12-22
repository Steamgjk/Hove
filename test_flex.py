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
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--bs', default=1, type=int, help='total batch size')
parser.add_argument('--chunkn', default=[4,2,2,4,2,2,4], nargs='+', type=int, help='chunk number')
parser.add_argument('--itern', default=20, type=int, help='total batch size')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--switch', default=1, type=int, help='total batch size')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--replica', default="1", type=int, help='Master Port')
parser.add_argument('--partition', default=[0,26,1, 26,53,1, 53,-1,3, 26,53,2,0,26,2], nargs='+', type=int)
args = parser.parse_args()



def test():
    net = ResNet152()
    batch_sz = 16
    net.to("cuda")
    net.zero_grad()
    opt = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    cnt = 0
    time_list = []
    while True:
        input_tensor = torch.randn(batch_sz,3,32,32)
        input_tensor = input_tensor.cuda()
        fin_output = net(input_tensor)
        #print(net.features)
        fake_target = torch.from_numpy(np.random.randint(0,999,size=batch_sz))
        fake_target = fake_target.cuda()
        print(fin_output.size())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(fin_output, fake_target.cuda())
        loss.backward()
        opt.step()
        time_list.append(time.time())
        iter_num = len(time_list)-1
        if iter_num > 0:
            print("iter_num=",iter_num,"\titer_time=",float(time_list[-1]-time_list[0]*1.0)/iter_num)



if __name__ == '__main__':
	bs = args.bs
	itern = args.itern
	subbs = args.subbs
	input_tensor = torch.randn(bs,3,32,32)
	TOKEN_LAYERS = 7
	switch = args.switch
	if switch == 0:
		re = ResNet152()
		re.to("cuda")
		time_list = []
		for i in range(itern):
			input_tensor = input_tensor.cuda()
			output_tensor = re(input_tensor)
			fake_target = torch.from_numpy(np.random.randint(0,999,size=bs))
			fake_target = fake_target.cuda()
			criterion = nn.CrossEntropyLoss()
			loss = criterion(output_tensor, fake_target)
			loss.backward()
			time_list.append(time.time())
			iter_num = len(time_list)-1
			if iter_num>0:
				print("iter_num=",iter_num, (1.0*time_list[-1]-time_list[0])/iter_num)
		exit(0)

	f= open("./resnet_info.dump", "rb")
	profile_list = pickle.load(f)
	f.close()
	shp = profile_list[15]["shape"]
	print("shp=",shp)
	shp[0] = bs
	input_tensor = torch.randn(shp)
	subnet = ResNet152(sta_lidx = 15, ed_lidx = 51)
	subnet.to("cuda")
	input_tensor = input_tensor.cuda()
	time_list = []
	#
	for i in range(itern):
		output_tensor = subnet(input_tensor)
		time_list.append(time.time())
		iter_num = len(time_list)-1
		if iter_num > 0:
			print("iter_num=",iter_num," time=",float(time_list[-1]-time_list[0])/iter_num)



'''
if __name__ == '__main__':
	bs = args.bs
	itern = args.itern
	subbs = args.subbs
	input_tensor = torch.randn(bs,3,32,32)
	TOKEN_LAYERS = 7
	switch = args.switch
	if switch == 0:
		re = ResNet152()
		re.to("cuda")
		time_list = []
		for i in range(itern):
			input_tensor = input_tensor.cuda()
			output_tensor = re(input_tensor)
			fake_target = torch.from_numpy(np.random.randint(0,999,size=bs))
			fake_target = fake_target.cuda()
			criterion = nn.CrossEntropyLoss()
			loss = criterion(output_tensor, fake_target)
			loss.backward()
			time_list.append(time.time())
			iter_num = len(time_list)-1
			if iter_num>0:
				print("iter_num=",iter_num, (1.0*time_list[-1]-time_list[0])/iter_num)
		exit(0)


	
	f= open("./resnet_info.dump", "rb")
	profile_list = pickle.load(f)
	f.close()
	SUB_MODELS = [None] * TOKEN_LAYERS
	SUB_MODELS[0] = ResNet152(sta_lidx = 0, ed_lidx = 15)
	SUB_MODELS[1] = ResNet152(sta_lidx = 15, ed_lidx = 33)
	SUB_MODELS[2] = ResNet152(sta_lidx = 33, ed_lidx = 51)
	SUB_MODELS[3] = ResNet152(sta_lidx = 51, ed_lidx = -1)
	SUB_MODELS[4] = SUB_MODELS[2]
	SUB_MODELS[5] = SUB_MODELS[1]
	SUB_MODELS[6] = SUB_MODELS[0]
	DATA_STORAGE =[None] * 7

	shp1 = profile_list[15]["shape"]
	shp1[0] = args.bs
	shp2 = profile_list[33]["shape"]
	shp2[0] = args.bs
	shp3 = profile_list[51]["shape"]
	shp3[0] = args.bs
	DATA_STORAGE[0] = torch.zeros(shp1)
	DATA_STORAGE[1] = torch.zeros(shp2)
	DATA_STORAGE[2] = torch.zeros(shp3)
	DATA_STORAGE[3] = torch.zeros(shp3)
	DATA_STORAGE[4] = torch.zeros(shp2)
	DATA_STORAGE[5] = torch.zeros(shp1)
	criterion = nn.CrossEntropyLoss()
	chunkn = args.chunkn
	
	for i in range(TOKEN_LAYERS):
		SUB_MODELS[i].to("cuda")
	print("check mem...")
	#time.sleep(5)

	INPUT_PLACE_HOLDERS = [None]*TOKEN_LAYERS
	time_list = []
	for t in range(itern):
		for i in range(TOKEN_LAYERS):
			chunk_num = chunkn[i]
			#CHUNKS = DATA_STORAGE[i].chunk(chunk_num)
			subbs = int(bs/chunk_num)
			print("i=",i, " chunk_num=",chunk_num," subbs=",subbs)
			should_retain_graph = True
			for j in range(chunk_num):
				if j == chunk_num -1:
					should_retain_graph = False
				sta = j* subbs
				if i == 0:
					if should_retain_graph==False:
						input_data = torch.randn(subbs,3,32,32)
						input_data.requires_grad = True
						input_data = input_data.cuda()
						output_data = SUB_MODELS[i](input_data)
						output_data = output_data.cpu()
						INPUT_PLACE_HOLDERS[TOKEN_LAYERS-1-i]= output_data
						input_data = input_data.cpu()
						DATA_STORAGE[i][sta:(sta+subbs)].copy_(output_data.data)
					else:
						with torch.no_grad():
							input_data = torch.randn(subbs,3,32,32)
							input_data.requires_grad = True
							input_data = input_data.cuda()
							output_data = SUB_MODELS[i](input_data)
							output_data = output_data.cpu()
							INPUT_PLACE_HOLDERS[TOKEN_LAYERS-1-i]= output_data
							#print("output_data sz = ",output_data.size(),"  layer = ",TOKEN_LAYERS-1-i )
							input_data = input_data.cpu()
							DATA_STORAGE[i][sta:(sta+subbs)].copy_(output_data.data)
					#print("i=0")

				elif i == 1 or i == 2:
					if should_retain_graph==False:
						input_data = DATA_STORAGE[i-1][sta:(sta+subbs)].detach()
						input_data.requires_grad = True
						input_data = input_data.cuda()
						output_data = SUB_MODELS[i](input_data)
						output_data = output_data.cpu()
						INPUT_PLACE_HOLDERS[TOKEN_LAYERS-1-i]= output_data
						input_data = input_data.cpu()
						DATA_STORAGE[i][sta:(sta+subbs)].copy_(output_data)
					else:
						with torch.no_grad():
							input_data = DATA_STORAGE[i-1][sta:(sta+subbs)]
							input_data.requires_grad = True
							input_data = input_data.cuda()
							output_data = SUB_MODELS[i](input_data)
							output_data = output_data.cpu()
							INPUT_PLACE_HOLDERS[TOKEN_LAYERS-1-i]= output_data
							input_data = input_data.cpu()
							DATA_STORAGE[i][sta:(sta+subbs)].copy_(output_data.data)
					#print("i=1")

				elif i == 3:
					input_data = DATA_STORAGE[i-1][sta:(sta+subbs)].detach()
					input_data.requires_grad = True
					input_data = input_data.cuda()
					fin_output = SUB_MODELS[i](input_data)
					fake_target = torch.from_numpy(np.random.randint(0,999,size=subbs))
					fake_target = fake_target.cuda()
					loss = criterion(fin_output, fake_target)
					loss.backward()
					output_data = ResNetHookFunc.backward_ctx
					fake_target = fake_target.cpu()
					fin_output = fin_output.cpu()
					output_data = output_data.cpu()
					input_data = input_data.cpu()
					DATA_STORAGE[i][sta:(sta+subbs)].copy_(output_data.data)
					#print("i=2")

				elif i == 4 or i == 5:
					input_data = DATA_STORAGE[i-1][sta:(sta+subbs)]
					input_data = input_data.cuda()
					out_data = INPUT_PLACE_HOLDERS[i]
					out_data.copy_(DATA_STORAGE[TOKEN_LAYERS-1-i][sta:(sta+subbs)].data)
					out_data = out_data.cuda()
					out_data.backward(input_data, retain_graph=should_retain_graph)
					output_data = ResNetHookFunc.backward_ctx
					output_data = output_data.cpu()
					input_data = input_data.cpu()
					DATA_STORAGE[i][sta:(sta+subbs)].copy_(output_data.data)
					#print("i=3")

				elif i == 6:
					input_data = DATA_STORAGE[i-1][sta:(sta+subbs)]
					input_data = input_data.cuda()
					out_data = INPUT_PLACE_HOLDERS[i]
					out_data = out_data.cuda()
					out_data.backward(input_data,retain_graph=should_retain_graph)
					output_data = ResNetHookFunc.backward_ctx
					output_data = output_data.cpu()
					#print("output_data sz=",output_data.size())
					input_data = input_data.cpu()
					out_data = out_data.cpu()
					#print("i=4")
		#torch.cuda.empty_cache()
		time_list.append(time.time())
		iter_num = len(time_list)-1
		if iter_num>0:
			print("iter_num=",iter_num, (1.0*time_list[-1]-time_list[0])/iter_num)
	
'''

	

