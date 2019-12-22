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




if __name__ == '__main__':
	bs = args.bs
	itern = args.itern
	subbs = args.subbs
	input_tensor = torch.randn(bs,3,32,32)
	TOKEN_LAYERS = 7
	switch = args.switch
	if switch == 0:
		re = GoogLeNet()
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

	f= open("./googlenet_info.dump", "rb")
	profile_list = pickle.load(f)
	f.close()
	shp = profile_list[5]["shape"]
	print("shp=",shp)
	shp[0] = bs
	input_tensor = torch.randn(shp)
	subnet = GoogLeNet(sta_lidx = 5, ed_lidx = -1)
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



	
	
	

	

