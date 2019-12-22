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

if __name__ == '__main__':
	mymod = myVGG("VGG19", sta_lidx = 53, end_lidx = -1)
	input_tensor = torch.randn([32,512,7,7])
	input_tensor.requires_grad= True
	input_tensor = input_tensor.cuda()
	mymod.to("cuda")

	output_tensor = mymod(input_tensor)
	criterion = nn.CrossEntropyLoss()
	fake_target = torch.from_numpy(np.random.randint(0,999,size=int(32)))
	loss = criterion(output_tensor, fake_target.cuda())
	loss.backward()
