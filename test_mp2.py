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
global SYNC_FLAG, no_share_variable,TOKEN_LOCKS
TOKEN_LAYERS = 5
TOKEN_LOCKS = [mp.Lock() for i in range(TOKEN_LAYERS)]
SYNC_FLAG = torch.zeros(1, dtype=torch.int32)
SYNC_FLAG = SYNC_FLAG.share_memory_()
no_share_variable = torch.zeros(1, dtype= torch.int32)
INPUT_LIST = [None] * TOKEN_LAYERS
large_tensor = torch.ones(10, 32,512,7,7)
large_tensor = large_tensor.share_memory_()
def init():
	for i in range(TOKEN_LAYERS):
		INPUT_LIST[i] = torch.ones(5, dtype = torch.int32)
		INPUT_LIST[i] = INPUT_LIST[i].share_memory_()

def func1():
	INPUT_LIST[1].mul_(5)
	time.sleep(5)
	print("wid=",wid, " 0:", INPUT_LIST[0])
	print("wid=",wid, " 1:", INPUT_LIST[1])

def func2():
	#INPUT_LIST[0].mul_(3)
	print("Ichange")
	INPUT_LIST[0].data = torch.ones(5, dtype = torch.int32)
	time.sleep(5)
	print("wid=",wid, " 0:", INPUT_LIST[0])
	print("wid=",wid, " 1:", INPUT_LIST[1])

def train_sync_proc(wid):
	global SYNC_FLAG, no_share_variable,TOKEN_LOCKS
	#print("wid ", wid, " ", "SYN ", SYNC_FLAG)
	#print(wid, "  :id ", id(SYNC_FLAG)," ",id(no_share_variable))
	if wid == 0:
		print("allocating...")
		this_tensor = torch.zeros(10, 32,512,7,7)
		print("allocated...")
		large_tensor.copy_(this_tensor)
		print("copyed")
		#func1()
		'''
		print("lock_acqure")
		TOKEN_LOCKS[0].acquire()
		print("after sleep")
		print("SYNC ", SYNC_FLAG, " ", no_share_variable)
		'''

	else:
		time.sleep(5)
		print("2allocating...")
		that_tensor = torch.ones(10, 32,512,7,7)
		print("allocated...")
		large_tensor.copy_(this_tensor)
		print("copyed")
		#func2()
		'''
		TOKEN_LOCKS[0].acquire()
		print("wid = 1 acquired lock")
		SYNC_FLAG += 1
		no_share_variable += 1
		print("SYNC1 ", SYNC_FLAG, " ", no_share_variable)
		time.sleep(10)
		print("wid=1 now I will free the lock for you")
		TOKEN_LOCKS[0].release()
		'''


if __name__ == '__main__':

	ts_list = []
	#init()
	print("main allocating...")
	that_tensor = torch.ones(10, 32,512,7,7)
	print("main allocated...")
	
	for wid in range(2):
	    ts_p = mp.Process(target=train_sync_proc, kwargs={"wid":wid})
	    ts_p.start()
	    ts_list.append(ts_p)
	for wid in range(2):
	    ts_list[wid].join()
	
	'''
	zeone = True
	mem = torch.zeros([10000,100000])
	mem = mem.cuda()
	while True:
		time.sleep(1)
		print("switch")
		if zeone:
			mem = torch.zeros([10000,100000])
			mem = mem.cuda()
		else:
			mem = torch.ones([10000,100000])
			mem = mem.cuda()
	'''