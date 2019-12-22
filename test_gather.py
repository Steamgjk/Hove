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

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "44442"
    dist.init_process_group(backend, rank=rank, world_size=size)

def gather_proc(rank):
	print("rank = ", rank)
	if rank == 0:
		ga_t = torch.zeros(2)
	if rank == 1:
		ga_t = torch.ones(4)
	if rank == 2:
		ga_t = torch.ones(4)*2
	if rank == 3:
		ga_t = torch.ones(4)*3
	print("gather tensor = ", ga_t)
	init_processes(rank, 4, backend='gloo')
	if rank == 3:
		g1 = torch.zeros(4)
		g2 = torch.zeros(4)
		g3 = torch.zeros(4)
		g4 = g3
		gather_list=[g1,g2,g3,g4]
		dist.gather(tensor= ga_t, gather_list = gather_list, dst = 3 )
		print(gather_list)
	else:
		dist.gather(tensor= ga_t, gather_list = [], dst = 3 )

if __name__ == '__main__':
	proc_list = []
	for rank in range(4):
		pr = mp.Process(target=gather_proc, kwargs={"rank": rank})
		pr.start()
		proc_list.append(pr)
	for pr in proc_list:
		pr.join()