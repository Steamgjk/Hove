# -*- coding: utf-8 -*
'''Train CIFAR10 with PyTorch.'''
## TODO: 1. Interaction between C and Python  2. Dynamic add/delete net layer  3. Aggregate Gradient
# -*- coding: utf-8 -*
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

mp.set_start_method("spawn", force=True)
# In-place aggregation
#os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--nproc', default=1, type=int, help='number of procs')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
args = parser.parse_args()

'''
if args.wid == 0:
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
'''

def init_processes(comm_rank, wid, wn, nproc, backend='gloo'):
    """ Initialize the distributed environment. """
    world_sz = wn * nproc

    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("Init Process comm_rank=",comm_rank, " master addr =",args.ip, "  master port=",args.prt )
    dist.init_process_group(backend, rank=comm_rank, world_size=world_sz)


def train_proc(rank, bs,  wid, wn, nproc, global_step):
    print("rank: ", rank, " bs: ",bs, " wid: ", wid, " wn: ", wn, " nproc:", nproc)
    pid = os.getpid()
    print("train pid = ", pid)
    wrank = rank * wn + wid
    init_processes(wrank,wid, wn, nproc, 'gloo')
    print("Worker End Connection Initialized") 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sub_net = VGG('VGG19', sta_lidx = None, end_lidx = None)
    sub_net.to(device)
    sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sub_net.train()
    fake_input = torch.randn(bs,3,224,224)
    fake_target = torch.from_numpy(np.random.randint(0,999,size=bs))
    criterion = nn.CrossEntropyLoss()
    sub_optimizer.zero_grad()
    sta = time.time()
    ed = None
    #with torch.autograd.profiler.emit_nvtx():
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        while True:
            
            inputs = fake_input.to(device)
            targets = fake_target.to(device)
            outputs = sub_net(inputs)
            #print(targets.size())  #4
            #print(outputs.size())  # [4,1000]
            #exit(0)
            loss = criterion(outputs, targets)
            loss.backward()
            comm_time_sta = time.time()
            para_num = 0
            for name, parameters in sub_net.named_parameters():
                if(parameters.grad is not None):
                    grad_content = parameters.grad.to("cpu")
                    para_num += grad_content.numel()
                    dist.all_reduce(tensor=grad_content, op = dist.ReduceOp.SUM)
                    grad_content = grad_content/wn 
                    parameters.grad.copy_(grad_content)
            sub_optimizer.step()
            sub_optimizer.zero_grad()
            #print("iter=",iter_n," comm_time=",str(comm_time_ed-comm_time_ed))
            global_step += 1
            
            if global_step ==10:
                sta = time.time()
                print("sta(10) = ", sta)
                cuda.profile_start()
            if global_step > 10 and global_step%10 == 0:
                ed = time.time()
                print("iter_n=",global_step," time=",float(1.0* ed-sta)/(1.0 * global_step-10))
            if global_step == 20:
                cuda.profile_stop()
                print("Break")
                break
    #print(prof)



if __name__ == '__main__':
    pid = os.getpid()
    print("main pid = ", pid)
    wn = args.wn
    wid = args.wid
    bs = args.bs
    master_ip = args.ip
    master_port = args.prt
    num_processes = args.nproc
    criterion = nn.CrossEntropyLoss()
    train_proc_list = []
    global_step = torch.zeros(1, dtype=torch.int32)
    global_step = global_step.share_memory_()
    for rank in range(num_processes):
        train_p = mp.Process(target=train_proc, kwargs={"rank":rank, "bs":bs,  "wid":wid, "wn": wn, "nproc":num_processes, "global_step":global_step})
        train_p.start()
        train_proc_list.append(train_p)
    for train_proc in train_proc_list:
        train_proc.join()
    print("main exit")
    '''
    start_prof = False
    while True:
        if global_step == 10 and start_prof == False:
            print("prof start")
            cuda.profile_start()
            start_prof = True
        if global_step == 20:
            print("prof stop")
            cuda.profile_stop()
            break
    print("exit")
    '''
