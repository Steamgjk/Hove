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
import torch.distributed as dist
import torch.multiprocessing as mp

# In-place aggregation
os.environ["CUDA_VISIBLE_DEVICES"]='1'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--adjust_epoch', default=3, type=int, help='batch size')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
worker_num = args.wn
worker_id = args.wid # 1|2|3|
batch_size = args.bs
adjust_epoch = args.adjust_epoch
work_partition = [0,14,18,37,-1]
boundary_shape = [[batch_size, 128, 56, 56], [batch_size, 256, 56, 56],[batch_size, 512, 28, 28]]
boundary_size = [batch_size*128*56*56, batch_size*256*56*56, batch_size*512*28*28]
fake_input = torch.randn(batch_size,3,224,224)
fake_target = torch.from_numpy(np.random.randint(0,999,size=batch_size))
print(fake_input.size())
print(fake_target.size())
#VGG19 54
criterion = nn.CrossEntropyLoss()
sta_lidx = work_partition[worker_id]
end_lidx = work_partition[worker_id+1]
sub_net = VGG('VGG19', sta_lidx = sta_lidx, end_lidx = end_lidx)
sub_net.to(device)

def gen_fp_bp_tensor_list(bs, wid, wn):
    global boundary_size, boundary_shape
    fp_head_tensor = None
    fp_tail_tensor = None
    bp_head_tensor = None
    bp_tail_tensor = None
    for i in range(bs):
        if not wid == 0:
            shp = [bs]+boundary_shape[wid-1]
            nume = boundary_size[wid-1]*bs
            fp_head_tensor = torch.zeros(nume,dtype=torch.float)
            fp_head_tensor = fp_head_tensor.reshape(shp)
            bp_head_tensor = torch.zeros(nume,dtype=torch.float)
            bp_head_tensor = bp_head_tensor.reshape(shp)
        if not wid == wn -1:
            shp = [bs]+boundary_shape[wid]
            nume = boundary_size[wid]*bs
            fp_tail_tensor = torch.zeros(nume, dtype=torch.float)
            fp_tail_tensor = fp_tail_tensor.reshape(shp)
            bp_tail_tensor = torch.zeros(nume, dtype=torch.float)
            bp_tail_tensor = bp_tail_tensor.reshape(shp)

        if fp_head_tensor is not None:
            fp_head_tensor =  fp_head_tensor.share_memory_()
        if fp_tail_tensor is not None:
            fp_tail_tensor =  fp_tail_tensor.share_memory_()
        if bp_head_tensor is not None:
            bp_head_tensor =  bp_head_tensor.share_memory_()
        if bp_tail_tensor is not None:
            bp_tail_tensor =  bp_tail_tensor.share_memory_()
    return fp_head_tensor, fp_tail_tensor, bp_head_tensor, bp_tail_tensor

def gen_shared_counter():
    cnters = torch.zeros(4, dtype=torch.float)
    cnters = cnters.share_memory_()
    return cnters

def fp_send_proc(rank, te, shared_cnters):
    print("ok rank=",rank)
    print("shared_cnters=",shared_cnters)
    te1 = te[1]
    te1[0][0][0][0] += 1
    print("te1  =", te1[0][0][0][0])

def comm_proc1(rank):
    print("comm_proc1 ", rank)
    os.environ['MASTER_ADDR'] = '12.12.10.12'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="gloo", rank=rank, world_size=2, group_name = "comm_proc1")
    if rank == 0:
        time.sleep(3)
        ten = torch.zeros(5)
        dist.send(tensor = ten, dst = 1)
        time.sleep(3)
        dist.recv(tensor = ten, src = 1)
        print("Recv0 ", ten)
    elif rank == 1:
        time.sleep(3)
        ten = torch.ones(5)
        dist.recv(tensor = ten, src = 0)
        print("Recv1 ", ten)
        time.sleep(3)
        dist.send(tensor = ten, dst = 0)
        print("OS ", os.environ)

def comm_proc2(rank):
    print("comm_proc2 ", rank)
    os.environ['MASTER_ADDR'] = '12.12.10.12'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend="gloo", rank=rank, world_size=2, group_name = "comm_proc2")
    if rank == 0:
        ten = torch.ones(5)
        dist.send(tensor = ten, dst = 1)
    elif rank == 1:
        ten = torch.zeros(5)
        dist.recv(tensor = ten, src = 0)
        print("Recv2 ", ten) 
        print("OS2 ", os.environ)       
def extra_proc():
    print("extra proc")
    os.environ['MASTER_ADDR'] = '12.12.10.12'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="gloo", rank=2, world_size=3, group_name = "comm_proc1")
    #while True:
    #    time.sleep(1)
    print("Will exit")
if __name__ == '__main__':
    wn = args.wn
    wid = args.wid
    bs = args.bs
    num_processes = 2
    comm_proc_list1=[]
    comm_proc_list2=[]
    for rank in range(num_processes):
        comm_p1 = mp.Process(target=comm_proc1, args={rank,})
        comm_p1.start()
        comm_proc_list1.append(comm_p1)
    #extra_p1 = mp.Process(target=extra_proc)
    #extra_p1.start()
    #comm_proc_list1.append(extra_p1)
 
    for rank in range(num_processes):
        comm_p2 = mp.Process(target=comm_proc2, args={rank,})
        comm_p2.start()
        comm_proc_list2.append(comm_p2)    

    for proc in comm_proc_list1:
        proc.join()

    for proc in comm_proc_list2:
        proc.join()

'''
if __name__ == '__main__':
    wn = args.wn
    wid = args.wid
    bs = args.bs

    num_processes = 2
    te = torch.ones(5)
    fp_send_proc_list=[]
    te = te.share_memory_()
    fp_head_tensor, fp_tail_tensor, bp_head_tensor, bp_tail_tensor = gen_fp_bp_tensor_list(bs, wid, wn)
    tensor_list = [fp_head_tensor, fp_tail_tensor, bp_head_tensor, bp_tail_tensor]
    shared_cnters = gen_shared_counter()
    for rank in range(num_processes):
        fp_send_p = mp.Process(target=fp_send_proc, args={rank,}, kwargs={"shared_cnters":shared_cnters, "te":tensor_list})
        fp_send_p.start()
        fp_send_proc_list.append(fp_send_p)

    for fsp in fp_send_proc_list:
        fsp.join()

'''