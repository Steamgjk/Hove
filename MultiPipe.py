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

mp.set_start_method("spawn", force=True)
# In-place aggregation
#os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--nproc', default=1, type=int, help='number of procs')
parser.add_argument('--gn', default=3, type=int, help='number of groups')
parser.add_argument('--gsz', default=3, type=int, help='size of group')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--adjust_epoch', default=3, type=int, help='adjust_epoch')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
adjust_epoch = args.adjust_epoch
subbs = args.subbs
'''
work_partition = [0,14,18,37,-1]
boundary_shape = [[subbs, 128, 56, 56], [subbs, 256, 56, 56],[subbs, 512, 28, 28]]
boundary_size = [subbs*128*56*56, subbs*256*56*56, subbs*512*28*28]
'''
work_partition = [0,14,37,-1]
boundary_shape = [[subbs, 128, 56, 56], [subbs, 512, 28, 28]]
boundary_size = [subbs*128*56*56, subbs*512*28*28]
#VGG19 54



def gen_fp_bp_tensor_list(iter_thresh, wid, wn, gn, gsz, wrank):
    global boundary_size, boundary_shape
    fp_head_list = []
    fp_tail_list = []
    bp_head_list = []
    bp_tail_list = []
    for i in range(iter_thresh):
        fp_head_tensor = None
        fp_tail_tensor = None
        bp_head_tensor = None
        bp_tail_tensor = None
        if not wrank == 0:
            fp_head_tensor = torch.zeros(boundary_shape[wrank-1],dtype=torch.float)
            bp_head_tensor = torch.zeros(boundary_shape[wrank-1],dtype=torch.float)
        if not wrank == gsz -1:
            fp_tail_tensor = torch.zeros(boundary_shape[wrank], dtype=torch.float)
            bp_tail_tensor = torch.zeros(boundary_shape[wrank], dtype=torch.float)
        if fp_head_tensor is not None:
            fp_head_tensor =  fp_head_tensor.share_memory_()
        if fp_tail_tensor is not None:
            fp_tail_tensor =  fp_tail_tensor.share_memory_()
        if bp_head_tensor is not None:
            bp_head_tensor =  bp_head_tensor.share_memory_()
        if bp_tail_tensor is not None:
            bp_tail_tensor =  bp_tail_tensor.share_memory_()
        fp_head_list.append(fp_head_tensor)
        fp_tail_list.append(fp_tail_tensor)
        bp_head_list.append(bp_head_tensor)
        bp_tail_list.append(bp_tail_tensor)
    return fp_head_list, fp_tail_list, bp_head_list, bp_tail_list

def gen_shared_counter():
    cnters = torch.zeros(4, dtype=torch.int32)
    cnters = cnters.share_memory_()
    return cnters

'''
def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '12.12.11.11'
    os.environ['MASTER_PORT'] = '29400'
    dist.init_process_group(backend, rank=rank, world_size=size)
'''

def init_processes(comm_rank, wid, wn, nproc, gn, gsz, backend='gloo'):
    """ Initialize the distributed environment. """
    world_sz = nproc * wn * 4 + wn
    #os.environ['MASTER_ADDR'] = '12.12.11.11'
    #os.environ['MASTER_PORT'] = '29311'
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("Init Process comm_rank=",comm_rank, " master addr =",args.ip, "  master port=",args.prt )
    dist.init_process_group(backend, rank=comm_rank, world_size=world_sz)
    comm_gp_list = []
    base = nproc * wn *4
    for i in range(gsz):
        ranks = []
        for j in range(gn):
            ranks.append(base+i+j*gsz)
        comm_gp = dist.new_group(ranks=ranks, backend='gloo')
        comm_gp_list.append(comm_gp)
    return comm_gp_list


def fp_send_proc(rank, bs, subbs, wid, wn, wrank, nproc, gn,gsz, fp_tail_list, shared_cnters):
    #world_sz =  nproc * wn *4  #+1
    #fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    comm_rank =  wid * nproc*4 + rank*4
    iter_thresh = bs/subbs 
    #init_processes(comm_rank, world_sz)
    init_processes(comm_rank, wid, wn, nproc, gn, gsz, backend='gloo')
    #print("fp_send_proc comm_rank=", comm_rank)
    #if wid == wn -1:
    if wrank == gsz -1:
        shared_cnters[1] = 4
        return
    local_fp_sent_counter = 0
    dst_rank = (wid +1)*nproc*4+rank*4+1
    while True:
        #print("fp send ", local_fp_sent_counter, " ", shared_cnters[1])
        #fp_tail_tensor
        if local_fp_sent_counter < shared_cnters[1]:
            # is it okay to directly send gpu tensor?
            #print("fp send ", comm_rank, "  -> ", dst_rank)
            dist.send(tensor = fp_tail_list[local_fp_sent_counter], dst = dst_rank )
            #print("fp send ", fp_tail_list[local_fp_sent_counter].numel())
            #print("fin fp send ", comm_rank, "  -> ", dst_rank)
            local_fp_sent_counter += 1
        else:
            time.sleep(0.001)
        if local_fp_sent_counter == iter_thresh:
            #reset 
            local_fp_sent_counter = 0
            shared_cnters[1].zero_()

def bp_send_proc(rank, bs, subbs, wid, wn, wrank, nproc,gn,gsz, bp_head_list, shared_cnters):
    #world_sz =  nproc * wn *4  #+1
    #fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    comm_rank =  wid * nproc*4 + rank*4 + 2
    iter_thresh = int(bs/subbs) 
    #init_processes(comm_rank, world_sz)
    init_processes(comm_rank, wid, wn, nproc, gn, gsz, backend='gloo')
    print("bp_send_proc comm_rank=", comm_rank)
    #if wid == 0:
    if wrank == 0:
        shared_cnters[2] = 0
        return
    local_bp_sent_counter = 0
    dst_rank = (wid-1)*nproc*4+rank*4+3
    while True:
        if local_bp_sent_counter < shared_cnters[2]:
            dist.send(tensor = bp_head_list[local_bp_sent_counter], dst = dst_rank )
            #print("bp send ", bp_head_list[local_bp_sent_counter].numel())
            local_bp_sent_counter += 1
        else:
            time.sleep(0.001)
        if local_bp_sent_counter == iter_thresh:
            local_bp_sent_counter = 0
            shared_cnters[2].zero_()

def fp_recv_proc(rank, bs, subbs, wid, wn, wrank, nproc, gn,gsz, fp_head_list, shared_cnters):
    world_sz =  nproc * wn *4  #+1
    #proc fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    comm_rank =  wid * nproc*4 + rank*4 + 1
    iter_thresh = int(bs/subbs) 
    #init_processes(comm_rank, world_sz)
    init_processes(comm_rank, wid, wn, nproc, gn, gsz, backend='gloo')
    #print("fp_recv_proc comm_rank=", comm_rank)
    #if wid == 0:
    if wrank == 0:
        shared_cnters[0] = iter_thresh
        return
    src_rank = (wid-1) * nproc*4 + rank*4
    while True:
        if shared_cnters[0] < iter_thresh:
            #print("fp recv  ", comm_rank, " <- ", src_rank, " ", shared_cnters[0], " ", bs)
            dist.recv(tensor = fp_head_list[shared_cnters[0]], src = src_rank)
            shared_cnters[0] += 1
            #print("Fin fp recv  ", comm_rank, " <- ", src_rank, " ", shared_cnters[0])
        else:
            time.sleep(0.001)

def bp_recv_proc(rank, bs, subbs, wid, wn, wrank, nproc, gn,gsz, bp_tail_list, shared_cnters):
    #world_sz =  nproc * wn *4  #+1
    #fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    comm_rank =  wid * nproc*4 + rank*4 + 3
    iter_thresh = int(bs/subbs) 
    #init_processes(comm_rank, world_sz)
    init_processes(comm_rank, wid, wn, nproc, gn, gsz, backend='gloo')
    print("bp_recv_proc comm_rank=", comm_rank)
    #if wid == wn-1:
    if wrank == gsz -1:
        shared_cnters[3] = iter_thresh
        return
    src_rank = (wid+1) * nproc*4 + rank*4+2
    while True:
        if shared_cnters[3] < iter_thresh:
            dist.recv(tensor = bp_tail_list[shared_cnters[3]], src = src_rank)
            shared_cnters[3] += 1
        else:
            time.sleep(0.001)


def train_proc(rank, bs, subbs, wid, wn, wrank, nproc, gn,gsz, sub_net, sync_lock, fp_head_list, fp_tail_list, bp_head_list, bp_tail_list, shared_cnters, sync_counter, global_step, grad_dict):
    print("train_proc rank=", rank, " wid=", wid)
    pid = os.getpid()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    iter_thresh = int(bs/subbs) 
    fp_iter = 0
    bp_iter = 0
    inputs = None
    outputs = None
    #if wid == 0:
    if wrank == 0:
        fake_input = torch.randn(subbs,3,224,224)
        print(fake_input.size())
    #if wid == wn -1:
    if wrank == gsz -1:
        fake_target = torch.from_numpy(np.random.randint(0,999,size=subbs))
        criterion = nn.CrossEntropyLoss()
        print(fake_target.size())
    qu = Queue.Queue()
    local_step = 0
    sta = time.time()
    while True:
        if not (local_step == global_step):
            time.sleep(0.001)
            #print(local_step, " == ", global_step)
            continue
        #print("Ok train")
        if wrank == 0:
            #先查BP 再 FP
            #fp_head_tensor_list, fp_tail_tensor_list, bp_head_tensor_list, bp_tail_tensor_list

            if bp_iter < fp_iter:
                if bp_iter < shared_cnters[3]:
                    backward_ctx = bp_tail_list[bp_iter].cuda()
                    outputs = qu.get()
                    outputs.backward(backward_ctx)
                    bp_iter += 1
                    #print(wid," ", rank, "  bp complete  ", fp_iter, " ", bp_iter)
                if bp_iter == iter_thresh:
                    #bp_to_recv has reached bs, then it is time to update grad and reset cnter
                    sync_lock.acquire()
                    #optimizer.step()
                    #update grad_dict
                    for name, param in sub_net.named_parameters():
                        grad_dict[name].add_(param.grad)
                        #param.grad.zero_()
                    sync_counter += 1
                    #print("add one sync_counter ", sync_counter)
                    sync_lock.release()
                    optimizer.zero_grad()
                    fp_iter = 0
                    bp_iter = 0
                    shared_cnters[3].zero_()
                    local_step += 1
                    
                    #print(wid, " global_step:", global_step)
            #FP has not reached the threshold and can be executed
            if fp_iter < shared_cnters[0]:
                inputs = fake_input.cuda()
                outputs = sub_net(inputs)
                fp_tail_list[fp_iter].copy_(outputs)
                qu.put(outputs)
                shared_cnters[1] += 1
                fp_iter += 1
                #print(wid," ", rank, "  fp complete  ", fp_iter, "  ", bp_iter)

        #elif wid == wn -1:
        elif wrank == gsz -1:
            #print("last worker")
            #FP has not reached the threshold and can be executed
            if fp_iter < shared_cnters[0]:
                fp_head_list[fp_iter].requires_grad = True
                inputs = fp_head_list[fp_iter].cuda()
                outputs = sub_net(inputs)
                #shared_cnters[1] += 1
                fp_iter += 1
                target = fake_target.cuda()
                loss = criterion(outputs, target)
                loss.backward()
                #print(HookFunc.hook_dict)
                #time.sleep(5)
                #bp_ctx = HookFunc.hook_dict[pid]
                if HookFunc.hook_dict[pid] is not None:
                    #should be forked
                    bp_head_list[bp_iter].copy_(HookFunc.hook_dict[pid])
                    HookFunc.hook_dict[pid] = None
                    shared_cnters[2] += 1
                else:
                    print("Err")
                    exit(-1)
                bp_iter += 1            
                if bp_iter == iter_thresh:
                    #bp_to_recv has reached bs, then it is time to update grad and reset cnter
                    sync_lock.acquire()
                    #optimizer.step()
                    #optimizer.zero_grad()
                    for name, param in sub_net.named_parameters():
                        grad_dict[name].add_(param.grad)
                        #param.grad.zero_()
                    sync_counter += 1
                    #print("add one sync_counter ", sync_counter)
                    sync_lock.release()
                    optimizer.zero_grad()
                    fp_iter = 0
                    bp_iter = 0
                    shared_cnters[0].zero_()
                    local_step += 1
        else:
            #middle
            #print("ff ", fp_iter, "  ", shared_cnters[0], " ", bp_iter)
            if bp_iter < fp_iter:
                #print("Pre fp vs bp ", fp_iter, " ", bp_iter)
                if bp_iter < shared_cnters[3]:
                    backward_ctx = bp_tail_list[bp_iter].cuda()
                    outputs = qu.get()
                    outputs.backward(backward_ctx)
                    #bp_ctx = HookFunc.hook_dict[pid]
                    #exec('bp_ctx = HookFunc_{}.hook_dict["backward_ctx"]'.format(rank))
                    if HookFunc.hook_dict[pid] is not None:
                        #should be forked
                        bp_head_list[bp_iter].copy_(HookFunc.hook_dict[pid])
                        #exec('HookFunc_{}.hook_dict["backward_ctx"]=None'.format(rank))
                        HookFunc.hook_dict[pid] = None
                        shared_cnters[2] += 1
                    else:
                        print("Err")
                        exit(-1)
                    bp_iter += 1
                    #print("fp vs bp ", fp_iter, " ", bp_iter)
                if bp_iter == iter_thresh:
                    #bp_to_recv has reached bs, then it is time to update grad and reset cnter
                    sync_lock.acquire()
                    #optimizer.step()
                    #update grad_dict
                    for name, param in sub_net.named_parameters():
                        grad_dict[name].add_(param.grad)
                        #param.grad.zero_()
                    sync_counter += 1
                    #print("add one sync_counter ", sync_counter)
                    sync_lock.release()
                    optimizer.zero_grad()
                    fp_iter = 0
                    bp_iter = 0
                    shared_cnters[0].zero_()
                    shared_cnters[3].zero_()
                    local_step += 1

            #FP has not reached the threshold and can be executed
            #print("ff ", fp_iter, "  ", shared_cnters[0])
            if fp_iter < shared_cnters[0]:
                fp_head_list[fp_iter].requires_grad = True
                inputs = fp_head_list[fp_iter].cuda()
                outputs = sub_net(inputs)
                qu.put(outputs)
                fp_tail_list[fp_iter].copy_(outputs)
                shared_cnters[1] += 1
                fp_iter += 1
                #print(wid,"  fp complete")



def sync_proc(bs, subbs, wid, wn, wrank, nproc, gn,gsz, sub_net, sync_lock, grad_dict, shared_cnters, sync_counter, global_step):
    
    comm_rank =  nproc * wn * 4 + wid
    comm_gp_list = init_processes(comm_rank, wid, wn, nproc, gn, gsz, backend='gloo')
    optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    print("sync_proc")

    for name, param in sub_net.named_parameters():
        param.grad = grad_dict[name]

    while True:
        if sync_counter == nproc:
            #print("should Sync") #Allreduce
            for name, param in sub_net.named_parameters(): 
                #print(name, " Reducing...  ", param.numel())
                grad_tensor = param.grad.cpu()
                #print( "Grad CPU Reducing...  ", grad_tensor.numel())
                dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM, group=comm_gp_list[wrank], async_op=False)
                #dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=comm_gp_list[wrank], async_op=False)
                #print(name, " Reduced...")
                param.grad.copy_(grad_tensor)
                #print(" Copied... ", param.grad.device)
            #print("Sync Fin")
            for name, param in sub_net.named_parameters(): 
                param.grad.div_(gn)
            #print("Div Fin")
            #Then     
            optimizer.step()
            optimizer.zero_grad()
            sync_counter.zero_()
            global_step += 1
            #print("global_step=",global_step)
            if global_step == 10:
                sta = time.time()
                print("sta  ", sta)
            if global_step > 10 and global_step % 10 == 0:
                ed =time.time()
                ac_time = ed - sta
                print(global_step.item(), " ", float(ac_time)/(global_step.item()-10))
        else:
            time.sleep(0.001)
            #print(sync_counter, " == ", nproc)



            


def gen_shared_grad(sub_net):
    grad_dict = {}
    for name, param in sub_net.named_parameters():
        grad = param.data.clone()
        grad.zero_()
        grad = grad.share_memory_()
        grad_dict[name] = grad
    return grad_dict

if __name__ == '__main__':
    wn = args.wn
    wid = args.wid
    bs = args.bs
    subbs = args.subbs
    gn = args.gn
    gsz = args.gsz
    master_ip = args.ip
    master_port = args.prt
    wrank = wid % gsz
    iter_thresh = int(bs/subbs)
    num_processes = args.nproc
    criterion = nn.CrossEntropyLoss()
    sta_lidx = work_partition[wrank]
    end_lidx = work_partition[wrank+1]
    sub_net = VGG('VGG19', sta_lidx = sta_lidx, end_lidx = end_lidx)
    sub_net.to(device)
    train_proc_list = []
    sync_proc_list = []
    fp_send_proc_list = []
    fp_recv_proc_list = []
    bp_send_proc_list = []
    bp_recv_proc_list = []
    # fp_to_send, fp_recved, bp_send, bp_recv, grad_aggregated  should be conter auto-increment
    sub_net.share_memory()
    grad_dict = gen_shared_grad(sub_net)
    sync_lock = mp.Lock()
    sync_counter = torch.zeros(1, dtype=torch.int32)
    sync_counter = sync_counter.share_memory_()
    global_step = torch.zeros(1, dtype=torch.int32)
    global_step = global_step.share_memory_()

    for rank in range(num_processes):
        #fp_head_tensor, fp_tail_tensor, bp_head_tensor, bp_tail_tensor = gen_fp_bp_tensor_list(bs, wid, wn)
        fp_head_list, fp_tail_list, bp_head_list, bp_tail_list= gen_fp_bp_tensor_list(iter_thresh, wid, wn, gn, gsz, wrank)
        #print(fp_tail_tensor.size())
        #print("########")
        shared_cnters = gen_shared_counter()
    
        #rank, bs, wid, wn,fp_tail_list, shared_cnters
        fp_send_p = mp.Process(target=fp_send_proc, kwargs={"rank":rank, "bs":bs, "subbs":subbs, "wid":wid, "wn": wn, "wrank": wrank,  "nproc":num_processes, "gn":gn, "gsz":gsz, "fp_tail_list":fp_tail_list, "shared_cnters":shared_cnters})
        fp_send_p.start()
        fp_send_proc_list.append(fp_send_p)

        fp_recv_p = mp.Process(target=fp_recv_proc, kwargs={"rank":rank, "bs":bs, "subbs":subbs, "wid":wid, "wn": wn, "wrank": wrank, "nproc":num_processes, "gn":gn, "gsz":gsz, "fp_head_list": fp_head_list, "shared_cnters":shared_cnters})
        fp_recv_p.start()
        fp_recv_proc_list.append(fp_recv_p)

        bp_send_p = mp.Process(target=bp_send_proc, kwargs={"rank":rank, "bs":bs, "subbs":subbs, "wid":wid, "wn": wn, "wrank": wrank, "nproc":num_processes, "gn":gn, "gsz":gsz, "bp_head_list": bp_head_list, "shared_cnters":shared_cnters})
        bp_send_p.start()
        bp_send_proc_list.append(bp_send_p)

        bp_recv_p = mp.Process(target=bp_recv_proc, kwargs={"rank":rank, "bs":bs, "subbs":subbs, "wid":wid, "wn": wn, "wrank": wrank, "nproc":num_processes, "gn":gn, "gsz":gsz, "bp_tail_list": bp_tail_list, "shared_cnters":shared_cnters})
        bp_recv_p.start()
        bp_recv_proc_list.append(bp_recv_p)        

        train_p = mp.Process(target=train_proc, kwargs={"rank":rank, "bs":bs, "subbs":subbs, "wid":wid, "wn": wn, "wrank": wrank, "nproc":num_processes,"gn":gn, "gsz":gsz, "sub_net": sub_net, "sync_lock":sync_lock, "fp_head_list":fp_head_list, "fp_tail_list": fp_tail_list, "bp_head_list":bp_head_list, "bp_tail_list":bp_tail_list, "shared_cnters":shared_cnters,"sync_counter":sync_counter,  "global_step": global_step, "grad_dict":grad_dict})
        train_p.start()
        train_proc_list.append(train_p)

    sync_p = mp.Process(target=sync_proc, kwargs={"bs":bs, "subbs":subbs, "wid":wid, "wn": wn, "wrank": wrank, "nproc":num_processes,"gn":gn, "gsz":gsz, "sub_net": sub_net, "sync_lock":sync_lock, "grad_dict": grad_dict, "shared_cnters":shared_cnters, "sync_counter":sync_counter, "global_step": global_step})
    sync_p.start()
    sync_p.join()
    #sync_proc_list.append(sync_p)

    for fsp in fp_send_proc_list:
        fsp.join()
    for frp in fp_recv_proc_list:
        frp.join()
    for bsp in bp_send_proc_list:
        bsp.join()
    for brp in bp_recv_proc_list:
        brp.join()
    for tp in train_proc_list:
        tp.join()


