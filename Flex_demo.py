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
#os.environ["CUDA_VISIBLE_DEVICES"]='1'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--succ', default=-1, type=int, help='successor worker')
parser.add_argument('--pred', default=-1, type=int, help='predecessor worker')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--conv_wid', default=-1, type=int, help='conv worker id')
parser.add_argument('--conv_wn', default=3, type=int, help='conv worker number')
parser.add_argument('--fc_wid', default=-1, type=int, help='fc worker id')
parser.add_argument('--fc_wn', default=1, type=int, help='fc worker number')
parser.add_argument('--pd', default=1, type=int, help='parallel degree')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--subbs', default=1, type=int, help='sub batch size')
parser.add_argument('--nproc', default=1, type=int, help='number of procs')
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--partition', default=[0,28,53,-1], nargs='+', type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

subbs = args.subbs

work_partition = [0,28,53,-1]
boundary_shape = [[subbs, 512, 28, 28], [subbs, 512, 7, 7]]
boundary_size = [subbs*512*28*28, subbs*512*7*7]

successors=[2,2,3,-1]
#VGG19 54

def gen_work_info(partition, sta_lidx, end_lidx, wid, wn, pd, subbs):
    if len(partition)< wn:
        return None
    f= open("./vgg_info.dump", "rb")
    profile_list = pickle.load(f)
    cnt  = 0
    for prof in profile_list:
        print(cnt, " ", prof["shape"])
        cnt += 1
    for layer_id in partition:
        if profile_list[layer_id]["type"]== type(nn.ReLU):
            print("layer_id=",layer_id, " Relu") 
            return None
    print(partition)
    input_shp = profile_list[sta_lidx]["shape"]
    print("pre input_shp=", input_shp)
    input_shp[0] *= (subbs / pd)
    if not wid == wn-1:
        output_shp = profile_list[end_lidx]["shape"]
        print("pre output_shp=", output_shp)
        output_shp[0] *= (subbs / pd)
    else:
        output_shp = None
    print("input_shp=",input_shp, "output_shp=",output_shp)
    #exit(0)
    return input_shp, output_shp

def gen_fp_bp_tensor_list(iter_thresh, wid, wn, input_shp, output_shp):
    #global boundary_size, boundary_shape
    fp_head_list = []
    fp_tail_list = []
    bp_head_list = []
    bp_tail_list = []
    for i in range(iter_thresh):
        fp_head_tensor = None
        fp_tail_tensor = None
        bp_head_tensor = None
        bp_tail_tensor = None
        if not wid == 0:
            fp_head_tensor = torch.zeros(input_shp,dtype=torch.float)
            bp_head_tensor = torch.zeros(input_shp,dtype=torch.float)
        if not wid == wn -1:
            fp_tail_tensor = torch.zeros(output_shp, dtype=torch.float)
            bp_tail_tensor = torch.zeros(output_shp, dtype=torch.float)
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

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    dist.init_process_group(backend, rank=rank, world_size=size)
    conv_ranks_1 = [0,1]
    conv_ranks_2 = [0,1,2]
    conv_group_1 = dist.new_group(ranks=conv_ranks_1, backend=backend)
    conv_group_2 = dist.new_group(ranks=conv_ranks_2, backend=backend)
    return conv_group_1, conv_group_2
    '''
    for i in range(gsz):
        ranks = []
        for j in range(gn):
            ranks.append(base+i+j*gsz)
        comm_gp = dist.new_group(ranks=ranks, backend='gloo')
        comm_gp_list.append(comm_gp)
    return comm_gp_list
    '''


def fp_send_proc(conv_wid, conv_wn, fc_wid, fc_wn, wid, wn,  pred_wid, succ_wid, comm_rank, world_sz, bs, subbs, pd, input_shp, output_shp, fp_tail_list, shared_cnters, sta_lidx, end_lidx):
    #fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    iter_thresh = bs/subbs 
    conv_group_1, conv_group_2 = init_processes(comm_rank, world_sz)
    print("fp_send_proc comm_rank=", comm_rank)
    #if wid == wn -1:
    if succ_wid == -1:
        shared_cnters[1] = 4
        return
    local_fp_sent_counter = 0
    dst_rank = succ_wid * 4 + 1 
    place_tensor_list = [torch.zeros(1)]
    while True:
        #print("fp send ", local_fp_sent_counter, " ", shared_cnters[1])
        #fp_tail_tensor
        if local_fp_sent_counter < shared_cnters[1]:
            # is it okay to directly send gpu tensor?
            #print("fp send ", comm_rank, "  -> ", dst_rank)
            #Hard code
            if wid == 0 or wid == 1:
                dist.gather(tensor= fp_tail_list[local_fp_sent_counter], gather_list = place_tensor_list, dst = dst_rank, group = conv_group_2, async_op=False )
            elif wid == 2:
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

def bp_send_proc(conv_wid, conv_wn, fc_wid, fc_wn, wid, wn,  pred_wid, succ_wid, comm_rank, world_sz, bs, subbs, pd, input_shp, output_shp, fp_tail_list, shared_cnters, sta_lidx, end_lidx):
    #fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    iter_thresh = int(bs/subbs) 
    conv_group_1, conv_group_2 = init_processes(comm_rank, world_sz)
    print("bp_send_proc comm_rank=", comm_rank)
    #if wid == 0:
    if pred_wid == -1:
        shared_cnters[2] = 0
        return
    local_bp_sent_counter = 0
    dst_rank = pred_wid * 4 + 3
    scatter_src = 2 * 4 + 2
    place_tensor = torch.zeros(1)
    while True:
        if local_bp_sent_counter < shared_cnters[2]:
            # hard code
            if wid == 3:
                dist.send(tensor = bp_head_list[local_bp_sent_counter], dst = dst_rank )
            elif wid == 2:
                slist =  bp_head_list[local_bp_sent_counter].chunk(chunks=2, dim=0)
                slist.append(place_tensor)
                dist.scatter(tensor=place_tensor, scatter_list=slist, src=scatter_src, group=conv_group_2, async_op=False)
            local_bp_sent_counter += 1
        else:
            time.sleep(0.001)
        if local_bp_sent_counter == iter_thresh:
            local_bp_sent_counter = 0
            shared_cnters[2].zero_()

def fp_recv_proc(conv_wid, conv_wn, fc_wid, fc_wn, wid, wn,  pred_wid, succ_wid, comm_rank, world_sz, bs, subbs, pd, input_shp, output_shp, fp_tail_list, shared_cnters, sta_lidx, end_lidx):
    world_sz =  nproc * wn *4  #+1
    #proc fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    iter_thresh = int(bs/subbs) 
    conv_group_1, conv_group_2 = init_processes(comm_rank, world_sz)
    #print("fp_recv_proc comm_rank=", comm_rank)
    if pred_wid == -1:
        shared_cnters[0] = iter_thresh
        return
    src_rank = pred_wid * 4 
    place_tensor = torch.zeros(1)
    while True:
        if shared_cnters[0] < iter_thresh:
            #print("fp recv  ", comm_rank, " <- ", src_rank, " ", shared_cnters[0], " ", bs)
            if wid == 3:
                dist.recv(tensor = fp_head_list[shared_cnters[0]], src = src_rank)
            elif wid == 2:
                glist = fp_head_list[shared_cnters[0]].chunks(chunks=2, dim=0)
                glist.append(place_tensor)
                dist.gather(tensor= place_tensor, gather_list = glist, dst = comm_rank, group = conv_group_2, async_op=False )
            shared_cnters[0] += 1
            #print("Fin fp recv  ", comm_rank, " <- ", src_rank, " ", shared_cnters[0])
        else:
            time.sleep(0.001)

def bp_recv_proc(conv_wid, conv_wn, fc_wid, fc_wn, wid, wn,  pred_wid, succ_wid, comm_rank, world_sz, bs, subbs, pd, input_shp, output_shp, fp_tail_list, shared_cnters, sta_lidx, end_lidx):
    #fp_send:0; fp_recv:1; bp_send:2; bp_recv:3 
    iter_thresh = int(bs/subbs) 
    conv_group_1, conv_group_2 = init_processes(comm_rank, world_sz)
    print("bp_recv_proc comm_rank=", comm_rank)
    #if wid == wn-1:
    if succ_wid == -1:
        shared_cnters[3] = iter_thresh
        return
    src_rank = succ_wid * 4 + 2
    place_tensor_list = [torch.zeros(1)]
    while True:
        if shared_cnters[3] < iter_thresh:
            if wid == 2:
                dist.recv(tensor = bp_tail_list[shared_cnters[3]], src = src_rank)
            elif wid == 0 or wid == 1:
                dist.scatter(tensor=bp_tail_list[shared_cnters[3]], scatter_list=place_tensor_list, src=src_rank, group=conv_group_2, async_op=False)
            shared_cnters[3] += 1
        else:
            time.sleep(0.001)



def train_proc(conv_wid, conv_wn, fc_wid, fc_wn, wid, wn, bs, subbs, pd, input_shp, output_shp,  sub_net, sync_lock, fp_head_list, fp_tail_list, bp_head_list, bp_tail_list, shared_cnters, train_step, global_step, sta_lidx, end_lidx):

    pid = os.getpid()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    iter_thresh = int(bs/subbs) 
    fp_iter = 0
    bp_iter = 0
    inputs = None
    outputs = None
    if sta_lidx == 0:
        fake_input = torch.randn(input_shp)
        print(fake_input.size())
    if end_lidx == -1:
        fake_target = torch.from_numpy(np.random.randint(0,999,size=subbs/pd))
        criterion = nn.CrossEntropyLoss()
        print(fake_target.size())
    qu = Queue.Queue()
    local_step = 0
    sta = time.time()
    while True:
        if not (local_step == global_step):
            time.sleep(0.001)
            continue
        #if wid == 0:
        if pred_wid == -1:
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
                    optimizer.step()
                    optimizer.zero_grad()
                    train_step += 1
                    fp_iter = 0
                    bp_iter = 0
                    shared_cnters[3].zero_()
                    local_step += 1
                    #print(wid, " ", sync_iter)
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
        elif succ_wid == -1:
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
                    optimizer.step()
                    optimizer.zero_grad()
                    train_step += 1
                    global_step += 1
                    fp_iter = 0
                    bp_iter = 0
                    shared_cnters[0].zero_()
                    local_step += 1
                    #print(wid, " ", sync_iter)
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
                    optimizer.step()
                    optimizer.zero_grad()
                    train_step += 1
                    global_step += 1
                    fp_iter = 0
                    bp_iter = 0
                    shared_cnters[0].zero_()
                    shared_cnters[3].zero_()
                    local_step += 1
                    #print(wid, " ", sync_iter)

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



def sync_proc(conv_wid, conv_wn, fc_wid, fc_wn, wid, wn, comm_rank, world_sz, bs, subbs, pd,  sub_net, sync_lock, train_step, global_step):
    optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    print("sync_proc")
    conv_group_1, conv_group_2 = init_processes(comm_rank, world_sz)
    sta = 0
    ed = 0
    while True:
        #hard code, only two conv workers that need to sync
        if train_step > global_step:
            #print("should Sync") #Allreduce
            #Then     
            for name, param in sub_net.named_parameters():
                #hard code, two conv workers to sync
                param.data.div_(2)
            for name,param in sub_net.named_parameters():
                cpu_tensor = param.data.cpu()
                dist.all_reduce(tensor=cpu_tensor, op=dist.ReduceOp.SUM, group=conv_group_1, async_op=False)
                param.data.copy_(cpu_tensor)
            
            global_step += 1
            #("sync FIN")
            if global_step == 10:
                sta = time.time()
                print("sta  ", sta)
            if global_step > 10 and global_step % 10 == 0:
                ed =time.time()
                ac_time = ed - sta
                print(global_step.item(), " ", float(ac_time)/(global_step.item()-10))
        else:
            time.sleep(0.001)
        #    print(sync_counter, " == ", nproc)



def gen_shared_grad(sub_net):
    grad_dict = {}
    for name, param in sub_net.named_parameters():
        grad = param.data.clone()
        grad.zero_()
        grad = grad.share_memory_()
        grad_dict[name] = grad
    return grad_dict

if __name__ == '__main__':
    conv_wid = args.conv_wid
    conv_wn = args.conv_wn
    fc_wid = args.fc_wid
    fc_wn = args.fc_wn
    pd = args.pd
    wn = conv_wn + fc_wn
    # each worker has one send proc and one receive proc, and the first two has one additinal sync proc
    world_size = wn * 4 + 2
    fp_send_rank = wid * 4 
    fp_recv_rank = wid * 4 + 1
    bp_send_rank = wid * 4 + 2
    bp_recv_rank = wid * 4 + 3

    if conv_wid < 0:
        wid = conv_wn + fc_wid
    else:
        wid = conv_wid
    succ_wid = args.succ
    pred_wid = args.pred
    bs = args.bs
    subbs = args.subbs
    work_partition = args.partition
    iter_thresh = int(bs/subbs)
    criterion = nn.CrossEntropyLoss()
    sta_lidx = work_partition[wid*2]
    end_lidx = work_partition[wid*2 + 1]
    input_shp, output_shp = gen_work_info(work_partition, sta_lidx, end_lidx, wid, wn, pd, subbs)
    print("Work Partition:", work_partition, input_shp, output_shp)

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
    #grad_dict = gen_shared_grad(sub_net)
    sync_lock = mp.Lock()
    train_step = torch.zeros(1, dtype=torch.int32)
    train_step = train_step.share_memory_()
    global_step = torch.zeros(1, dtype=torch.int32)
    global_step = global_step.share_memory_()


    fp_head_list, fp_tail_list, bp_head_list, bp_tail_list= gen_fp_bp_tensor_list(iter_thresh, wid, wn, input_shp, output_shp)

    shared_cnters = gen_shared_counter()
    
    #rank, bs, wid, wn,fp_tail_list, shared_cnters
    fp_send_p = mp.Process(target=fp_send_proc, kwargs={"conv_wid":conv_wid, "conv_wn":conv_wn, "fc_wid": fc_wid, "fc_wn":fc_wn, "wid":wid, "wn": wn, "pred_wid":pred_wid, "succ_wid":succ_wid,  "comm_rank":fp_send_rank, "world_sz":world_size,  "bs":bs, "subbs":subbs, "pd": pd, "input_shp":input_shp, "output_shp": output_shp, "fp_tail_list":fp_tail_list, "shared_cnters":shared_cnters, "sta_lidx": sta_lidx, "end_lidx": end_lidx})
    fp_send_p.start()
    fp_send_proc_list.append(fp_send_p)

    fp_recv_p = mp.Process(target=fp_recv_proc, kwargs={"conv_wid":conv_wid, "conv_wn":conv_wn, "fc_wid": fc_wid, "fc_wn":fc_wn, "wid":wid, "wn": wn,  "pred_wid":pred_wid, "succ_wid":succ_wid, "comm_rank":fp_recv_rank, "world_sz":world_size,   "bs":bs, "subbs":subbs, "pd": pd, "input_shp":input_shp, "output_shp": output_shp,  "fp_head_list": fp_head_list, "shared_cnters":shared_cnter, "sta_lidx": sta_lidx, "end_lidx": end_lidx})
    fp_recv_p.start()
    fp_recv_proc_list.append(fp_recv_p)

    bp_send_p = mp.Process(target=bp_send_proc, kwargs={"conv_wid":conv_wid, "conv_wn":conv_wn, "fc_wid": fc_wid, "fc_wn":fc_wn, "wid":wid, "wn": wn,  "pred_wid":pred_wid, "succ_wid":succ_wid, "comm_rank":bp_send_rank, "world_sz":world_size,  "bs":bs, "subbs":subbs, "pd": pd, "input_shp":input_shp, "output_shp": output_shp,  "bp_head_list": bp_head_list, "shared_cnters":shared_cnters,"sta_lidx": sta_lidx, "end_lidx": end_lidx})
    bp_send_p.start()
    bp_send_proc_list.append(bp_send_p)

    bp_recv_p = mp.Process(target=bp_recv_proc, kwargs={"conv_wid":conv_wid, "conv_wn":conv_wn, "fc_wid": fc_wid, "fc_wn":fc_wn, "wid":wid, "wn": wn,  "pred_wid":pred_wid, "succ_wid":succ_wid, "comm_rank":bp_recv_rank, "world_sz":world_size,  "bs":bs, "subbs":subbs, "pd": pd, "input_shp":input_shp, "output_shp": output_shp,  "bp_tail_list": bp_tail_list, "shared_cnters":shared_cnters, "sta_lidx": sta_lidx, "end_lidx": end_lidx})
    bp_recv_p.start()
    bp_recv_proc_list.append(bp_recv_p)        

    train_p = mp.Process(target=train_proc, kwargs={"conv_wid":conv_wid, "conv_wn":conv_wn, "fc_wid": fc_wid, "fc_wn":fc_wn, "wid":wid,  "wn": wn,  "bs":bs, "subbs":subbs, "pd": pd, "input_shp":input_shp, "output_shp": output_shp, "sub_net": sub_net, "sync_lock":sync_lock, "fp_head_list":fp_head_list, "fp_tail_list": fp_tail_list, "bp_head_list":bp_head_list, "bp_tail_list":bp_tail_list, "shared_cnters":shared_cnters,"train_step":train_step,  "global_step": global_step, "sta_lidx": sta_lidx, "end_lidx": end_lidx})
    train_p.start()
    train_proc_list.append(train_p)

    if wid == 0 or wid == 1:
        sync_rank = wn * 4 + wid

        sync_p = mp.Process(target=sync_proc, kwargs={"conv_wid":conv_wid, "conv_wn":conv_wn, "fc_wid": fc_wid, "fc_wn":fc_wn, "wid":wid, "wn": wn, "comm_rank":sync_rank, "world_sz":world_size,  "bs":bs, "subbs":subbs, "pd": pd, "sub_net": sub_net,  "train_step":train_step, "global_step": global_step})
        sync_p.start()
        sync_p.join()


