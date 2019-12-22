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
parser.add_argument('--ip', default="12.12.11.11", type=str, help='Master IP Address')
parser.add_argument('--prt', default="21331", type=str, help='Master Port')
parser.add_argument('--replica', default="1", type=int, help='Master Port')
parser.add_argument('--partition', default=[0,6,1, 6,12,1, 12,-1,3, 6,12,2,0,6,2], nargs='+', type=int)
args = parser.parse_args()

TOKEN_LAYERS = 5
TOKEN_CAPACITY = args.replica * 32
HOLD_MAP = torch.zeros([TOKEN_LAYERS,TOKEN_CAPACITY], dtype=torch.int32)
TOKEN_WEIGHT = [1,4,1,4,1]
TOKEN_NUMBER = [ int(TOKEN_CAPACITY/val) for val in TOKEN_WEIGHT]

WK_BASE = 0
TS_BASE = args.wn
WC_BASE = TS_BASE + args.wn
TC_BASE = WC_BASE + args.wn
SY_BASE = TC_BASE + args.wn
WORLD_SIZE = 5 * args.wn

criterion = nn.CrossEntropyLoss()
fake_input = torch.randn([args.subbs,3,224,224], dtype=torch.float)
fake_target = torch.from_numpy(np.random.randint(0,999,size=int(args.subbs*TOKEN_WEIGHT[2])))
fake_input = fake_input.share_memory_()
fake_target = fake_target.share_memory_()
MODEL_PARTITION = args.partition
CHUNK_WIDTH = args.subbs
OP_CODES = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
OP_CODES = OP_CODES.share_memory_()
TOKEN_DATA_STORAGE = [None for j in range(TOKEN_LAYERS)]
INPUT_PLACEHOLDERS = [None for i in range(TOKEN_LAYERS)]
INPUT_SIZE = [None for i in range(TOKEN_LAYERS)]
OUTPUT_SIZE = [None for i in range(TOKEN_LAYERS)]
SUB_MODEL_LIST = [None for i in range(TOKEN_LAYERS)]
SUB_OPTIMIZERS = [None for i in range(TOKEN_LAYERS)]


NEED_SYNC = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
NEED_SYNC = NEED_SYNC.share_memory_()
CAN_SYNC = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
CAN_SYNC = NEED_SYNC.share_memory_()

W2TS_MSG_SIZE = 1+2
TS2W_MSG_SIZE = 1+5
NEW_REQUEST = 0
REPORT_PROGRESS = 1
SYNC_FIN = 4
DISTRIBUTE_TOKEN = 2
SYNC_CMD = 3
NO_AVAILABLE = 5

def ini_data_storage():
    global TOKEN_DATA_STORAGE, INPUT_PLACEHOLDERS, INPUT_SIZE, OUTPUT_SIZE, SUB_MODEL_LIST, fake_target,criterion,fake_input, OP_CODES
    f= open("./alexnet_info.dump", "rb")
    profile_list = pickle.load(f)
    f.close()
    #profile and initialize the data memory of possible boundary data 
    for i in range(TOKEN_LAYERS):
        sta_lidx = MODEL_PARTITION[i*3]
        end_lidx = MODEL_PARTITION[i*3 +1]
        #for fc part, end_lids = -1
        if i < 3:
            SUB_MODEL_LIST[i] = myAlexNet("myAlexNet", sta_lidx = sta_lidx, end_lidx = end_lidx)
            SUB_MODEL_LIST[i].to("cuda")
            SUB_MODEL_LIST[i].share_memory()
            SUB_OPTIMIZERS[i] = optim.SGD(SUB_MODEL_LIST[i].parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        else:
            SUB_MODEL_LIST[i] = SUB_MODEL_LIST[TOKEN_LAYERS-1-i]
            SUB_OPTIMIZERS[i] = SUB_OPTIMIZERS[TOKEN_LAYERS-1-i]

        shp = profile_list[sta_lidx]["shape"]
        shp[0] = args.subbs * TOKEN_WEIGHT[i]
        ed_shp = profile_list[end_lidx]["shape"]
        ed_shp[0] = args.subbs * TOKEN_WEIGHT[i]
        OP_CODES[i] = MODEL_PARTITION[i*3+2]

        if OP_CODES[i] == 1:
            #FP
            INPUT_SIZE[i] = shp
            OUTPUT_SIZE[i] = ed_shp
        elif OP_CODES[i] == 2:
            #BP need warm-up
            if i == 3:
                print("output_data shp =", shp)
            output_data = torch.randn(shp, dtype = torch.float)
            output_data.requires_grad = True
            output_data = output_data.cuda()
            INPUT_PLACEHOLDERS[i] = SUB_MODEL_LIST[i](output_data)

            INPUT_SIZE[i] = INPUT_PLACEHOLDERS[i].size()
            output_data = output_data.cpu()
        elif OP_CODES[i] == 3: 
            #FP+BP, FC layers
            #初始必有forward，不用warm up
            INPUT_SIZE[i] = shp 
            OUTPUT_SIZE[i] = shp 
            print("opcode ==3  ",shp)

    for i in range(TOKEN_LAYERS):
        print("dddd ", INPUT_SIZE[i])
    torch.cuda.empty_cache()
    
    for i in range(TOKEN_LAYERS - 1):
        total_shp =[]
        for sz in INPUT_SIZE[i+1]:
            total_shp.append(sz)
        total_shp[0] = TOKEN_CAPACITY * CHUNK_WIDTH
        TOKEN_DATA_STORAGE[i] = torch.zeros(total_shp, dtype=torch.float)
        TOKEN_DATA_STORAGE[i] = TOKEN_DATA_STORAGE[i].share_memory_()
        print("i=",int(i),"\t","total_shp=",total_shp)
    for i in range(TOKEN_LAYERS):
        print("i=",i," input=",INPUT_SIZE[i], " output=",OUTPUT_SIZE[i])


def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("GLOO: ", os.environ['MASTER_ADDR'], " ",  os.environ['MASTER_PORT'])
    dist.init_process_group(backend, rank=rank, world_size=size)
    train_sync_ranks = [SY_BASE]*(args.wn)
    train_sync_ranks_fc = [] #TODO
    for i in range(args.wn):
        train_sync_ranks[i] += i
    train_sync_group = dist.new_group(ranks=train_sync_ranks, backend=backend)
    train_sync_fc_group = None
    return train_sync_group, train_sync_fc_group

def get_fc_rank(token_no):
    return args.wn-1+WK_BASE
def get_conv_rank(token_no):
    return token_no % args.wn + WK_BASE
def is_fc_worker(wid):
    if wid == args.wn-1:
        return True
    else:
        return False
def is_fc_depth(depth):
    if depth == 2:
        return True
    else:
        return False

def train_model(depth, token_no, input_data):
    if is_fc_depth(depth) and (not is_fc_worker(args.wid)):
        dist.send(tensor=input_data, dst = get_fc_rank(token_no))
        dist.recv(tensor = input_data, src = get_fc_rank(token_no))
        #print("CONV:", "\t", int(token_no))
        return
    elif is_fc_depth(depth) and is_fc_worker(args.wid):
        conv_rank = get_conv_rank(token_no)
        if conv_rank == args.wid + WK_BASE:
            pass
        else:
            dist.recv(tensor = input_data, src = conv_rank)
            #print("FC: recv:",int(conv_rank), "\t", int(token_no))
        input_data.requires_grad = True
        #print("opcode=3 ", input_data.requires_grad,"\t", input_data.size())
        input_data = input_data.cuda()
        fin_output = SUB_MODEL_LIST[depth](input_data)
        print("fin_output:",fin_output.size(),"\t", "fake_target size:",fake_target.size())
        loss = criterion(fin_output, fake_target.cuda())
        loss.backward()
        output_data = HookFunc.backward_ctx
        output_data = output_data.cpu()
        if conv_rank == args.wid + WK_BASE:
            pass
        else:
            dist.send(tensor=output_data, dst= conv_rank)
    else:
        output_data = None
        if OP_CODES[depth] == 1:
            #FP
            input_data = input_data.cuda()
            output_data = SUB_MODEL_LIST[depth](input_data)
            output_data = output_data.cpu()
            #print("train FP FIn output_sz = ", OUTPUT_PLACEHOLDERS[my_workload_no].size())
        elif OP_CODES[depth] == 2:
            #BP
            #INPUT_PLACEHOLDERS[depth].data.copy_(input_data)
            #print("opcode=2 ", INPUT_PLACEHOLDERS[my_workload_no].requires_grad)
            INPUT_PLACEHOLDERS[depth] = INPUT_PLACEHOLDERS[depth].cuda()
            input_data = input_data.cuda()
            INPUT_PLACEHOLDERS[depth].backward(input_data, retain_graph=True)
            output_data = HookFunc.backward_ctx
            output_data = output_data.cpu()
        elif OP_CODES[depth] == 3:
            #FP+BP
            #print("FP+BP: my_workload_no=",int(my_workload_no))
            input_data.requires_grad = True
            #print("opcode=3 ", input_data.requires_grad,"\t", input_data.size())
            input_data = input_data.cuda()
            fin_output = SUB_MODEL_LIST[depth](input_data)
            loss = criterion(fin_output, fake_target.cuda())
            loss.backward()
            output_data = HookFunc.backward_ctx
            output_data = output_data.cpu()
            dist.send(tensor=input_data, src=get_conv_rank(token_no))
            #print("FIN FP+BP---  ", type(HookFunc.backward_ctx))

    unit_size = TOKEN_WEIGHT[depth]*CHUNK_WIDTH
    base_offset = token_no * unit_size
    if depth < TOKEN_LAYERS-1:
        #the output of the last layer does not neded to be stored
        data_storage_tensor  = TOKEN_DATA_STORAGE[depth][base_offset:(base_offset+unit_size)]
        data_storage_tensor.copy_(output_data.data.cpu())

def get_input_data(depth, token_no):
    global fake_input
    if depth == 0:
        return fake_input
    else:
        unit_size = TOKEN_WEIGHT[depth] * CHUNK_WIDTH
        sta = token_no * unit_size
        return TOKEN_DATA_STORAGE[depth-1][sta:(sta+unit_size)]



def train_sync_proc(wid):
    my_rank = wid + WK_BASE
    dst_rank = wid + TS_BASE
    print("Start train_sync_proc init process rank=",my_rank)
    train_sync_group, train_sync_fc_group = init_processes(my_rank, WORLD_SIZE, "gloo")
    print("Started train_sync_proc init process rank=",my_rank)

    worker2ts_tensor = torch.zeros(W2TS_MSG_SIZE, dtype = torch.int32)
    ts2worker_tensor = torch.zeros(TS2W_MSG_SIZE, dtype = torch.int32)
    worker2ts_tensor[0] = NEW_REQUEST
    iter_cnt = 0
    time_list = []
    CAN_SYNC.zero_()
    NEED_SYNC.zero_().add_(1)
    NEED_SYNC[0] = 0
    NEED_SYNC[1] = 0
    print("NEED:",NEED_SYNC)
    print("CAN:",CAN_SYNC)
    dist.send(tensor = worker2ts_tensor, dst = dst_rank)
    while True:
        #print("RECV...")
        dist.recv(tensor = ts2worker_tensor, src = dst_rank)
        #print("RECVED ..", ts2worker_tensor)
        if ts2worker_tensor[0]== DISTRIBUTE_TOKEN:
            depth = ts2worker_tensor[1]
            token_no = ts2worker_tensor[2]
            input_data = get_input_data(depth, token_no)
            train_model(depth, token_no, input_data)
            worker2ts_tensor[0] = REPORT_PROGRESS
            worker2ts_tensor[1] = depth
            worker2ts_tensor[2] = token_no
            dist.send(tensor = worker2ts_tensor, dst = dst_rank)
            worker2ts_tensor[0] = NEW_REQUEST
            dist.send(tensor = worker2ts_tensor, dst = dst_rank)
        elif ts2worker_tensor[0]== SYNC_CMD:
            #print("SYNC_CMD...")
        
            while NEED_SYNC.sum() > 0:
                continue
            #reset
            CAN_SYNC.zero_()
            NEED_SYNC.zero_().add_(1)
            #print("NEED:",NEED_SYNC)
            #print("CAN:",CAN_SYNC)

            worker2ts_tensor[0] = SYNC_FIN
            dist.send(tensor = worker2ts_tensor, dst = dst_rank)
            worker2ts_tensor[0] = NEW_REQUEST
            dist.send(tensor = worker2ts_tensor, dst = dst_rank)
        elif ts2worker_tensor[0]== NO_AVAILABLE:
            for j in range(TOKEN_LAYERS):
                if ts2worker_tensor[j+1] == TOKEN_CAPACITY:
                    CAN_SYNC[j] = 1
            #print("complete_num ",ts2worker_tensor)
            worker2ts_tensor[0] = NEW_REQUEST
            dist.send(tensor = worker2ts_tensor, dst = dst_rank)


def model_sync_process(wid):
    my_rank = wid + SY_BASE
    print("Start model_sync_process init process rank=",my_rank)
    train_sync_group, train_sync_fc_group = init_processes(my_rank, WORLD_SIZE, "gloo")
    print("Started model_sync_process init process rank=",my_rank)
    while True:
        for i in range(TOKEN_LAYERS):
            if i == 0 or i == 1:
                NEED_SYNC[i] = 0
                continue
            if NEED_SYNC[i] == 1 and CAN_SYNC[i] == 1:
                #print("sync i=",int(i), "\t", NEED_SYNC)
                if is_fc_worker(wid) and is_fc_depth(i):
                    SUB_OPTIMIZERS[i].step()
                else:
                    req_list = []
                    for name, parameters in SUB_MODEL_LIST[i].named_parameters():
                        if(parameters.grad is not None):
                            grad_content = parameters.grad
                            grad_content.div_(args.wn)
                            grad_content = parameters.grad.cpu()
                            req = dist.all_reduce(grad_content, op=dist.ReduceOp.SUM, group=train_sync_group, async_op = True)
                            req_list.append(req)
                            parameters.grad.copy_(grad_content)

                    for req in req_list:
                        req.wait()
                    SUB_OPTIMIZERS[i].step()
                NEED_SYNC[i] = 0
            


def coordinate_proc(wid):
    my_rank = wid + WC_BASE
    src_rank = wid + TC_BASE
    print("Start coordinate_proc init process rank = ", my_rank)
    init_processes(my_rank, WORLD_SIZE, "gloo")
    print("coordinate_proc init process rank = ", my_rank)
    while True:
        time.sleep(1)

if __name__ == '__main__':
    ini_data_storage()
    
    wid = args.wid
    print("wid = ", wid)
    
    c_p = mp.Process(target=coordinate_proc, kwargs={"wid":wid})
    c_p.start()

    ms_p =  mp.Process(target=model_sync_process, kwargs={"wid":wid})
    ms_p.start()
    train_sync_proc(wid)

    ms_p.join()
    c_p.join()

    '''
    alex_test = myAlexNet("myalexnet",sta_lidx = -1, end_lidx = -1)
    alex_test.to("cuda")
    fin_output= alex_test(fake_input.cuda())
    print("output sz:",fin_output.size(),"\t", "target sz:",fake_target.size())
    print(fake_target)
    loss = criterion(fin_output, fake_target.cuda())
    loss.backward()
    exit(0)
    '''