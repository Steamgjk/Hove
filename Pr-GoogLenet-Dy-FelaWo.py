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
import random
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
parser.add_argument('--replica', default="1", type=int, help='replica number')
parser.add_argument('--partition', default=[0,4,1, 4,9,1, 9,-1,3, 4,9,2,0,4,2], nargs='+', type=int)
parser.add_argument('--tokencap', default="32", type=int, help='token capacity')
parser.add_argument('--weight', default=[1,4,4,4,1], nargs='+', type=int)
parser.add_argument('--sleepn', default=1, type=int, help='sleep time')
parser.add_argument('--prb', default=10, type=float, help='probability')
args = parser.parse_args()

TOKEN_LAYERS = 5
TOKEN_CAPACITY = args.replica * args.tokencap
CHUNK_HOLD_MAP = torch.zeros([TOKEN_LAYERS,TOKEN_CAPACITY], dtype=torch.int32)
CHUNK_HOLD_MAP = CHUNK_HOLD_MAP.share_memory_()

TOKEN_WEIGHT = args.weight
TOKEN_NUMBER = [ int(TOKEN_CAPACITY/val) for val in TOKEN_WEIGHT]

WK_BASE = 0
TS_BASE = args.wn
WCR_BASE = TS_BASE + args.wn
WCS_BASE = WCR_BASE + args.wn
TC_BASE = WCS_BASE + args.wn
SY_BASE = TC_BASE + args.wn
TSY_BASE = SY_BASE + args.wn
WORLD_SIZE = 7 * args.wn

criterion = nn.CrossEntropyLoss()
fake_input = torch.randn([args.subbs * TOKEN_WEIGHT[0],3,32,32], dtype=torch.float)
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
CAN_SYNC = CAN_SYNC.share_memory_()
GLOBAL_STEP = torch.zeros(1, dtype = torch.int32)
GLOBAL_STEP = GLOBAL_STEP.share_memory_()

W2TS_MSG_SIZE = 1+0
TS2W_MSG_SIZE = 1+2
TS2C_MSG_SIZE = 1+3
TS2S_MSG_SIZE = 1+1
S2TS_MSG_SIZE = 1+1

NEW_REQUEST = 0
REPORT_PROGRESS = 1
SYNC_FIN = 4
DISTRIBUTE_TOKEN = 2
SYNC_CMD = 3
NO_AVAILABLE = 5
OTHER_TOKENS = 6
CHUNK_REQUEST = 7
CHUNK_RESPONSE = 8
SYNC_REQUEST = 9
SYNC_RESPONSE = 10
CONN_ESTABLISH = 11
CONNECTION_RST = 12
CONNECTION_REQUEST = 13
RST_FIN = 14

def ini_data_storage():
    global TOKEN_DATA_STORAGE, INPUT_PLACEHOLDERS, INPUT_SIZE, OUTPUT_SIZE, SUB_MODEL_LIST, fake_target,criterion,fake_input, OP_CODES
    f= open("./googlenet_info.dump", "rb")
    profile_list = pickle.load(f)
    f.close()
    #profile and initialize the data memory of possible boundary data 
    for i in range(TOKEN_LAYERS):
        sta_lidx = MODEL_PARTITION[i*3]
        end_lidx = MODEL_PARTITION[i*3 +1]
        print("sta_lidx=",sta_lidx, " end_lidx=",end_lidx)
        #for fc part, end_lids = -1
        if i < 3:
            SUB_MODEL_LIST[i] = GoogLeNet(sta_lidx = sta_lidx, ed_lidx = end_lidx)
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
     
    torch.cuda.empty_cache()
    
    for i in range(TOKEN_LAYERS - 1):
        total_shp =[]
        for sz in INPUT_SIZE[i+1]:
            total_shp.append(sz)
        total_shp[0] = TOKEN_CAPACITY * CHUNK_WIDTH
        TOKEN_DATA_STORAGE[i] = torch.zeros(total_shp, dtype=torch.float)
        TOKEN_DATA_STORAGE[i] = TOKEN_DATA_STORAGE[i].share_memory_()
    for i in range(TOKEN_LAYERS):
        print("i=",i," input=",INPUT_SIZE[i], " output=",OUTPUT_SIZE[i])

    CHUNK_HOLD_MAP.zero_()

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
    output_data = None
    if OP_CODES[depth] == 1:
        #FP
        with torch.no_grad():
            input_data = input_data.cuda()
            output_data = SUB_MODEL_LIST[depth](input_data)
            output_data = output_data.cpu()
        #print("train FP FIn output_sz = ", OUTPUT_PLACEHOLDERS[my_workload_no].size())
    elif OP_CODES[depth] == 2:
        #BP
        bp_data = get_bp_input_data(depth, token_no)
        INPUT_PLACEHOLDERS[depth].data.copy_(bp_data)
        INPUT_PLACEHOLDERS[depth] = INPUT_PLACEHOLDERS[depth].cuda()
        input_data = input_data.cuda()
        INPUT_PLACEHOLDERS[depth].backward(input_data, retain_graph=True)
        output_data = GoogleNetHookFunc.backward_ctx
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
        output_data = GoogleNetHookFunc.backward_ctx
        output_data = output_data.cpu()
        #print("FIN FP+BP---  ", type(HookFunc.backward_ctx))

    unit_size = TOKEN_WEIGHT[depth]*CHUNK_WIDTH
    base_offset = token_no * unit_size
    chunk_offset = token_no * TOKEN_WEIGHT[depth]
    if depth < TOKEN_LAYERS-1:
        #the output of the last layer does not neded to be stored
        data_storage_tensor  = TOKEN_DATA_STORAGE[depth][base_offset:(base_offset+unit_size)]
        data_storage_tensor.copy_(output_data.data.cpu())
        CHUNK_HOLD_MAP[depth][chunk_offset:(chunk_offset+TOKEN_WEIGHT[depth])] = 1

def train_model_bk(depth, token_no, input_data):
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
        loss = criterion(fin_output, fake_target.cuda())
        loss.backward()
        output_data = GoogleNetHookFunc.backward_ctx
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
            #add bp_data
            bp_data = get_bp_input_data(depth, token_no)
            #INPUT_PLACEHOLDERS[depth].data.copy_(input_data)
            INPUT_PLACEHOLDERS[depth].data.copy_(bp_data)
            #print("opcode=2 ", INPUT_PLACEHOLDERS[my_workload_no].requires_grad)
            #????? have some problem
            INPUT_PLACEHOLDERS[depth] = INPUT_PLACEHOLDERS[depth].cuda()
            input_data = input_data.cuda()
            INPUT_PLACEHOLDERS[depth].backward(input_data, retain_graph=True)
            output_data = GoogleNetHookFunc.backward_ctx
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
            output_data = GoogleNetHookFunc.backward_ctx
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
        chunk_offset = token_no * TOKEN_WEIGHT[depth]
        while CHUNK_HOLD_MAP[depth-1][chunk_offset:(chunk_offset+TOKEN_WEIGHT[depth])].sum() < TOKEN_WEIGHT[depth]:
            continue
        unit_size = TOKEN_WEIGHT[depth] * CHUNK_WIDTH
        sta = token_no * unit_size
        return TOKEN_DATA_STORAGE[depth-1][sta:(sta+unit_size)]

def get_bp_input_data(depth, token_no):
    fp_depth = TOKEN_LAYERS - 1 - depth
    unit_size = TOKEN_WEIGHT[depth] * CHUNK_WIDTH    
    sta = token_no * unit_size
    chunk_offset = token_no * TOKEN_WEIGHT[depth]
    while CHUNK_HOLD_MAP[fp_depth][chunk_offset:(chunk_offset+TOKEN_WEIGHT[depth])].sum() < TOKEN_WEIGHT[depth]:
        continue
    return TOKEN_DATA_STORAGE[fp_depth][sta:(sta+unit_size)]



def is_bp_depth(depth):
    if depth == 3 or depth ==4:
        return True
    else:
        return False
def check_dependency(my_depth, my_token_no):
    if my_depth ==0:
        return True
    sta = my_token_no * TOKEN_WEIGHT[my_depth]
    pre_depth = my_depth -1
    for i in range(sta, sta+TOKEN_WEIGHT[my_depth]):
        if CHUNK_HOLD_MAP[pre_depth][i] == 0:
            #print("fp no :", int(my_depth),"\t",int(my_token_no),"\t", int(pre_depth),"\t", int(i) )
            return False
    if is_bp_depth(my_depth):
        fp_depth = TOKEN_LAYERS-1 - my_depth
        for i in range(sta, sta+TOKEN_WEIGHT[my_depth]):
            if CHUNK_HOLD_MAP[fp_depth][i] == 0:
                #print("bp no :", int(my_depth),"\t",int(my_token_no),"\t", int(fp_depth),"\t", int(i) )
                return False
    return True       


def train_sync_proc(wid):
    my_rank = wid + WK_BASE
    dst_rank = wid + TS_BASE
    print("Start train_sync_proc init process rank=",my_rank)
    train_sync_group, train_sync_fc_group = init_processes(my_rank, WORLD_SIZE, "gloo")
    print("Started train_sync_proc init process rank=",my_rank)

    connection_request_tensor = torch.zeros(W2TS_MSG_SIZE, dtype = torch.int32)
    new_request_tensor = torch.zeros(W2TS_MSG_SIZE, dtype = torch.int32)
    report_progress_tensor = torch.zeros(W2TS_MSG_SIZE, dtype = torch.int32)
    ts2worker_tensor = torch.zeros(TS2W_MSG_SIZE, dtype = torch.int32)
    rc2wc_tensor = torch.zeros(TS2C_MSG_SIZE, dtype = torch.int32)
    connection_request_tensor[0] = CONNECTION_REQUEST
    new_request_tensor[0] = NEW_REQUEST
    report_progress_tensor[0] = REPORT_PROGRESS
    local_step = 0

    dist.send(tensor = connection_request_tensor, dst = dst_rank)
    while True:
        #print("RECV...")
        dist.recv(tensor = ts2worker_tensor, src = dst_rank)
        #print("RECVED ..", ts2worker_tensor)
        if ts2worker_tensor[0]== CONN_ESTABLISH:
            local_step = int(ts2worker_tensor[1])
            print("local_step=",local_step)
            
            if random.randint(0,99) < args.prb:
                print("I need sleep {:d} s".format(args.sleepn))
                if args.sleepn > 0:
                    time.sleep(args.sleepn)
            
            dist.send(tensor = new_request_tensor, dst = dst_rank)

        elif ts2worker_tensor[0]== DISTRIBUTE_TOKEN:
            depth = ts2worker_tensor[1]
            token_no = ts2worker_tensor[2]
            input_data = get_input_data(depth, token_no)
            #print("training self... ", int(depth),"\t", int(token_no))
            train_model(depth, token_no, input_data)
            #report_progress_tensor[1] = depth
            #report_progress_tensor[2] = token_no
            dist.send(tensor = report_progress_tensor, dst = dst_rank)
            dist.send(tensor = new_request_tensor, dst = dst_rank)
        elif ts2worker_tensor[0]== OTHER_TOKENS:
            #need depdencies

            depth = ts2worker_tensor[1]
            token_no = ts2worker_tensor[2]
            #print("other token... ", int(depth), " ", int(token_no))
            #print("get other tokens ", int(depth), "\t", int(token_no))
            while check_dependency(depth, token_no) == False:
                #print("checking dependency false ", int(depth),"\t",int(token_no))
                #time.sleep(1)
                continue
            input_data = get_input_data(depth, token_no)
            #print("training others... ", int(depth),"\t", int(token_no))
            train_model(depth, token_no, input_data)
            #report_progress_tensor[1] = depth
            #report_progress_tensor[2] = token_no
            #print("train fin sending")
            dist.send(tensor = report_progress_tensor, dst = dst_rank)
            dist.send(tensor = new_request_tensor, dst = dst_rank) 
            #print("asking new request")
        elif ts2worker_tensor[0]== NO_AVAILABLE:
            dist.send(tensor = new_request_tensor, dst = dst_rank)
        elif ts2worker_tensor[0] == CONNECTION_RST:
            dist.send(tensor = connection_request_tensor, dst = dst_rank)



def model_sync_process(wid):
    my_rank = wid + SY_BASE
    print("Start model_sync_process init process rank=",my_rank)
    train_sync_group, train_sync_fc_group = init_processes(my_rank, WORLD_SIZE, "gloo")
    print("Started model_sync_process init process rank=",my_rank)
    ts2ms_rank = wid + TSY_BASE
    ts2ms_tensor = torch.zeros(TS2S_MSG_SIZE, dtype=torch.int32)
    ms2ts_tensor = torch.zeros(S2TS_MSG_SIZE, dtype=torch.int32)
    ms2ts_tensor[0] = SYNC_RESPONSE
    while True:
        #for recv
        #print("model recving...", wid)
        dist.recv(tensor=ts2ms_tensor, src=ts2ms_rank)
        to_sync_layer = ts2ms_tensor[1]
        #print("to_sync_layer=",int(to_sync_layer))
        req_list = []
        for name, parameters in SUB_MODEL_LIST[to_sync_layer].named_parameters():
            if(parameters.grad is not None):
                grad_content = parameters.grad
                grad_content.div_(args.wn)
                grad_content = parameters.grad.cpu()
                req = dist.all_reduce(grad_content, op=dist.ReduceOp.SUM, group=train_sync_group, async_op = True)
                req_list.append(req)
                parameters.grad.copy_(grad_content)

        for req in req_list:
            req.wait()
        SUB_OPTIMIZERS[to_sync_layer].step()
        ms2ts_tensor[1] = to_sync_layer
        dist.send(tensor=ms2ts_tensor, dst = ts2ms_rank)
        #print("sync fin ", int(to_sync_layer))

        if to_sync_layer == TOKEN_LAYERS - 1:
            CHUNK_HOLD_MAP.zero_()
            

#depth|chunk_no|sender/receiver
def coordinate_proc_request(wid):
    my_rank = wid + WCR_BASE
    src_rank = wid + TC_BASE
    print("Start coordinate_proc init process rank = ", my_rank)
    init_processes(my_rank, WORLD_SIZE, "gloo")
    print("coordinate_proc init process rank = ", my_rank)
    ts2wc_tensor = torch.zeros(TS2C_MSG_SIZE, dtype=torch.int32)
    while True:
        dist.recv(tensor = ts2wc_tensor, src = src_rank)
        #print("WC recved..", ts2wc_tensor)
        if ts2wc_tensor[0] == CHUNK_REQUEST:
            request_sender_wid = ts2wc_tensor[1]
            request_depth = ts2wc_tensor[2]
            request_chunk_no = ts2wc_tensor[3]
            sta = request_chunk_no*CHUNK_WIDTH
            recv_tensor = TOKEN_DATA_STORAGE[request_depth][sta:(sta+CHUNK_WIDTH)]
            #print("request wid=",int(wid),"request_depth=",int(request_depth),"\twho gives me=",int(request_sender_wid), "\tchunk_no=",int(request_chunk_no))
            dist.recv(tensor = recv_tensor, src = request_sender_wid + WCS_BASE)
            CHUNK_HOLD_MAP[request_depth][request_chunk_no] = 1
            #print("fin request wid=",int(wid),"request_depth=",int(request_depth),"\twho gives me=",int(request_sender_wid), "\tchunk_no=",int(request_chunk_no))

def coordinate_proc_response(wid):
    my_rank = wid + WCS_BASE
    src_rank = wid + TC_BASE
    print("Start coordinate_proc init process rank = ", my_rank)
    init_processes(my_rank, WORLD_SIZE, "gloo")
    print("coordinate_proc init process rank = ", my_rank)
    ts2wc_tensor = torch.zeros(TS2C_MSG_SIZE, dtype=torch.int32)
    while True:
        dist.recv(tensor = ts2wc_tensor, src = src_rank)
        #print("WC recved..", ts2wc_tensor)
        if ts2wc_tensor[0] == CHUNK_RESPONSE:
            requester_wid = ts2wc_tensor[1]
            request_depth = ts2wc_tensor[2]
            request_chunk_no = ts2wc_tensor[3]
            sta = request_chunk_no*CHUNK_WIDTH
            chunk_tensor = TOKEN_DATA_STORAGE[request_depth][sta:(sta+CHUNK_WIDTH)]
            #print("response wid=",int(wid),"\twho need it=",int(requester_wid), "\tchunk_no=",int(request_chunk_no))
            dist.send(tensor = chunk_tensor, dst = requester_wid + WCR_BASE)
            #print("fin response wid=",int(wid),"\twho need it=",int(requester_wid), "\tchunk_no=",int(request_chunk_no))

if __name__ == '__main__':
    ini_data_storage()
    wid = args.wid
    print("wid = ", wid)
    
    c_p_1 = mp.Process(target=coordinate_proc_request, kwargs={"wid":wid})
    c_p_1.start()
    c_p_2 = mp.Process(target=coordinate_proc_response, kwargs={"wid":wid})
    c_p_2.start()
    ms_p =  mp.Process(target=model_sync_process, kwargs={"wid":wid})
    ms_p.start()
    train_sync_proc(wid)

    ms_p.join()
    c_p.join()