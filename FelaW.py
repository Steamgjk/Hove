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
parser.add_argument('--partition', default=[0,26,1, 26,53,1, 53,-1,3, 26,53,2,0,26,2], nargs='+', type=int)
args = parser.parse_args()

global TOKEN_DATA_STORAGE, TOKEN_IPT_STORAGE, INPUT_PLACEHOLDERS, OUTPUT_PLACEHOLDERS, INPUT_CTX_PLACEHOLDERS, OUTPUT_CTX_PLACEHOLDERS, INPUT_SIZE, OUTPUT_SIZE, SUB_MODEL_LIST,CONSUMED_SAMPLES, fake_target,criterion,fake_input

#os.environ["CUDA_VISIBLE_DEVICES"]='1'

TOKEN_LAYERS = 5
TOKEN_CAPACITY = 32
TS2W_HEADER=4
WORKER_NUM = args.wn
FC_WORKER_NUM = args.fcwn
TOKEN_WEIGHT = [1,8,32,8,1]
DOWN_TENSOT_SIZE =  TOKEN_CAPACITY * 2 + TS2W_HEADER
UP_TENSOR_SIZE = 3
REQUEST_TENSOR_SIZE = 3
WORLD_SIZE = args.wn * 4
TS_BASE = 0
RQ_BASE = args.wn * 3
TQ_BASE = args.wn 
CQ_BASE = args.wn * 2
MODEL_PARTITION = args.partition
CHUNK_WIDTH = args.subbs
OP_CODES = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
OP_CODES = OP_CODES.share_memory_()
READY_WORKLOAD = torch.zeros(2,dtype=torch.int32).add_(-1)
READY_WORKLOAD = READY_WORKLOAD.share_memory_()
COMPUTE_ITERS = torch.zeros(TOKEN_LAYERS,dtype=torch.int32)
COMPUTE_ITERS = COMPUTE_ITERS.share_memory_()
SYNC_ITERS = torch.zeros(TOKEN_LAYERS, dtype = torch.int32)
SYNC_ITERS = SYNC_ITERS.share_memory_()
NEED_SYNC = torch.zeros(TOKEN_LAYERS,dtype=torch.int32)
NEED_SYNC = NEED_SYNC.share_memory_()
'''
TS: wn<-> W: wn; TS: wn-> W: wn
TS；wn master_channel
Worker: 2 channel: worker_channel and coordination_channel
WC->MC  MC->WC MC->CC' CC'->WC
'''
'''
TOKEN_SLICES[i][j]: bitmap, Which tokens (bitmap) are needed to compute token_j in layer_i 
WORKER_DATA_INFO[i][j]: bitmap, As for workload_i, which tokens (bitmap) do the worker_j have undertaken
TOKEN_WORKER_INFO[i][j]: int32 As for token_j in workload_i, which worker has fetched it for training? if the value is 0, then no worker is holding it, then it can be fetched
TOKEN_CNTERS[i]: for workload_i, how many tokens have been generated (and possibly can be fetched)
CACHE_WEIGHT[i]: for workload_i, how many weights have been cached. if it reaches the weigth threshold, it can generate a new token for workload_{i+1}, and then the CACHE_WEIGHT[i] will be zeroed
FLAG_BITS[i]: to be abandoned
REQUEST_TENSORS[i][j][k]: to faciliate asynchronous send to coordiante thread. worker_i is requesting the workload_j for token_k, to another worker
SYNC_FLAG: bitmap, 1 represents that TS has sent sync signal to that worker, when SYNC_FLAG become all 1s, reset the env and start the next iteration

TOKEN_DATA_STORAGE[i][j]: to storage the output data after training the token_j in workload_i
BOUNDARY_CHUNK_NUM[i][j]: for TOKEN_DATA_STORAGE[i][j], how many chunks should it be chunked to be used for next stage?
BOUNDARY_DATA_CHUNKS[i][j]: the prt lists of TOKEN_DATA_STORAGE[i][j] chunks
'''

TOKEN_DATA_STORAGE = [None for j in range(TOKEN_LAYERS)]
CHUNK_DATA_PTRS = [None for j in range(TOKEN_LAYERS)]
INPUT_PLACEHOLDERS = [None for i in range(TOKEN_LAYERS)]
INPUT_SIZE = [None for i in range(TOKEN_LAYERS)]
OUTPUT_SIZE = [None for i in range(TOKEN_LAYERS)]
SUB_MODEL_LIST = [None for i in range(TOKEN_LAYERS)]
SUB_OPTIMIZERS = [None for i in range(TOKEN_LAYERS)]

CONSUMED_SAMPLES = 0
criterion = nn.CrossEntropyLoss()
fake_input = torch.randn([args.subbs,3,224,224], dtype=torch.float)
fake_target = torch.from_numpy(np.random.randint(0,999,size=int(args.subbs*TOKEN_WEIGHT[2])))
fake_input = fake_input.share_memory_()
fake_target = fake_target.share_memory_()


def ini_data_storage():
    global TOKEN_DATA_STORAGE, INPUT_PLACEHOLDERS, INPUT_SIZE, OUTPUT_SIZE, SUB_MODEL_LIST,fake_target,criterion,fake_input, OP_CODES
    f= open("./vgg_info.dump", "rb")
    profile_list = pickle.load(f)
    f.close()
    #profile and initialize the data memory of possible boundary data 
    for i in range(TOKEN_LAYERS):
        sta_lidx = MODEL_PARTITION[i*3]
        end_lidx = MODEL_PARTITION[i*3 +1]
        print("sta_lidx=",sta_lidx, " end_lidx=",end_lidx)
        #for fc part, end_lids = -1
        if i < 3:
            SUB_MODEL_LIST[i] = myVGG("VGG19", sta_lidx = sta_lidx, end_lidx = end_lidx)
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
        CHUNK_DATA_PTRS[i] = TOKEN_DATA_STORAGE[i].chunk(TOKEN_CAPACITY)
    for i in range(TOKEN_LAYERS):
        print("i=",i," input=",INPUT_SIZE[i], " output=",OUTPUT_SIZE[i])


def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = args.prt
    print("GLOO: ", os.environ['MASTER_ADDR'], " ",  os.environ['MASTER_PORT'])
    dist.init_process_group(backend, rank=rank, world_size=size)
    train_sync_ranks = [args.wn]*(args.wn)
    train_sync_ranks_fc = [] #TODO
    for i in range(args.wn):
        train_sync_ranks[i] += i
    train_sync_group = dist.new_group(ranks=train_sync_ranks, backend=backend)
    train_sync_fc_group = None
    return train_sync_group, train_sync_fc_group
def read_samples(mybs):
    global CONSUMED_SAMPLES
    CONSUMED_SAMPLES += mybs
    return fake_input

def train_model(input_data, input_bp_data, my_workload_no, my_token_idx):
    output_data = None
    #output_ctx = None
    if OP_CODES[my_workload_no] == 1:
        #FP
        input_data = input_data.cuda()
        output_data = SUB_MODEL_LIST[my_workload_no](input_data)
        output_data = output_data.cpu()
        #print("train FP FIn output_sz = ", OUTPUT_PLACEHOLDERS[my_workload_no].size())
    elif OP_CODES[my_workload_no] == 2:
        #BP
        INPUT_PLACEHOLDERS[my_workload_no].data.copy_(input_bp_data)
        #print("opcode=2 ", INPUT_PLACEHOLDERS[my_workload_no].requires_grad)
        INPUT_PLACEHOLDERS[my_workload_no] = INPUT_PLACEHOLDERS[my_workload_no].cuda()
        input_data = input_data.cuda()
        INPUT_PLACEHOLDERS[my_workload_no].backward(input_data, retain_graph=True)
        output_data = HookFunc.backward_ctx
        output_data = output_data.cpu()
        #output_ctx = HookFunc.backward_ctx
        #output_ctx = output_ctx.cpu()
    elif OP_CODES[my_workload_no] == 3:
        #FP+BP
        #print("FP+BP: my_workload_no=",int(my_workload_no))
        input_data.requires_grad = True
        #print("opcode=3 ", input_data.requires_grad,"\t", input_data.size())
        input_data = input_data.cuda()
        fin_output = SUB_MODEL_LIST[my_workload_no](input_data)
        loss = criterion(fin_output, fake_target.cuda())
        loss.backward()
        #output_ctx = HookFunc.backward_ctx
        #output_data = input_data
        #output_ctx = output_ctx.cpu()
        output_data = HookFunc.backward_ctx
        output_data = output_data.cpu()
        #print("FIN FP+BP---  ", type(HookFunc.backward_ctx))

    unit_size = TOKEN_WEIGHT[my_workload_no]*CHUNK_WIDTH
    base_offset = my_token_idx * unit_size

    if my_workload_no < TOKEN_LAYERS-1:
        #the output of the last layer does not neded to be stored
        data_storage_tensor  = TOKEN_DATA_STORAGE[my_workload_no][base_offset:(base_offset+unit_size)]
        data_storage_tensor.copy_(output_data.data.cpu())
    '''
    if output_ctx is not None:
        ctx_storage_tensor = TOKEN_CTX_STORAGE[my_workload_no][base_offset:(base_offset+unit_size)]
        ctx_storage_tensor.copy_(output_ctx.data.cpu())
    '''

def is_bp(workload_no):
    if workload_no == 3 or workload_no == 4:
        return True
    else:
        return False
def gen_input_data(needed_workload_no, my_workload_no, my_token_idx):
    unit_size =  TOKEN_WEIGHT[my_workload_no] * CHUNK_WIDTH
    row_idx =  my_token_idx * unit_size
    input_data = TOKEN_DATA_STORAGE[needed_workload_no][row_idx:(row_idx+unit_size)]
    return input_data
def gen_input_bp_data(fp_workload_no, fp_token_id):
    #print("gen_input_bp_data:",int(fp_workload_no),"\t", int(fp_token_id))
    #patch:
    if fp_token_id < 0:
        fp_token_id = 0
    if fp_workload_no < 0:
        return None
    else:
        unit_size = TOKEN_WEIGHT[fp_workload_no] * CHUNK_WIDTH
        row_idx = fp_token_id * unit_size
        bp_input_data = TOKEN_DATA_STORAGE[fp_workload_no][row_idx: (row_idx+unit_size)]
        #print("unit_size=",int(unit_size),"\trow_idx=",int(row_idx),"\tbp_input_data_sz=",bp_input_data.size())
        return bp_input_data
def has_ctx(workload_no):
    if workload_no == 0 or workload_no == 1:
        return False
    else:
        return True
def is_fc_model(workload_no):
    if workload_no == 2:
        return True
    else:
        return False
def is_fc():
    return False

def need_grad_sync(workload_no):
    if workload_no < 3:
        return False
    else:
        return True

def model_to_cuda():
    for i in range(TOKEN_LAYERS):
        SUB_MODEL_LIST[i].to("cuda")

#down_tensor: |workload_no|token_id|pre_chunk_num(int32)|pre_wid_0,pre_chunk_0,...
#request_tensor: needed_workload_no (int32)|needed_chunk_idx(int32)|request_wid

def train_sync_proc(wid):
    my_rank = wid + TQ_BASE
    print("Start train_sync_proc init process rank=",my_rank)
    train_sync_group, train_sync_fc_group = init_processes(my_rank, WORLD_SIZE, "gloo")
    print("train_sync_proc init process rank=",my_rank)

    up_tensor = torch.zeros(UP_TENSOR_SIZE, dtype = torch.int32)
    up_tensor[0] = -1 
    down_tensor = torch.zeros(DOWN_TENSOT_SIZE, dtype = torch.int32)
    iter_cnt = 0
    sta_time = 0
    ed_time = 0
    while True:
        #print("W2TS... up_tensor=", up_tensor)
        dist.send(tensor = up_tensor, dst = wid)
        dist.recv(tensor = down_tensor, src = wid)
        my_workload_no = down_tensor[0]
        my_token_idx = down_tensor[1]

        #print("R4TS... my_workload_no=",int(my_workload_no), "  my_token_idx=", int(my_token_idx))
        input_data = None
        #Recv from coordinate
        if my_workload_no == -2: 
            #this iteration has been completed sync and start next iteration
            #model_sync(train_sync_group, train_sync_fc_group)
            iter_cnt += 1
            print("iter_cnt=",iter_cnt)
            if iter_cnt ==2:
                sta_time = time.time()
            elif iter_cnt == 7:
                ed_time = time.time()
                print("iterat  time = ", float(ed_time - sta_time)/5)
            up_tensor[0] = -1
            continue
        elif my_workload_no == -1:
            #currently no available workload obtained, request again
            up_tensor[0] = -1 
            up_tensor[1] = -1
            up_tensor[2] = READY_WORKLOAD[1]
            continue
        else: 
            if my_workload_no > 0:
                pre_chunk_num = down_tensor[2]
                READY_WORKLOAD[0] = down_tensor[3]
                data_chunk_num = pre_chunk_num
                needed_workload_no = my_workload_no - 1
                recv_data_reqs = [None for row in range(data_chunk_num)]
                base_offset = TS2W_HEADER
                fp_workload_no = -1
                fp_token_id = -1
                data_recv_req_list = []
                bp_data_recv_req_list = []
                if is_bp(my_workload_no):
                    data_chunk_num = int(pre_chunk_num/2)
                    fp_workload_no = TOKEN_LAYERS - 1 - my_workload_no
                    fp_token_id = down_tensor[-1]

                for j in range(pre_chunk_num):
                    #The Token Server has eliminated those local requests
                    needed_wid = down_tensor[base_offset]
                    needed_chunk_id = down_tensor[base_offset+1]
                    
                    #if the needed wid is myself, no need to recv
                    if not (needed_wid == wid):
                        src_rank = needed_wid + CQ_BASE
                        #print("R4TS:",int(needed_wid),"->",int(wid),"\t",int(needed_workload_no),"\t",int(needed_chunk_id),"\t",int(j),"\t",int(pre_chunk_num),"\t",down_tensor)
                        if j < data_chunk_num:
                            req = dist.irecv(tensor= CHUNK_DATA_PTRS[needed_workload_no][needed_chunk_id], src = src_rank)
                            data_recv_req_list.append(req)
                        else:
                            #print("BP: j=",j,"\t","data_chunk_num=",data_chunk_num)
                            #bp input assist
                            req = dist.irecv(tensor= CHUNK_DATA_PTRS[fp_workload_no][needed_chunk_id], src = src_rank)
                            bp_data_recv_req_list.append(req)
                            #print("FIN BP: j=",j,"\t","data_chunk_num=",data_chunk_num)

                        #print("R4TS-FIN:",int(needed_wid),"->",int(wid),"\t",int(needed_workload_no),"\t",int(needed_chunk_id))


                    base_offset+=2
                for data_req in data_recv_req_list:
                    #print("waiting...")
                    data_req.wait()
                    #print("got")
                input_data = gen_input_data(needed_workload_no, my_workload_no, my_token_idx)
                if not(fp_workload_no < 0 ):
                    for bp_data_req in bp_data_recv_req_list:
                        #print("waiting2...")
                        bp_data_req.wait()
                        #print("got2")
                    input_bp_data = gen_input_bp_data(fp_workload_no, fp_token_id)
                else:
                    input_bp_data = None

            else:
                #print("come to read samples")
                input_data  = read_samples(CHUNK_WIDTH * TOKEN_WEIGHT[0])
                input_bp_data = None
            '''
            if (input_data is not None) and (input_bp_data is not None):
                print("Training... my_workload_no=",int(my_workload_no), "  my_token_idx=", int(my_token_idx), "input_sz =", input_data.size(), "input_bp_data_sz=", input_bp_data.size())
            '''
            #while my_workload_no < READY_WORKLOAD[1]:

            train_model(input_data, input_bp_data, my_workload_no, my_token_idx)
            #print("FIN Training... my_workload_no=",int(my_workload_no), "  my_token_idx=", int(my_token_idx))
            up_tensor[0] = my_workload_no
            up_tensor[1] = my_token_idx
            up_tensor[2] = READY_WORKLOAD[1]

def get_requested_data(needed_workload_no, needed_chunk_id):
    #need_ctx = has_ctx(needed_workload_no)
    base_offset = needed_chunk_id* CHUNK_WIDTH
    rq_data = TOKEN_DATA_STORAGE[needed_workload_no][base_offset:(base_offset+CHUNK_WIDTH)]
    return rq_data

def model_sync_process(wid):
    for i in range(TOKEN_LAYERS):
        if need_grad_sync(i):
            NEED_SYNC[i] = 1
        else:
            NEED_SYNC[i] = 0
    #NEED_SYNC[2] = 0
    READY_WORKLOAD[1] = 0
    req_list = []
    while True:
        #print("MODEL READY=",int(READY_WORKLOAD[0]))
        if READY_WORKLOAD[1] < READY_WORKLOAD[0] or READY_WORKLOAD[1] == READY_WORKLOAD[0]:
            ready_to_sync = int(READY_WORKLOAD[1]) % TOKEN_LAYERS
            if NEED_SYNC[ready_to_sync] == 1:
                for name, parameters in SUB_MODEL_LIST[ready_to_sync].named_parameters():
                    if(parameters.grad is not None):
                        grad_content = parameters.grad
                        grad_content.div_(WORKER_NUM)
                        grad_content = parameters.grad.cpu()
                        #print(int(args.wid),"\tallreduce:name = ", name,"\t",grad_content.numel() )
                        req = dist.all_reduce(grad_content, op=dist.ReduceOp.SUM, group=train_sync_group, async_op = True)
                        req_list.append(req)
                        parameters.grad.copy_(grad_content)
                for req in req_list:
                    req.wait()
                SUB_OPTIMIZERS[ready_to_sync].step()
                SUB_OPTIMIZERS[ready_to_sync].zero_grad()

                print("SYNC Complete:",int(ready_to_sync))
            READY_WORKLOAD[1] += 1
        '''
        else:
            print(int(READY_WORKLOAD[1])," vs ",int(READY_WORKLOAD[0]))
            time.sleep(1)
        '''




#send_tensor: |workload_no|token_id|pre_chunk_num(int32)|pre_wid_0,pre_chunk_0,...
#request_tensor: needed_workload_no (int32)|needed_chunk_idx(int32)|request_wid

def coordinate_proc(wid):
    my_rank = wid + CQ_BASE
    src_rank = wid + RQ_BASE
    print("Start coordinate_proc init process rank = ", my_rank)
    init_processes(my_rank, WORLD_SIZE, "gloo")
    print("coordinate_proc init process rank = ", my_rank)
    request_tensor = torch.zeros(REQUEST_TENSOR_SIZE, dtype=torch.int32)
    '''
    while(CAN_START[0] ==0):
        time.sleep(1)
        print("CP sleep...")
    '''
    while True:
        #print("CRTS-:", int(src_rank), "->", int(my_rank))
        dist.recv(tensor = request_tensor, src = src_rank)
        #req.wait()

        request_wid_rank = request_tensor[-1] + TQ_BASE
        #print("CRTS: ",int(wid),"->",int(request_tensor[-1]),"\t", int(request_tensor[0]),"\t",int(request_tensor[1]))
        #TODO: get the requested tensor 
        #request_data,request_ctx = get_requested_data(request_tensor[0], request_tensor[1])
        request_data = get_requested_data(request_tensor[0], request_tensor[1])
        #print("CP2T:",  int(my_rank), "->", int(request_wid_rank) )
        send_req = dist.isend(tensor = request_data, dst = request_wid_rank)
        send_req.wait()
        #print("CP2T-FIN:",  int(my_rank), "->", int(request_wid_rank) )


if __name__ == '__main__':
    ini_data_storage()
    wid = args.wid
    print("wid = ", wid)
    #ts_p = mp.Process(target=train_sync_proc, kwargs={"wid":wid})
    #ts_p.start()
    '''
    for i in range(100):
        fake_input = fake_input.cuda()
        out_put = SUB_MODEL_LIST[0](fake_input)
        fake_input = fake_input.cpu()
        out_put = out_put.cpu()
        torch.cuda.empty_cache()
        time.sleep(2)
        print("i = ", i)
    '''
    c_p = mp.Process(target=coordinate_proc, kwargs={"wid":wid})
    c_p.start()

    ms_p =  mp.Process(target=model_sync_process, kwargs={"wid":wid})
    ms_p.start()
    train_sync_proc(wid)

    ms_p.join()
    c_p.join()