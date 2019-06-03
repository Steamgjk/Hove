# -*- coding: utf-8 -*
'''Train CIFAR10 with PyTorch.'''
## TODO: 1. Interaction between C and Python  2. Dynamic add/delete net layer  3. Aggregate Gradient
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
gluer = CDLL('./glue.so')

# In-place aggregation

#test_net = VGG("VGG19")
#summary(test_net, (3, 32, 32))

os.environ["CUDA_VISIBLE_DEVICES"]=''
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--pid', default=0, type=int, help='worker id')
parser.add_argument('--pn', default=1, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
args = parser.parse_args()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device =  'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

aggregate_cnt_dict={}
aggregate_data_dict = {}
gluer = CDLL('./ps_glue.so')
worker_num = args.wn
ps_id = args.pid # 1|2|3|
ps_num = args.pn

pickle_dump_span = 0
pickle_dump_len = 0
pickle_load_span = 0
pickle_load_len = 0
aggregation_span = 0
aggregation_size = 0

def get_aggregated_para(para_list,div_partition):
    #print("para_list_len=",len(para_list))
    para_sum = para_list[0]
    for idx in range(len(para_list)):
        para_sum.add_(para_list[idx])
    para_sum.div_(div_partition)
    return para_sum
#PS0: [][][][]...
def aggregate_func():
    gluer.init_info(ps_id, ps_num, worker_num)
    gluer.init_mem()
    
    global pickle_dump_span
    global pickle_dump_len
    global pickle_load_span
    global pickle_load_len
    global aggregation_span
    global aggregation_size 
    for worker_id in range(worker_num):
        recv_channel_id = worker_id  
        gluer.launch_recv_thread(recv_channel_id)   

    time.sleep(5)

    for worker_id in range(worker_num):
        send_channel_id = worker_id  
        gluer.launch_send_thread(send_channel_id)   

    print("worker_num=",worker_num)
    input("Connection Initialized worker") 
    t_sta = time.process_time()
    while True:
        for worker_id in range(worker_num):
            data_len = gluer.inquire_recv_mem_len(worker_id)
            #print("data_len=", data_len)
            if data_len > 0:
                #print("worker_id=", worker_id, " ", "data_len=", data_len)
                data_out = bytes(data_len)
                ret = gluer.dequeue_recv_mem(worker_id, data_out, data_len)
                pickle_load_len += data_len
                sta = time.process_time() 
                data_out = pickle.loads(data_out)
                ed = time.process_time()
                pickle_load_span += ed - sta
                key_name= data_out['key_name']
                content = data_out['gradient']
                #print("recv key_name=", key_name,"  data_len=",data_len," ret=",ret," worker_id=",worker_id)
                if key_name in aggregate_cnt_dict:
                    aggregate_cnt_dict[key_name] = aggregate_cnt_dict[key_name] + 1
                    aggregate_data_dict[key_name].append(content)
                else:
                    aggregate_cnt_dict[key_name]= 1
                    content_list =[content]
                    aggregate_data_dict[key_name] = content_list
                #print("key_name=",key_name," cnt=", aggregate_cnt_dict[key_name])

                if aggregate_cnt_dict[key_name] == worker_num:
                    #can aggregate
                    #print("can aggregate ", key_name, "worker_num=",worker_num)
                    sta = time.process_time()
                    aggregation_size += aggregate_data_dict[key_name][0].numel()
                    aggregated_data = get_aggregated_para(aggregate_data_dict[key_name],worker_num)
                    ed = time.process_time()
                    aggregation_span += ed - sta
                    aggregated_dict = {"key_name":key_name,"gradient":aggregated_data}
                    #print(type(aggregated_data))
                    sta = time.process_time()
                    data_in = pickle.dumps(aggregated_dict)
                    ed = time.process_time()
                    pickle_dump_len += len(data_in)
                    pickle_dump_span += ed - sta

                    for wid in range(worker_num):
                        #print("Enque  worker_id=", wid, "key_name=",key_name)
                        gluer.enque_send_mem(wid, data_in, len(data_in))
                    aggregate_data_dict[key_name]=[]
                    aggregate_cnt_dict[key_name]= 0
                    t_ed = time.process_time()
                    print("pickle_dump_span=",pickle_dump_span," pickle_dump_len=",pickle_dump_len," speed=",str(pickle_dump_len*1.0/pickle_dump_span))
                    print("pickle_load_span=",pickle_load_span," pickle_load_len=",pickle_load_len," speed=",str(pickle_load_len*1.0/pickle_load_span))
                    print("aggregation_span=",aggregation_span," aggregation_size=",aggregation_size," speed=",str(aggregation_span*1.0/aggregation_span))
                    print("total_span=",str(t_ed-t_sta))



aggregate_func()

