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
import torch.distributed as dist

# In-place aggregation
os.environ["CUDA_VISIBLE_DEVICES"]='1'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--adjust_epoch', default=5, type=int, help='batch size')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
worker_num = args.wn
worker_id = args.wid # 1|2|3|
work_partition = [0,3,28,43,-1]
batch_size = args.bs
adjust_epoch = args.adjust_epoch
boundary_shape = [[batch_size, 64, 224, 224], [batch_size, 512, 28, 28],[batch_size, 512, 14, 14]]
boundary_size = [batch_size*64*224*224, batch_size*512*28*28, batch_size*512*14*14]
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
sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
forward_workload = 0
backward_workload = 0
forward_span = 0
backward_span = 0
#perform_info =[[fw, fs, bw, bs],...,] work_num * 4
#partition = [0, -1] work_num+1
forward_recv_tensor=None
backward_recv_tensor = None
perf_info = torch.zeros(worker_num*4, dtype=torch.float)
partition_info = torch.zeros(worker_num+1, dtype = torch.int)
new_partition = partition_info.clone()
shape_list = []
break_list = []
outputs_list= []

f = open("profile_log.txt", "w") 

def grad_div(para_groups,div_partition):
    for group in para_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.div_(div_partition)
def get_flops(layer, Batchsz, Cin, Cout, Hin, Hout, Win, Wout):
    if isinstance(layer, torch.nn.Conv2d):
        kernel_sz = layer.kernel_size
        kk = torch.prod(torch.tensor(kernel_sz)).item()
        flops1 = Cout * Cin *kk*Hout * Wout * Batchsz
        flops2 = Cout * Hout * Wout * Batchsz
        flops3 = Cin * Cout * kk * Hin *Win * Batchsz
        flops4 = Cout * Hout  * Wout * Batchsz
        flops = flops1 + flops2 +flops3 + flops4
        return flops
    elif isinstance(layer, torch.nn.Linear):
        flops1 = Batchsz * layer.in_features * layer.out_features
        flops2 = Batchsz * layer.out_features
        flops = flops1 + flops2
        return flops
    elif isinstance(layer, torch.nn.MaxPool2d):
        flops = Batchsz * Cin * Hin * Win
        return flops
    else:
        return 0


def profile_tensor_shape():
    profile_shape = []
    break_points = []
    test_net = VGG("VGG19")
    out = fake_input
    cnt = 0
    for layer in test_net.features:
        in_sz = out.size()
        Cin = in_sz[1]
        Hin = in_sz[2]
        Win = in_sz[3]
        out = layer(out) 
        shp = out.size()
        Cout = shp[1]
        Hout = shp[2]
        Wout = shp[3]
        num = torch.tensor(shp).prod().item()
        layer_type = type(layer)
        flops = get_flops(layer, batch_size, Cin, Cout, Hin, Hout, Win, Wout)
        dict_item = {"shape":shp,"size":num,"type":layer_type, "flops": flops}
        profile_shape.append(dict_item)
        if isinstance(layer, torch.nn.Conv2d):
            break_points.append(cnt)
        cnt = cnt + 1
    out = out.view(out.size(0), -1)
    for layer in test_net.fc_layers:
        out = layer(out)
        shp = out.size()
        num = torch.tensor(shp).prod().item()
        layer_type = type(layer)
        flops = get_flops(layer, batch_size, -1, -1, -1, -1, -1, -1)
        dict_item = {"shape":shp,"size":num,"type":layer_type, "flops": flops}
        profile_shape.append(dict_item)

    break_points.append(-1)
    out = test_net.classifier(out)
    shp = out.size()
    num = torch.tensor(shp).prod()
    layer_type = type(layer)
    
    dict_item = {"shape":shp,"size":num,"type":layer_type}
    profile_shape.append(dict_item)

    return profile_shape,break_points

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '12.12.10.13'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
def come_to_adjust(cur_iter, epo):
    if False:
    #if cur_iter > 0 and iter_n % epo == 0:
        return True
    else:
        return False
def re_partition():
    global partition_info
    global perf_info
    new_partition = partition_info.clone()
    #xxxxx
    return new_partition

def partition_diff(old_partition, new_partition, worker_id):
    foward_s = []
    foward_r = []
    backward_s = []
    backward_r = []
    old_sta = old_partition[worker_id]
    old_ed = old_partition[worker_id+1]
    new_sta = new_partition[worker_id]
    new_ed = new_partition[worker_id+1]

    fs_idx = new_ed
    while fs_idx < old_ed:
        foward_s.append(fs_idx)
        fs_idx += 1
    br_idx = old_ed
    while br_idx < new_ed:
        backward_r.append(br_idx)
        br_idx += 1
    fr_idx = new_sta
    while fr_idx < old_sta:
        foward_r.append(fr_idx)
        fr_idx += 1
    bs_idx = old_sta
    while bs_idx < new_sta:
        backward_s.append(bs_idx)
        bs_idx += 1
    return foward_s, foward_r, backward_s,backward_r

def get_cache_tensor(idx):
    #BIAS.data, BIAS.grad
    #we need to cache both the data and the gradient
    weight_data = torch.tensor.zeros(shape_list[idx]["size"])
    grad_data = torch.tensor.zeros(shape_list[idx]["size"])
    tensor_data = torch.cat((weight_data,grad_data),0)
    return tensor_data
def pack_layer_info(idx):
    sz = shape_list[idx]["size"]
    weight_tensor = sub_net.feature_arr[idx].weight.data.view(1,-1)
    grad_tensor = sub_net.feature_arr[idx].weight.grad.view(1,-1)
    tensor_data = torch.cat((weight_tensor,grad_tensor),0)
    return tensor_data
def unpack_tensor(layer_idx, tensor_data):
    shp = shape_list[layer_idx]["shape"]
    sub_net.feature_arr[layer_idx].weight.data = tensor_data[0].reshape(shp)
    sub_net.feature_arr[layer_idx].weight.grad = tensor_data[1].reshape(shp)
def auto_tune(iter_n, adjust_epoch):
    if come_to_adjust(iter_n,adjust_epoch):
        if worker_id == 0:
            new_partition = re_partition(perf_info)
            dist.send(tensor=new_partition, dst = worker_id+1)
        elif worker_id<worker_num-1:
            dist.recv(tensor=new_partition, src = worker_id -1)
            dist.send(tensor=new_partition, dst = worker_id+1)
        else:
            dist.recv(tensor=new_partition, src = worker_id -1)
        
        forward_layer_to_send, forward_layer_to_recv, backward_layer_to_send,backward_layer_to_recv = partition_diff(partition_info,new_partiion, worker_id)
        for layer_idx in forward_layer_to_recv:
            tensor_data = get_cache_tensor(layer_idx)
            dist.recv(tensor = tensor_data, src = worker_id-1)
            unpack_tensor(layer_idx, tensor_data)
        for layer_idx in forward_layer_to_send:
            tensor_data = pack_layer_info(layer_idx)
            dist.send(tensor = tensor_data, dst = worker_id+1)
        for layer_idx in backward_layer_to_recv:
            tensor_data = get_cache_tensor(layer_idx)
            dist.recv(tensor= tensor_data, src = worker_id+1)
            unpack_tensor(layer_idx, tensor_data)
        for layer_idx in backward_layer_to_send:
            tensor_data = pack_layer_info(layer_idx)
            dist.send(tensor = tensor_data, dst = worker_id-1)
        sub_net._repack_layers(new_partition[worker_id],new_partition[worker_id+1])

def pipe_train():
    global shape_list,break_list, sub_optimizer
    global forward_recv_tensor,backward_recv_tensor
    global forward_span,backward_span

    shape_list,break_list = profile_tensor_shape()
    '''
    for i  in range(len(shape_list)):
        print("i=",i," ", shape_list[i])
    for i  in range(len(break_list)):
        print("i=",i," ", break_list[i])    
    input("stop here")
    '''

    init_processes(worker_id,worker_num,'gloo')
    print('\n sta %d  end %d' % (sta_lidx, end_lidx))
    print("worker_id=",worker_id," batch_size=",batch_size)
    if not worker_id == 0:
        forward_recv_tensor = torch.zeros(boundary_size[worker_id-1])
        forward_recv_tensor = forward_recv_tensor.reshape(boundary_shape[worker_id-1])
    if not worker_id == worker_num -1:
        backward_recv_tensor = torch.zeros(boundary_size[worker_id])
        backward_recv_tensor = backward_recv_tensor.reshape(boundary_shape[worker_id])
    
    input("dddd") 
    
    sub_net.train()
    inputs = None
    outputs = None
    forward_para_num = 0
    backward_para_num = 0
    iteration_num = 100
    forward_iters = 0
    backward_iters = 0
    batch_thresh = worker_num
    iter_n = 0
    loss = None
    sub_optimizer.zero_grad()
    
    sta = time.time()

    while iter_n<= iteration_num:
        #auto_tune(iter_n,adjust_epoch)
        #Forward
        if forward_iters < batch_thresh:
            if worker_id == 0:
                #first worker, no need to receive inputs
                #time_sta = time.time()
                inputs = fake_input.to(device)
                inputs.requires_grad = True
                outputs = sub_net(inputs)
                outputs_list.append(outputs)
                cpu_tensor = outputs.data.cpu()
                forward_para_num += cpu_tensor.numel()
                dist.send(tensor=cpu_tensor, dst=worker_id+1)
                forward_iters = forward_iters+1
                #time_ed =time.time()
                #forward_span += time_ed - time_sta
            elif worker_id == worker_num-1:                    
                dist.recv(tensor=forward_recv_tensor, src = worker_id-1)
                #time_sta = time.time()
                forward_recv_tensor.requires_grad = True
                inputs = forward_recv_tensor.to(device)
                outputs = sub_net(inputs)  
                forward_iters = forward_iters +1
                targets = fake_target.to(device)
                loss = criterion(outputs, targets)
                #time_ed = time.time()
                #forward_span += time_ed - time_sta
                #time_sta = time.time()
                loss.backward()
                if HookFunc.hook_dict["backward_ctx"] is not None:
                    cpu_tensor = HookFunc.hook_dict["backward_ctx"].cpu()
                    backward_para_num += cpu_tensor.numel()
                    dist.send(tensor=cpu_tensor, dst = worker_id-1)
                    HookFunc.hook_dict["backward_ctx"] = None
                    backward_iters = backward_iters + 1
                else:
                    #print("Error-1")
                    exit(-1)
                #time_ed = time.time()
                #backward_span += time_ed - time_sta

            else:
                dist.recv(tensor=forward_recv_tensor, src = worker_id-1)
                #time_sta = time.time()
                forward_recv_tensor.requires_grad = True
                inputs = forward_recv_tensor.to(device)
                outputs = sub_net(inputs)
                outputs_list.append(outputs)
                cpu_tensor = outputs.data.cpu()
                forward_para_num += cpu_tensor.numel()
                dist.send(tensor= cpu_tensor, dst = worker_id +1)
                forward_iters = forward_iters+1
                #time_ed = time.time()
                #forward_span += time_ed - time_sta

        #Backward:  only for non-last workers
        if backward_iters < batch_thresh:
            if backward_iters < forward_iters:
                #get recved parameter
                dist.recv(tensor=backward_recv_tensor, src = worker_id+1)
                #time_sta = time.time()
                outputs = outputs_list.pop()
                backward_ctx = backward_recv_tensor.to(device)
                outputs.backward(backward_ctx)
                backward_iters = backward_iters + 1
                if HookFunc.hook_dict["backward_ctx"] is not None:
                    if (not worker_id == 0):
                        cpu_tensor = HookFunc.hook_dict["backward_ctx"].cpu()
                        backward_para_num += cpu_tensor.numel()
                        dist.send(tensor=cpu_tensor, dst = worker_id-1)
                        HookFunc.hook_dict["backward_ctx"] = None       
                else:
                    print("Error-2")
                    exit(-1)        
                #time_ed = time.time()
                #backward_span += time_ed - time_sta        
        if forward_iters == batch_thresh and backward_iters == batch_thresh:
            #div by batch_thresh
            #print("grad div")
            grad_div(sub_optimizer.param_groups,batch_thresh)
            sub_optimizer.step()
            sub_optimizer.zero_grad()
            forward_iters = 0
            backward_iters = 0
            #print("iter_n=", iter_n, "forward_span=",forward_span," backward_span=",backward_span)
            forward_span = 0
            backward_span = 0
            if(iter_n%10 == 0):
                ed = time.time()
                print("span=", str(ed-sta*1.0))
            if(iter_n == 100):
                exit(0)
            iter_n = iter_n +1


    #print(prof,file=f)    




pipe_train()


