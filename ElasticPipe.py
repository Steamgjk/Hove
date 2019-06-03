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
sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
forward_workload = 0
backward_workload = 0
forward_span = 0
backward_span = 0
#perform_info =[[fw, fs, bw, bs],...,] worker_num * 4
#partition = [0, -1] worker_num+1
forward_recv_tensor=None
backward_recv_tensor = None
perf_info = torch.zeros(worker_num*2, dtype=torch.float)
partition_info = work_partition
new_partition = [0]*(worker_num+1)
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
        if isinstance(layer, HookLayer):
            print("HookLayer jump off")
            continue
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
        weight_size = 0
        bias_size = 0
        if(hasattr(layer, "weight")):
            weight_shape = layer.weight.data.size()
            weight_size = torch.tensor(weight_shape).prod().item()
        if(hasattr(layer, "bias")):
            bias_shape = layer.bias.data.size()
            bias_size = torch.tensor(bias_shape).prod().item()
        dict_item = {"shape":shp,"size":num,"type":layer_type, "flops": flops,"weight_shape":weight_shape, "weight_size":weight_size,"bias_shape":bias_shape, "bias_size":bias_size, "bias_size":bias_size}
        profile_shape.append(dict_item)
        if isinstance(layer, torch.nn.Conv2d) and cnt >0 :
            break_points.append(cnt)
        cnt = cnt + 1
    out = out.view(out.size(0), -1)
    for layer in test_net.fc_layers:
        out = layer(out)
        shp = out.size()
        num = torch.tensor(shp).prod().item()
        layer_type = type(layer)
        flops = get_flops(layer, batch_size, -1, -1, -1, -1, -1, -1)
        if(hasattr(layer, "weight")):
            weight_shape = layer.weight.data.size()
            weight_size = torch.tensor(weight_shape).prod().item()
        if(hasattr(layer, "bias")):
            bias_shape = layer.bias.data.size()
            bias_size = torch.tensor(bias_shape).prod().item()
        dict_item = {"shape":shp,"size":num,"type":layer_type, "flops": flops,"weight_shape":weight_shape, "weight_size":weight_size,"bias_shape":bias_shape, "bias_size":bias_size, "bias_size":bias_size}
        profile_shape.append(dict_item)

    layer = test_net.classifier
    out = layer(out)
    shp = out.size()
    num = torch.tensor(shp).prod()
    layer_type = type(layer)
    flops = get_flops(layer, batch_size, -1, -1, -1, -1, -1, -1)
    if(hasattr(layer, "weight")):
        weight_shape = layer.weight.data.size()
        weight_size = torch.tensor(weight_shape).prod().item()
    if(hasattr(layer, "bias")):
        bias_shape = layer.bias.data.size()
        bias_size = torch.tensor(bias_shape).prod().item()
    dict_item = {"shape":shp,"size":num,"type":layer_type, "flops": flops,"weight_shape":weight_shape, "weight_size":weight_size,"bias_shape":bias_shape, "bias_size":bias_size, "bias_size":bias_size}
    profile_shape.append(dict_item)
    return profile_shape,break_points

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '12.12.10.13'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
def come_to_adjust(cur_iter, epo):
    #if False:
    if cur_iter > 0 and cur_iter % epo == 0:
        return True
    else:
        return False
def sum_flops(slidx, elidx):
    global shape_list
    flops_sum = 0
    for i in range(slidx, elidx):
        flops_sum += shape_list[i]['flops']
    return flops_sum
def re_partition():
    global partition_info
    global perf_info
    global shape_list
    global break_list
    global new_partition
    global worker_num
    total_speed = 0
    total_flops = 0
    speed_list = []
    sum_flops = []
    for i in range(len(shape_list)):
        #print(shape_list[i])
        #input("let me see")
        flops = shape_list[i]['flops']
        if i == 0:
            sum_flops.append(flops)
        else:
            sum_flops.append(sum_flops[-1]+shape_list[i]['flops'])
    for i in range(worker_num):
        fspan = perf_info[i*2]
        bspan = perf_info[i*2+1]
        s_lidx = partition_info[i]
        e_lidx = partition_info[i+1]
        if(e_lidx == -1):
            e_lidx = len(shape_list)
        if s_lidx == 0:
            flops_sum = sum_flops[e_lidx-1]
        else:
            flops_sum = sum_flops[e_lidx-1] - sum_flops[s_lidx-1] 
        span_sum = fspan+bspan
        speed = flops_sum/span_sum
        #print("i=",i," flops_sum=",flops_sum, " span_sum=",span_sum, " speed=",speed)
        speed_list.append(speed)
        total_speed += speed
        total_flops += flops_sum
    unit_workload = total_flops/total_speed
    #print("break_list=",break_list)
    break_sta_idx = 0
    break_ed_idx = 1
    new_partition[0] = 0
    for i in range(worker_num-1):
        target_workload = unit_workload * speed_list[i]
        while True:
            candidate_break_idx = break_list[break_sta_idx]
            tail_idx = candidate_break_idx -1 
            head_idx = new_partition[i]
            acc_workload = sum_flops[tail_idx] - sum_flops[head_idx]
            if acc_workload >target_workload or acc_workload == target_workload or break_sta_idx > len(break_list)-(worker_num-i):
                new_partition[i+1] = candidate_break_idx
                break_sta_idx += 1
                break
            else:
                break_sta_idx += 1

    new_partition[worker_num]= -1

    return new_partition

def partition_diff(old_partition, new_partition, worker_id):
    global shape_list
    foward_s = []
    foward_r = []
    backward_s = []
    backward_r = []
    old_sta = old_partition[worker_id]
    old_ed = old_partition[worker_id+1]
    new_sta = new_partition[worker_id]
    new_ed = new_partition[worker_id+1]
    '''
    print("old_partition ")
    print(old_partition)
    print("new_partition ")
    print(new_partition)
    '''
    fs_idx = new_ed
    while old_ed>0 and fs_idx < old_ed:
        foward_s.append(fs_idx)
        fs_idx += 1
    br_idx = old_ed
    while new_ed>0 and br_idx < new_ed:
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
    if shape_list[idx]["weight_size"] == 0:
        return None
    weight_data = torch.zeros(shape_list[idx]["weight_size"],dtype=torch.float)
    bias_data = torch.zeros(shape_list[idx]["bias_size"],dtype=torch.float)
    weight_data = weight_data.view(1,-1)
    bias_data = bias_data.view(1,-1)
    tensor_data = torch.cat((weight_data,bias_data),1)
    tensor_data = tensor_data.view(1,-1)
    return tensor_data
def pack_layer_info(idx):
    if shape_list[idx]["weight_size"] == 0:
        return None
    weight_tensor = sub_net.feature_arr[idx].weight.data.cpu().clone()
    bias_tensor = sub_net.feature_arr[idx].bias.data.cpu().clone()
    weight_tensor = weight_tensor.view(1,-1)
    bias_tensor = bias_tensor.view(1,-1)
    tensor_data = torch.cat((weight_tensor,bias_tensor),1)
    tensor_data = tensor_data.view(1,-1)
    return tensor_data
def unpack_tensor(layer_idx, tensor_data):
    weight_size = shape_list[layer_idx]["weight_size"]
    grad_size = shape_list[layer_idx]["bias_size"]
    weight_data = tensor_data[0,0:weight_size]
    bias_data = tensor_data[0,weight_size:]
    weight_shape = shape_list[layer_idx]["weight_shape"]
    bias_shape = shape_list[layer_idx]["bias_shape"]
    sub_net.feature_arr[layer_idx].weight.data = weight_data.reshape(weight_shape)
    sub_net.feature_arr[layer_idx].bias.data = bias_data.reshape(bias_shape)

def auto_tune(iter_n, adjust_epoch):
    global forward_span,backward_span,new_partition,perf_info
    global work_partition,boundary_shape,boundary_size
    global forward_recv_tensor,backward_recv_tensor
    global device 
    if come_to_adjust(iter_n,adjust_epoch):
        perf_info[worker_id*2]=forward_span
        perf_info[worker_id*2+1]=backward_span
        #actually we should transfer one-by-one
        dist.reduce(tensor = perf_info, dst = 0, op=dist.ReduceOp.SUM)
        #print("perf_info:",perf_info)
        if worker_id == 0:
            new_partition = re_partition()
            dist.broadcast(tensor=torch.tensor(new_partition), src = 0)
        else:
            cache_partition = torch.tensor(new_partition)
            dist.broadcast(tensor=cache_partition, src = 0)
            new_partition = cache_partition.tolist()
        perf_info.zero_()
        #print(new_partition)

        forward_layer_to_send, forward_layer_to_recv, backward_layer_to_send,backward_layer_to_recv = partition_diff(partition_info,new_partition, worker_id)
        #print(forward_layer_to_send)
        #print(forward_layer_to_recv)
        #print(backward_layer_to_send)
        #print(backward_layer_to_recv)
        #input("dddddd")
        for layer_idx in forward_layer_to_recv:
            tensor_data = get_cache_tensor(layer_idx)
            if tensor_data is None:
                continue
            tensor_data = tensor_data.view(1,-1)
            #print("layer_idx=",layer_idx," come to recv ",tensor_data.size())
            dist.recv(tensor = tensor_data, src = worker_id-1)
            tensor_data =tensor_data.to(device)
            unpack_tensor(layer_idx, tensor_data)
        #input("Forward Recv Complete")
        for layer_idx in forward_layer_to_send:
            tensor_data = pack_layer_info(layer_idx)
            if tensor_data is None:
                continue
            #print("layer_idx=",layer_idx," come to send ",tensor_data.size())
            tensor_data = tensor_data.cpu()
            tensor_data = tensor_data.view(1,-1)
            dist.send(tensor = tensor_data, dst = worker_id+1)
        #input("Forward Send Complete")
        for layer_idx in backward_layer_to_recv:
            tensor_data = get_cache_tensor(layer_idx)
            if tensor_data is None:
                continue
            #print("layer_idx=",layer_idx," come to recv ",tensor_data.size())
            dist.recv(tensor= tensor_data, src = worker_id+1)
            tensor_data =tensor_data.to(device)
            unpack_tensor(layer_idx, tensor_data)
        #input("Backward Recv Complete")
        for layer_idx in backward_layer_to_send:
            tensor_data = pack_layer_info(layer_idx)
            if tensor_data is None:
                continue
            tensor_data = tensor_data.cpu()
            dist.send(tensor = tensor_data, dst = worker_id-1)
        #input("Backward Send Complete")
        if not (len(forward_layer_to_send) == 0 and len(forward_layer_to_recv) == 0 and len(backward_layer_to_send) == 0 and len(backward_layer_to_recv) == 0):
            #print("repack ",new_partition[worker_id]," ", new_partition[worker_id+1])
            sub_net._repack_layers(new_partition[worker_id],new_partition[worker_id+1])
        for i in range(len(work_partition)):
            work_partition[i] = new_partition[i]
        for i in range(len(boundary_shape)):
            boundary_idx = work_partition[i+1]
            boundary_shape[i] = shape_list[boundary_idx-1]['shape']
            boundary_size[i] = shape_list[boundary_idx-1]['size']
        if not worker_id == 0:
            forward_recv_tensor = torch.zeros(boundary_size[worker_id-1],dtype=torch.float)
            forward_recv_tensor = forward_recv_tensor.reshape(boundary_shape[worker_id-1])
        if not worker_id == worker_num -1:
            backward_recv_tensor = torch.zeros(boundary_size[worker_id], dtype=torch.float)
            backward_recv_tensor = backward_recv_tensor.reshape(boundary_shape[worker_id])

        sub_net.to(device)
        sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        forward_span = 0
        backward_span = 0

def pipe_train():
    global shape_list,break_list, sub_optimizer
    global forward_recv_tensor,backward_recv_tensor
    global forward_span,backward_span
    global perf_info
    global sub_net,sub_optimizer, device

    shape_list,break_list = profile_tensor_shape()
    '''
    for i  in range(len(shape_list)):
        print("i=",i," ", shape_list[i])
    for i  in range(len(break_list)):
        print("i=",i," ", break_list[i])    
    input("stop here")
    '''

    init_processes(worker_id,worker_num,'gloo')
    print('sta %d  end %d' % (sta_lidx, end_lidx))
    print("worker_id=",worker_id," batch_size=",batch_size)
    if not worker_id == 0:
        forward_recv_tensor = torch.zeros(boundary_size[worker_id-1],dtype=torch.float)
        forward_recv_tensor = forward_recv_tensor.reshape(boundary_shape[worker_id-1])
    if not worker_id == worker_num -1:
        backward_recv_tensor = torch.zeros(boundary_size[worker_id], dtype=torch.float)
        backward_recv_tensor = backward_recv_tensor.reshape(boundary_shape[worker_id])
    
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
    is_cpu_mode = False
    sta = time.time()

    while iter_n<= iteration_num:
        auto_tune(iter_n,adjust_epoch)
        #print("auto_tune complete")
        #Forward
        if forward_iters < batch_thresh:
            if worker_id == 0:
                #first worker, no need to receive inputs
                time_sta = time.time()
                inputs = fake_input.to(device)
                #inputs.requires_grad = True
                #input("chadddd")
                outputs = sub_net(inputs)
                outputs_list.append(outputs)
                cpu_tensor = outputs.data.cpu()
                forward_para_num += cpu_tensor.numel()
                time_ed =time.time()
                forward_span += time_ed - time_sta
                dist.send(tensor=cpu_tensor, dst=worker_id+1)
                forward_iters = forward_iters+1
                
            elif worker_id == worker_num-1:                    
                dist.recv(tensor=forward_recv_tensor, src = worker_id-1)
                time_sta = time.time()
                forward_recv_tensor.requires_grad = True
                inputs = forward_recv_tensor.to(device)
                outputs = sub_net(inputs)  
                forward_iters = forward_iters +1
                targets = fake_target.to(device)
                #print("targets size=",targets.size(), " device=",targets.device)
                #print("outputs size= ", outputs.size()," device=",outputs.device)
                loss = criterion(outputs, targets)
                time_ed = time.time()
                forward_span += time_ed - time_sta
                time_sta = time.time()
                loss.backward()
                if HookFunc.hook_dict["backward_ctx"] is not None:
                    cpu_tensor = HookFunc.hook_dict["backward_ctx"].cpu()
                    backward_para_num += cpu_tensor.numel()
                    time_ed = time.time()
                    backward_span += time_ed - time_sta
                    dist.send(tensor=cpu_tensor, dst = worker_id-1)
                    HookFunc.hook_dict["backward_ctx"] = None
                    backward_iters = backward_iters + 1
                else:
                    #print("Error-1")
                    exit(-1)
                

            else:
                dist.recv(tensor=forward_recv_tensor, src = worker_id-1)
                time_sta = time.time()
                forward_recv_tensor.requires_grad = True
                inputs = forward_recv_tensor.to(device)
                #print("input device", inputs.device)
                outputs = sub_net(inputs)
                outputs_list.append(outputs)
                cpu_tensor = outputs.data.cpu()
                forward_para_num += cpu_tensor.numel()
                time_ed = time.time()
                forward_span += time_ed - time_sta
                dist.send(tensor= cpu_tensor, dst = worker_id +1)
                forward_iters = forward_iters+1
                
        #Backward:  only for non-last workers
        if backward_iters < batch_thresh:
            if backward_iters < forward_iters:
                #get recved parameter
                dist.recv(tensor=backward_recv_tensor, src = worker_id+1)
                time_sta = time.time()
                outputs = outputs_list.pop()
                backward_ctx = backward_recv_tensor.to(device)
                outputs.backward(backward_ctx)
                backward_iters = backward_iters + 1
                time_ed = time.time()
                backward_span += time_ed - time_sta 
                if (not worker_id == 0):
                    if HookFunc.hook_dict["backward_ctx"] is not None:
                        cpu_tensor = HookFunc.hook_dict["backward_ctx"].cpu()
                        backward_para_num += cpu_tensor.numel()
                        dist.send(tensor=cpu_tensor, dst = worker_id-1)
                        HookFunc.hook_dict["backward_ctx"] = None       
                    else:
                        print("Error-2")
                        exit(-1)        
                       
        if forward_iters == batch_thresh and backward_iters == batch_thresh:
            #div by batch_thresh
            #print("grad div")
            grad_div(sub_optimizer.param_groups,batch_thresh)
            sub_optimizer.step()
            sub_optimizer.zero_grad()
            forward_iters = 0
            backward_iters = 0
            print("iter_n=", iter_n, "forward_span=",forward_span," backward_span=",backward_span)
            if(iter_n%10 == 0):
                ed = time.time()
                print("span=", str(ed-sta*1.0))
            if(iter_n == 100):
                exit(0)
            if(iter_n > 0 and iter_n %10 == 0):
                cpu_node_idx = (iter_n/10)%worker_num
                if worker_id ==cpu_node_idx and is_cpu_mode==False:
                    print("switch to cpu")
                    os.environ["CUDA_VISIBLE_DEVICES"]=''
                    device = 'cpu'
                    sub_net = sub_net.to(device)
                    sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                    is_cpu_mode = True
                elif (not worker_id == cpu_node_idx) and is_cpu_mode== True:
                    print("switch to cuda")
                    os.environ["CUDA_VISIBLE_DEVICES"]='1'
                    device = 'cuda'
                    sub_net = sub_net.to(device)
                    sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                    is_cpu_mode= False
            '''
            for param in sub_net.parameters():
                print("Type=",type(param.data))
                print(param.data.size(),"----",param.grad.size())
            '''
            iter_n = iter_n +1


    #print(prof,file=f)    




pipe_train()


