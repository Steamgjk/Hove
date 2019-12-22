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

'''
batch = 4 iter_n= 100
span= 173.08674669265747
'''
# In-place aggregation
os.environ["CUDA_VISIBLE_DEVICES"]='1'
#test_net = VGG("VGG19")
#summary(test_net, (3, 224, 224))

#forward_recv_sz = [[],[4, 128, 56, 56],[4, 512, 28, 28],[4, 512, 14, 14]]


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--bs', default=1, type=int, help='batch size')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'



worker_num = args.wn
worker_id = args.wid # 1|2|3|
work_partition = [0,14,18,37,-1]
batch_size = args.bs
boundary_shape = [[batch_size, 128, 56, 56], [batch_size, 256, 56, 56],[batch_size, 512, 28, 28]]
shape_list= []
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

f = open("profile_log.txt", "w") 
global forward_recv_tensor
global backward_recv_tensor
global outputs_list
outputs_list= []
def grad_div(para_groups,div_partition):
    for group in para_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.div_(div_partition)

def profile_tensor_shape():
    profile_shape = []
    test_net = VGG("VGG19")
    out = fake_input
    for layer in test_net.features:
        out = layer(out)
        shp = out.size()
        num = torch.tensor(shp).prod()
        layer_type = type(layer)
        dict_item = {"shape":shp,"size":num,"type":layer_type}
        profile_shape.append(dict_item)
    out = out.view(out.size(0), -1)
    for layer in test_net.fc_layers:
        out = layer(out)
        shp = out.size()
        num = torch.tensor(shp).prod()
        layer_type = type(layer)
        dict_item = {"shape":shp,"size":num,"type":layer_type}
        profile_shape.append(dict_item)
    out = test_net.classifier(out)
    shp = out.size()
    num = torch.tensor(shp).prod()
    layer_type = type(layer)
    dict_item = {"shape":shp,"size":num,"type":layer_type}
    profile_shape.append(dict_item)

    return profile_shape

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '12.12.10.13'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

def pipe_train():
    shape_list = profile_tensor_shape()
    for i  in range(len(shape_list)):
        print("i=",i," ", shape_list[i])

    input("stop here")
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
    global sub_optimizer
    iteration_num = 100
    forward_iters = 0
    backward_iters = 0
    batch_thresh = worker_num
    iter_n = 0
    loss = None
    sub_optimizer.zero_grad()
    sta = time.time()
    #with torch.autograd.profiler.profile(use_cuda= True) as prof:
    #with torch.autograd.profiler.emit_nvtx():
    while iter_n<=100:
    #with torch.autograd.profiler.profile(use_cuda= True) as prof:
        #Forward
        if forward_iters < batch_thresh:
            if worker_id == 0:
                #first worker, no need to receive inputs
                inputs = fake_input.to(device)
                inputs.requires_grad = True
                outputs = sub_net(inputs)
                outputs_list.append(outputs)
                cpu_tensor = outputs.data.cpu()
                forward_para_num += cpu_tensor.numel()
                dist.send(tensor=cpu_tensor, dst=worker_id+1)
                forward_iters = forward_iters+1
            elif worker_id == worker_num-1:
                dist.recv(tensor=forward_recv_tensor, src = worker_id-1)
                forward_recv_tensor.requires_grad = True
                inputs = forward_recv_tensor.to(device)
                outputs = sub_net(inputs)  
                forward_iters = forward_iters +1
                targets = fake_target.to(device)
                loss = criterion(outputs, targets)
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

            else:
                dist.recv(tensor=forward_recv_tensor, src = worker_id-1)
                forward_recv_tensor.requires_grad = True
                inputs = forward_recv_tensor.to(device)
                outputs = sub_net(inputs)
                outputs_list.append(outputs)
                cpu_tensor = outputs.data.cpu()
                forward_para_num += cpu_tensor.numel()

                dist.send(tensor= cpu_tensor, dst = worker_id +1)
                forward_iters = forward_iters+1

        #Backward:  only for non-last workers
        if backward_iters < batch_thresh:
            if backward_iters < forward_iters:
                #get recved parameter
                dist.recv(tensor=backward_recv_tensor, src = worker_id+1)
                #print("recv  fin 3")
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
        if forward_iters == batch_thresh and backward_iters == batch_thresh:
            #div by batch_thresh
            #print("grad div")
            grad_div(sub_optimizer.param_groups,batch_thresh)
            sub_optimizer.step()
            sub_optimizer.zero_grad()
            forward_iters = 0
            backward_iters = 0
            print("iter_n=", iter_n, "forward_para_num=",forward_para_num," backward_para_num=",backward_para_num)
            forward_para_num = 0
            backward_para_num = 0
            if(iter_n%10 == 0):
                ed = time.time()
                print("span=", str(ed-sta*1.0))
            if(iter_n == 100):
                exit(0)
            iter_n = iter_n +1
    #print(prof,file=f)    




pipe_train()


