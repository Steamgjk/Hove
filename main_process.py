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

os.environ["CUDA_VISIBLE_DEVICES"]='1'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
'''
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
# Model
'''
print('==> Building model..')
net = VGG('VGG19')
print(net.total_layer_num)

# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = net.to(device)



if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
'''
#load the shared object file
#gluer = CDLL('../glue.so')
gluer.init_worker_info.argtypes=[c_int, c_int]
gluer.init_worker_info.restype = c_int
gluer.init_mem.restype = c_int

worker_num = args.wn
worker_id = args.wid # 1|2|3|
work_partition = [0,14,28,43,-1]
#fake_input = torch.randn(128,3,224,224)
#fake_target = torch.randint(0,1,(128,1000))
batch_size = 4
fake_input = torch.randn(batch_size,3,224,224)
fake_target = torch.from_numpy(np.random.randint(0,999,size=batch_size))
print(fake_input.size())
print(fake_target.size())

#summary(main_net, (3, 224, 224))
#VGG19 54
criterion = nn.CrossEntropyLoss()
sta_lidx = work_partition[worker_id]
end_lidx = work_partition[worker_id+1]
sub_net = VGG('VGG19', sta_lidx = sta_lidx, end_lidx = end_lidx)
sub_net.to(device)
sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

#net = VGG('VGG19', sta_lidx = 0, end_lidx = -1)
#net = VGG('VGG19')
#net = net.to(device)
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


#替换掉module的grad
#f = open("log2.txt", "w") 
#input("Log OK...")
global data_out
data_out = bytes(1024*1024*100)
outputs_list = []
def grad_div(para_groups,div_partition):
    for group in para_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.div_(div_partition)
def pipe_train(epoch):
    print('\nEpoch: %d  sta %d  end %d' % (epoch, sta_lidx, end_lidx))
    #Launch recv td
    gluer.init_worker_info(worker_id, worker_num)
    gluer.init_mem()
    forward_td = worker_id * 2
    backward_td = worker_id* 2 + 1
    if not worker_id == 0:
        gluer.launch_recv_thread(forward_td)
    if not worker_id == worker_num-1:
        gluer.launch_recv_thread(backward_td)

    
    time.sleep(5)

    if not worker_id == 0:
        print("send back ", backward_td)
        gluer.launch_send_thread(backward_td)
        
    if not worker_id == worker_num-1:
        print("send forward ", forward_td) 
        gluer.launch_send_thread(forward_td)
    input("dddd") 
    
    sub_net.train()
    inputs = None
    outputs = None
    
    train_loss = 0
    correct = 0
    total = 0
    global sub_optimizer
    iteration_num = 100
    forward_iters = 0
    backward_iters = 0
    batch_thresh = worker_num
    iter_n = 0
    loss = None
    sub_optimizer.zero_grad()
    sta = time.process_time()
    while True:
        #Forward
        if forward_iters < batch_thresh:
            if worker_id == 0:
                #first worker, no need to receive inputs
                inputs = fake_input.to(device)
                inputs.requires_grad = True
                outputs = sub_net(inputs)
                print("Forward: ", outputs.size())
                outputs_list.append(outputs)
                data_in = pickle.dumps(outputs)
                gluer.enque_forward_send_mem(data_in, len(data_in))

                forward_iters = forward_iters+1
            elif worker_id == worker_num-1:
                #TODO: recved transferred data, Forward+Backward
                data_len = gluer.inquire_forward_recv_mem_len()

                if data_len > 0:
                    gluer.dequeue_forward_recv_mem(data_out, data_len)
                    inputs = pickle.loads(data_out)
                    print("Forward: ", inputs.size())
                    inputs.requires_grad = True
                    inputs = inputs.to(device)
                    outputs = sub_net(inputs)                 
                    forward_iters = forward_iters +1

                    ##################################
                    targets = fake_target.to(device)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    if HookFunc.hook_dict["backward_ctx"] is not None:
                        print("Backward: ", HookFunc.hook_dict["backward_ctx"].size())
                        data_in = pickle.dumps(HookFunc.hook_dict["backward_ctx"])
                        res = gluer.enque_backward_send_mem(data_in, len(data_in))
                        HookFunc.hook_dict["backward_ctx"] = None
                        backward_iters = backward_iters + 1
                    else:
                        print("Error")
                        exit(-1)
            else:
                #TODO: recved transferred data
                data_len = gluer.inquire_forward_recv_mem_len()
                if data_len > 0:
                    gluer.dequeue_forward_recv_mem(data_out, data_len)
                    inputs = pickle.loads(data_out)
                    print("Forward: ", inputs.size())
                    inputs.requires_grad = True
                    inputs = inputs.to(device)
                    outputs = sub_net(inputs)
                    outputs_list.append(outputs)
                    data_in = pickle.dumps(outputs)
                    gluer.enque_forward_send_mem(data_in, len(data_in))
                    forward_iters = forward_iters+1


        #Backward:  only for non-last workers
        if backward_iters < batch_thresh:
            if backward_iters < forward_iters:
                #get recved parameter
                data_len = gluer.inquire_backward_recv_mem_len()
                if data_len > 0:
                    gluer.dequeue_backward_recv_mem(data_out, data_len)
                    backward_ctx = pickle.loads(data_out)
                    print("Backward: ", backward_ctx.size())
                    outputs = outputs_list.pop()
                    outputs.backward(backward_ctx)
                    if HookFunc.hook_dict["backward_ctx"] is not None:
                        data_in = pickle.dumps(HookFunc.hook_dict["backward_ctx"])
                        res = gluer.enque_backward_send_mem(data_in, len(data_in))
                        HookFunc.hook_dict["backward_ctx"] = None
                        backward_iters = backward_iters + 1
                    else:
                        print("Error")
                        exit(-1)                    

    
        if forward_iters == batch_thresh and backward_iters == batch_thresh:
            #div by batch_thresh
            grad_div(sub_optimizer.param_groups,batch_thresh)
            sub_optimizer.step()
            sub_optimizer.zero_grad()
            forward_iters = 0
            backward_iters = 0
            print("iter_n=", iter_n)
            if(iter_n%10 == 0):
                ed = time.process_time()
                print("span=", str(ed-sta*1.0))
            if(iter_n == 100):
                exit(0)
            iter_n = iter_n +1

            #input("STOP")
  

        #train_loss += loss.item()
        #_, predicted = output2.max(1)
        #total += targets.size(0) 
        #correct += predicted.eq(targets).sum().item()


for epoch in range(start_epoch, start_epoch+200):
    pipe_train(epoch)

'''
def train(epoch):
    for iter_n in range(100):
        inputs = fake_input.to(device)
        inputs.requires_grad = True
        targets = fake_target.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        print("loss=", loss)
        loss.backward() 
        print(type(HookFunc.hook_dict["backward_ctx"]))
        print(HookFunc.hook_dict["backward_ctx"].size())
        print(id(HookFunc.hook_dict["backward_ctx"]))
        optimizer.step() 
        input("ssss")  
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    sub_net1.train()
    sub_net2.train()

    train_loss = 0
    main_train_loss = 0
    test_train_loss = 0
    correct = 0
    main_correct = 0
    test_correct = 0
    total = 0
    global sub_optimizer1,sub_optimizer2
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        sub_optimizer1.zero_grad()
        output1 = sub_net1(inputs)

        # 断开之后的requires_grad 必须设置为True，经过验证，调用库函数，
        #如果自己穿进去的input requires_grad=False
        #执行完这一层以后，grad_fn还是会有值的，也就是说，会被加到计算图中
        #但是自己写的HookLayer，如果没有requires_grad==True,经过这一层之后grad_fn还是None
        transferred_input = torch.autograd.Variable(output1.data.clone(), requires_grad=True)
        print("Transferred data...")

        sub_optimizer2.zero_grad()
        output2 = sub_net2(transferred_input)
        loss = criterion(output2, targets)
        loss.backward()
        sub_optimizer2.step()

        backward_ctx = gl.get_value('backward_ctx')

        output1.backward(backward_ctx)
        sub_optimizer1.step()     

        train_loss += loss.item()
        _, predicted = output2.max(1)
        total += targets.size(0) 
        correct += predicted.eq(targets).sum().item()

        ### add 
        print("Reuse Optimizer")
        #update(aggregate) gradient
        for name, parameters in sub_net1.named_parameters():
            print("===",name,"=====")

        move_layer_num = 3
        sub_net1._chunk_tail(move_layer_num)
        #Transfer
        transfer_layer = sub_net1.tail_layer_to_send
        sub_net2._concat_head(transfer_layer, move_layer_num)

        sub_net1._repack()
        sub_net2._repack()
        for name, parameters in sub_net1.named_parameters():
            print("===",name,"++=====")        
        sub_optimizer1 = optim.SGD(sub_net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sub_optimizer2 = optim.SGD(sub_net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


        print("Epoch[%d] Train[%d]  Loss: %.3f vs %.3f vs %.3f | Acc: %.3f%% (%d/%d)  vs %.3f%%(%d/%d) vs  %.3f%%(%d/%d)"  
            % (epoch, batch_idx, train_loss/(batch_idx+1), main_train_loss/(batch_idx+1), test_train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*main_correct/total, main_correct, total, 100.*test_correct/total, test_correct, total), file=f)
        f.flush()
        
        print("Epoch[%d] Train[%d]  Loss: %.3f vs %.3f vs %.3f | Acc: %.3f%% (%d/%d)  vs %.3f%%(%d/%d) vs  %.3f%%(%d/%d)"  
            % (epoch, batch_idx, train_loss/(batch_idx+1), main_train_loss/(batch_idx+1), test_train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*main_correct/total, main_correct, total, 100.*test_correct/total, test_correct, total))

'''


