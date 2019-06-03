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


# In-place aggregation

#test_net = VGG("VGG19")
#summary(test_net, (3, 32, 32))

os.environ["CUDA_VISIBLE_DEVICES"]='1'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wid', default=0, type=int, help='worker id')
parser.add_argument('--wn', default=4, type=int, help='worker number')
parser.add_argument('--pn', default=1, type=int, help='worker id')

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
gluer = CDLL('./worker_glue.so')
ps_num = args.pn
worker_num = args.wn
worker_id = args.wid # 1|2|3|
#fake_input = torch.randn(128,3,224,224)
#fake_target = torch.randint(0,1,(128,1000))
batch_size = 1
fake_input = torch.randn(batch_size,3,224,224)
fake_target = torch.from_numpy(np.random.randint(0,999,size=batch_size))
print(fake_input.size())
print(fake_target.size())

#summary(main_net, (3, 224, 224))
#VGG19 54
criterion = nn.CrossEntropyLoss()

sub_net = VGG('VGG19', sta_lidx = -1, end_lidx = -1)
sub_net.to(device)
sub_optimizer = optim.SGD(sub_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

#替换掉module的grad
#f = open("log2.txt", "w") 
#input("Log OK...")

def train():
    #Launch recv td
    gluer.init_info(worker_id, ps_num, worker_num)
    gluer.init_mem()
    for ps_id in range(ps_num):
        recv_channel_id = ps_id
        gluer.launch_recv_thread(recv_channel_id)  

    time.sleep(5)

    for ps_id in range(ps_num):
        send_channel_id = ps_id
        gluer.launch_send_thread(send_channel_id)  

    input("Worker End Connection Initialized") 
    
    sub_net.train()
    inputs = None
    outputs = None
    train_loss = 0
    correct = 0
    total = 0
    iteration_num = 100
    iter_n = 0
    loss = None
    sub_optimizer.zero_grad()
    while True:
        inputs = fake_input.to(device)
        targets = fake_target.to(device)
        outputs = sub_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        robin_ps_id = 0
        grad_num_to_update = 0
        param_item_dict = {}
        for name, parameters in sub_net.named_parameters():
            if(parameters.grad is not None):
                grad_content = parameters.grad.to("cpu")
                grad_dict = {"key_name":name, "gradient":grad_content}
                param_item_dict.update({name:parameters})
                data_in = pickle.dumps(grad_dict)
                ret = gluer.enque_send_mem(robin_ps_id, data_in, len(data_in))
                #print("ret=",ret, " name=",name, "data_in=",len(data_in))
                grad_num_to_update = grad_num_to_update +1
                #print("ret=",ret," key_name=",name," gradient sz=", parameters.grad.size(), "data sz=",parameters.data.size()," ", id(parameters))
                robin_ps_id = (robin_ps_id +1) % ps_num

        print("Wait for update")
        grad_num_updated = 0
        while True:
            for robin_ps_id in range(ps_num):
                data_len = gluer.inquire_recv_mem_len(robin_ps_id)
                if(data_len > 0):
                    data_out = bytes(data_len)
                    gluer.dequeue_recv_mem(robin_ps_id, data_out, data_len)
                    grad_dict =  pickle.loads(data_out)
                    key_name = grad_dict["key_name"]
                    param_item_dict[key_name].grad = grad_dict["gradient"].to(device)
                    grad_num_updated = grad_num_updated +1
                    if(grad_num_to_update == grad_num_updated):
                        break
            if(grad_num_to_update == grad_num_updated):
                break
        sub_optimizer.step()
        iter_n = iter_n + 1
        if iter_n%10 == 0:
            print("iter_n=",iter_n)
        if iteration_num == iter_n:
            exit(0)        
  

        #train_loss += loss.item()
        #_, predicted = output2.max(1)
        #total += targets.size(0) 
        #correct += predicted.eq(targets).sum().item()


train()

