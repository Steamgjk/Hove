# -*- coding: utf-8 -*
'''Train CIFAR10 with PyTorch.'''
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

from models import *
from utils import progress_bar
import torch.nn.init as init

from models import globalvar as gl
gl._init()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
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
criterion = nn.CrossEntropyLoss()
sub_net1 = VGG('VGG19', sta_lidx = 0, end_lidx = 39)
sub_optimizer1 = optim.SGD(sub_net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
print("sub_net1 OK")
sub_net2 = VGG('VGG19', sta_lidx = 39, end_lidx =108, break_layer_idx =39)
sub_optimizer2 = optim.SGD(sub_net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
print("sub_net2 OK")
#main_net = VGG('VGG19', sta_lidx = 0, end_lidx =108, break_layer_idx = -1)
#main_optimizer = optim.SGD(main_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
print("main_net OK")
#替换掉module的grad
f = open("log.txt", "w") 
print("log OK")


raw_input("...")

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    #net.train()
    sub_net1.train()
    sub_net2.train()
    #main_net.train()

    train_loss = 0
    main_train_loss = 0
    correct = 0
    main_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #optimizer.zero_grad()
        #outputs = net(inputs)
        #loss = criterion(outputs, targets)
        #loss.backward()
        #print("inputs", inputs.requires_grad)
        inputs.requires_grad = True
        #orig_input = inputs.clone()
        sub_optimizer1.zero_grad()
        output1 = sub_net1(inputs)
        # 断开之后的requires_grad 必须设置为True，经过验证，调用库函数，
        #如果自己穿进去的input requires_grad=False
        #执行完这一层以后，grad_fn还是会有值的，也就是说，会被加到计算图中
        #但是自己写的HookLayer，如果没有requires_grad==True,经过这一层之后grad_fn还是None
        print("Grad", output1.grad)
        print("is_leaf", output1.is_leaf)
        print("Require_grad", output1.requires_grad)
        transferred_input = torch.autograd.Variable(output1.data, requires_grad=True)
        print("Transferred data...")
        sub_optimizer2.zero_grad()
        output2 = sub_net2(transferred_input)
        loss = criterion(output2, targets)
        loss.backward()
        print("Backward 2")
        sub_optimizer2.step()
        print("Update 2: sub op 2 step")
        backward_ctx = gl.get_value('backward_ctx')
        output1.backward(backward_ctx)
        print("Backward 1")

        sub_optimizer1.step()
        print("Update 1: sub op 1 step")
        #optimizer.step()
        #layer_0 = net.feature_arr[0]
        #layer_1 = net.feature_arr[1]
        #print(type(layer_0))
        #print(layer_0.bias.grad)
        #print(type(layer_1))
        #print(layer_1.weight)
        #raw_input("...")

        #main_optimizer.zero_grad()
        #main_outputs = main_net(inputs)
        #main_loss = criterion(main_outputs, targets)
        #main_loss.backward()
        #main_optimizer.step()
    
        #for layer_idx in range(39):
        #    print("Layer_idx ", layer_idx)
        #    main_net.feature_arr[layer_idx]
        #    if(hasattr(sub_net1.feature_arr[layer_idx], "weight")):
        #        if sub_net1.feature_arr[layer_idx].weight.equal(main_net.feature_arr[layer_idx].weight):
        #            print("Equal:", layer_idx)
        #        else:
        #            print("Not Equal", layer_idx, type(sub_net1.feature_arr[layer_idx]), "vs ", type(main_net.feature_arr[layer_idx]))
        #            print("self weigth == bias?? ", sub_net1.feature_arr[layer_idx].weight.equal(sub_net1.feature_arr[layer_idx].bias))
        #            print("self weigth == bias2?? ", main_net.feature_arr[layer_idx].weight.equal(main_net.feature_arr[layer_idx].bias))
        #            print(sub_net1.feature_arr[layer_idx].weight)
        #            print("=========================")
        #            print(main_net.feature_arr[layer_idx].weight)

        #            raw_input(".....")
        #    else:
        #        print("No Weight")

        train_loss += loss.item()
        #main_train_loss += main_loss.item()

        _, predicted = output2.max(1)
        #_, main_predicted = main_outputs.max(1)


        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #main_correct += main_predicted.eq(targets).sum().item()

        print("Epoch[%d] Train[%d]  Loss: %.3f vs %.3f | Acc: %.3f%% (%d/%d)  vs %.3f%%(%d/%d)"  
            % (epoch, batch_idx, train_loss/(batch_idx+1), main_train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*main_correct/total, main_correct, total), file=f)
        f.flush()
        print("Epoch[%d] Train[%d]  Loss: %.3f vs %.3f | Acc: %.3f%% (%d/%d)  vs %.3f%%(%d/%d)"  
            % (epoch, batch_idx, train_loss/(batch_idx+1), main_train_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*main_correct/total, main_correct, total))
        
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)

