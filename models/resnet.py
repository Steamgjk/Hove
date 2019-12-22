'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import time

# Inherit from Function
class ResNetHookFunc(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    #hook_dict={}
    backward_ctx = None
    @staticmethod
    # bias is an optional argument
    def forward(self, input):
        #self.save_for_backward(torch.tensor([layer_idx, break_layer_idx]))
        #pid = os.getpid()
        #print("Hook Forward ")
        #HookFunc.hook_dict[pid] = None
        #print("input sz=",input.size())
        return input.view_as(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        #pid = os.getpid()
        #print("pid = ", pid)
        #HookFunc.hook_dict[pid] = grad_output
        #print("grad_output.sz=",grad_output.size())
        ResNetHookFunc.backward_ctx = grad_output
        #print("ctx size = ", ResNetHookFunc.backward_ctx.size())
        return grad_output


class ResNetHookLayer(torch.nn.Module):
    def __init__(self, pipe_rank = -1):
        super(ResNetHookLayer, self).__init__()
        self.pipe_rank = pipe_rank

    def forward(self, input):
        # See the autograd section for explanation of what happens here
        #从此开始，维持和F.linear形参一致，更具体的是跟LinearFunction里面的forward函数的形参一致
        #print("HookLayer forward called:", self.layer_idx)
        return ResNetHookFunc.apply(input)
        #exec('return HookFunc_{}.apply(input)'.format(self.pipe_rank))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, sta_lidx = -1, ed_lidx = -1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.total_layers = [self.conv1, self.bn1, nn.ReLU(inplace=True)]+ self.layer1 + self.layer2+self.layer3+self.layer4
        self.total_num = len(self.total_layers)
        self.sta_lidx = 0
        self.ed_lidx = -1
        if sta_lidx > 0:
            self.sta_lidx = sta_lidx
        if ed_lidx > 0:
            self.ed_lidx = ed_lidx
        else:
            self.ed_lidx = self.total_num

        self.work_layers = self.total_layers[self.sta_lidx:self.ed_lidx]
        self.virtual_layer = ResNetHookLayer()
        self.features = nn.Sequential(*([self.virtual_layer]+self.work_layers))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers
        #return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        '''
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        '''
        out = x
        for layer in self.features:
            #print("layer_type:", type(layer))
            out = layer(out)
        if self.ed_lidx == self.total_num:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152(num_classes=1000, sta_lidx = -1, ed_lidx = -1):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, sta_lidx, ed_lidx)



def test():
    net = ResNet152()
    batch_sz = 16
    net.to("cuda")
    net.zero_grad()
    opt = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    cnt = 0
    time_list = []
    print("self.feature_num=",net.total_num)

    while True:
        input_tensor = torch.randn(batch_sz,3,32,32)
        input_tensor = input_tensor.cuda()
        fin_output = net(input_tensor)
        #print(net.features)
        fake_target = torch.from_numpy(np.random.randint(0,999,size=batch_sz))
        fake_target = fake_target.cuda()
        print(fin_output.size())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(fin_output, fake_target.cuda())
        loss.backward()
        opt.step()
        time_list.append(time.time())
        iter_num = len(time_list)-1
        if iter_num >1:
            break
        if iter_num > 0:
            print("iter_num=",iter_num,"\titer_time=",float(time_list[-1]-time_list[0]*1.0)/iter_num)



#test()
#profile_resnet_shape()
