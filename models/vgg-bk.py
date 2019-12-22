# -*- coding: utf-8 -*
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.init as init

import globalvar as gl

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

 
class VGG_O(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_O, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        init.constant_(self.classifier.weight, 0.02)
        init.constant_(self.classifier.bias, 0.02)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                init.constant_(layers[-1].weight, 0.01)
                init.constant_(layers[-1].bias, 0.01)
                layers += [nn.BatchNorm2d(x)]
                init.constant_(layers[-1].weight, 0.1)
                init.zeros_(layers[-1].bias)
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



# Inherit from Function
class HookFunc(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(self, input, layer_idx = -1, break_layer_idx= -1):
        #print("ForwardHook, layer_idx= ", layer_idx)
        self.save_for_backward(torch.tensor([layer_idx, break_layer_idx]))
        return input.view_as(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        layer_idx = self.saved_tensors[0].data[0]
        break_layer_idx = self.saved_tensors[0].data[1]
        #print("Backward,layer_idx=", layer_idx)
        if (break_layer_idx > 0) and (layer_idx == break_layer_idx):
            gl.set_value('backward_ctx', grad_output)
            print("OK has stored grad_output ",break_layer_idx)
            #print(grad_output)
        return grad_output, None, None



class HookLayer(torch.nn.Module):
    def __init__(self, layer_idx = -1, break_layer_idx= -1):
        super(HookLayer, self).__init__()
        self.layer_idx = layer_idx
        self.break_layer_idx = break_layer_idx
        #print("Virtual Layer:", layer_idx)
        if self.break_layer_idx > 0 and self.layer_idx == self.break_layer_idx:
            print("Break Layer Idx ", self.break_layer_idx)


    def forward(self, input):
        # See the autograd section for explanation of what happens here
        # 从此开始，维持和F.linear形参一致，更具体的是跟LinearFunction里面的forward函数的形参一致
        #print("HookLayer forward called:", self.layer_idx)
        
        return HookFunc.apply(input, self.layer_idx, self.break_layer_idx)

debug_arr = []
class VGG_D(nn.Module):
    def __init__(self, vgg_name, sta_lidx = -1, end_lidx=-1):
        super(VGG_D, self).__init__()
        self.feature_arr = [] #Important
        layer_cnt = 0
        self.feature_arr.append(HookLayer(-1))
        self.classifier = nn.Linear(512, 10)

    def forward(self, sta_input):
        out = sta_input
        for layer_idx in range(len(self.feature_arr)):
            feature_layer = self.feature_arr[layer_idx]
            #out = feature_layer(out)
            #out = nn.Linear(512,512)(out)
            out = HookLayer()(out)
            print("ff:",layer_idx, out.grad_fn)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out    
class VGG(nn.Module):
    def __init__(self, vgg_name, sta_lidx = -1, end_lidx=-1, break_layer_idx= -1):
        super(VGG, self).__init__()
        layer_arr = self._make_layers2(cfg[vgg_name]) 
        self.real_layer_num = len(layer_arr)
        self.break_layer_idx = break_layer_idx

        self.feature_arr = [] #Important
        self.working_layer_arr = []
        self.head_layer_to_send = []
        self.tail_layer_to_send = []
        #TODO  only keep one virtual layer
        layer_cnt = 0
        self.features = nn.Sequential()
        #self.feature_arr.append(HookLayer(-1))
        for layer_idx in range(self.real_layer_num):
            self.feature_arr.append(layer_arr[layer_idx])
            #print(type(layer_arr[layer_idx]), layer_arr[layer_idx].training)
            layer_cnt = layer_cnt + 1
            self.feature_arr.append(HookLayer(layer_cnt, self.break_layer_idx)) #virtual layer
            layer_cnt = layer_cnt + 1

        self.total_layer_num = layer_cnt
        self.sta_layer_idx = 0
        self.end_layer_idx = self.total_layer_num

        if not sta_lidx < 0:
            self.sta_layer_idx = sta_lidx
        if not end_lidx < 0:
            self.end_layer_idx = end_lidx
        self.working_layer_arr = self.feature_arr[self.sta_layer_idx:self.end_layer_idx]
        self.features = nn.Sequential(*(self.working_layer_arr))
        self.need_repack = False
        self.classifier = nn.Linear(512, 10)

        init.constant_(self.classifier.weight, 0.02)
        init.constant_(self.classifier.bias, 0.02)

    def _chunk_head(num = -1):
        if num == -1:
            pass
        else:
            self.head_layer_to_send = self.working_layer_arr[0:num]
            self.working_layer_arr = self.working_layer_arr[num:]
            self.sta_lidx = self.sta_lidx + num
            self.need_repack = True
    def _chunk_tail(num = -1):
        if num == -1:
            pass
        else:
            self.tail_layer_to_send = self.working_layer_arr[-num:]
            self.working_layer_arr = self.working_layer_arr[:-num]
            self.end_lidx = self.end_lidx - num
            self.need_repack = True
    def _concat_head(layer_arr=[], num = -1):
        if num == -1:
            pass
        else:
            self.working_layer_arr = self.layer_arr + self.working_layer_arr
            self.sta_lidx = self.sta_lidx - num
            self.need_repack = True
    def _concat_tail(layer_arr=[], num = -1):
        if num == -1:
            pass
        else:
            self.working_layer_arr = self.working_layer_arr + layer_arr
            self.end_lidx = self.end_lidx + num
            self.need_repack = True

    def forward(self, sta_input):
        if self.need_repack:
            self.features = nn.Sequential(*(self.working_layer_arr))
        out = sta_input
        for layer in self.features:
            out = layer(out)

        if self.end_layer_idx == self.total_layer_num:    
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
        return out

    def _make_layers2(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                init.constant_(layers[-1].weight, 0.01)
                init.constant_(layers[-1].bias, 0.01)
                #print("Init 0.01 ")
                layers += [nn.BatchNorm2d(x)]
                init.constant_(layers[-1].weight, 0.1)
                init.zeros_(layers[-1].bias)
                #print("Init 0.1 ")
                layers += [nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return layers        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG19')
    #x = torch.randn(2,3,32,32)
    x = torch.ones(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
'''
bb = torch.ones(1,512)
aa  =torch.ones(1,512)
bb = torch.autograd.Variable(bb, requires_grad= True)
aa = torch.autograd.Variable(aa, requires_grad = True)
aa = HookLayer()(aa)
print("aathis:", aa.grad_fn)
out = bb + aa
print(out.grad_fn)
out = out.sum()
out.backward()
print(out.grad_fn)
debug_out = HookLayer()(out)
print(type(out))
print("sure?? ",debug_out.grad_fn)
'''
'''
dd  = VGG_D("DEBUG")
x = torch.ones(1,512)
print(x.grad_fn)
print(x.requires_grad)
x.requires_grad = True
print(x.requires_grad)
#x = torch.autograd.Variable(torch.ones(1,512),requires_grad= True)
y= dd(x)
z = y.sum()
z.backward()
'''
