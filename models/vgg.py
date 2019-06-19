# -*- coding: utf-8 -*
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.init as init
import os
import globalvar as gl

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

gl._init()

# Inherit from Function
class HookFunc(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    hook_dict={}
    @staticmethod
    # bias is an optional argument
    def forward(self, input):
        #self.save_for_backward(torch.tensor([layer_idx, break_layer_idx]))
        #pid = os.getpid()
        #print("Hook Forward ", pid)
        #HookFunc.hook_dict[pid] = None
        return input.view_as(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        #layer_idx = self.saved_tensors[0].data[0]
        #break_layer_idx = self.saved_tensors[0].data[1]
        #print("Backward,layer")
        # will be replaced by transfer
        #print(type(grad_output))
        #print("Hook Backward  ", id(grad_output))
        pid = os.getpid()
        #print("pid = ", pid)
        HookFunc.hook_dict[pid] = grad_output
        return grad_output


class HookLayer(torch.nn.Module):
    def __init__(self, pipe_rank = -1):
        super(HookLayer, self).__init__()
        self.pipe_rank = pipe_rank

    def forward(self, input):
        # See the autograd section for explanation of what happens here
        # 从此开始，维持和F.linear形参一致，更具体的是跟LinearFunction里面的forward函数的形参一致
        #print("HookLayer forward called:", self.layer_idx)
        return HookFunc.apply(input)
        #exec('return HookFunc_{}.apply(input)'.format(self.pipe_rank))


class VGG_D(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

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
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, vgg_name, sta_lidx = -1, end_lidx=-1, pipe_rank = -1):
        super(VGG, self).__init__()
        self.feature_arr = self._make_layers2(cfg[vgg_name]) 
        self.conv_layer_num = len(self.feature_arr)
        self.working_layer_arr = []
        self.head_layer_to_send = []
        self.tail_layer_to_send = []
        #TODO  only keep one virtual layer
        '''
        self.sta_lidx = 0
        self.end_lidx = self.total_layer_num
        if not sta_lidx < 0:
            self.sta_lidx = sta_lidx
        if not end_lidx < 0:
            self.end_lidx = end_lidx
        if not pipe_rank < 0:
            self.pipe_rank = pipe_rank
        '''
        self.sta_lidx = sta_lidx
        self.end_lidx = end_lidx
        if self.sta_lidx is None or self.sta_lidx == -1:
            self.sta_lidx = 0
        if self.end_lidx is None or self.end_lidx == -1:
            self.end_lidx = self.conv_layer_num
        self.working_layer_arr = self.feature_arr[self.sta_lidx:self.end_lidx]
        self.virtual_layer = HookLayer()
        self.features = nn.Sequential(*([self.virtual_layer]+self.working_layer_arr))

        #print(self.features)
        self.need_repack = False
        input_dim = 512*7*7
        output_dim = 4096
        class_num = 1000
        if self.end_lidx == self.conv_layer_num:
            #self.fc_layers = nn.Sequential(nn.Linear(512, 512),nn.Linear(512, 512),nn.Linear(512, 512))
            #self.classifier = nn.Linear(512, 10)
            self.fc_layers = nn.Sequential(nn.Linear(input_dim, output_dim),nn.Linear(output_dim, output_dim),nn.Linear(output_dim, output_dim))
            self.classifier = nn.Linear(output_dim, class_num)
            #init.constant_(self.classifier.weight, 0.02)
            #init.constant_(self.classifier.bias, 0.02)

    def _chunk_head(self, num = -1):
        if num == -1:
            pass
        else:
            self.head_layer_to_send = self.working_layer_arr[0:num]
            self.working_layer_arr = self.working_layer_arr[num:]
            self.sta_lidx = self.sta_lidx + num
            self.need_repack = True
    def _chunk_tail(self, num = -1):
        if num == -1:
            pass
        else:
            self.tail_layer_to_send = self.working_layer_arr[-num:]
            self.working_layer_arr = self.working_layer_arr[:-num]
            self.end_lidx = self.end_lidx - num
            self.need_repack = True
    def _concat_head(self, layer_arr=[], num = -1):
        if num == -1:
            pass
        else:
            self.working_layer_arr = layer_arr + self.working_layer_arr
            self.sta_lidx = self.sta_lidx - num
            self.need_repack = True
    def _concat_tail(self, layer_arr=[], num = -1):
        if num == -1:
            pass
        else:
            self.working_layer_arr = self.working_layer_arr + layer_arr
            self.end_lidx = self.end_lidx + num
            self.need_repack = True
    def _repack(self):
        if self.need_repack:
            #print("Coming...")
            self.features = nn.Sequential(*([self.virtual_layer]+self.working_layer_arr))
            self.need_repack = False
    def _repack_layers(self, new_sta_lidx, new_end_lidx):
        self.sta_lidx = 0
        self.end_lidx = self.total_layer_num
        if not new_sta_lidx < 0:
            self.sta_lidx = new_sta_lidx
        if not new_end_lidx < 0:
            self.end_lidx = new_end_lidx
        self.working_layer_arr = self.feature_arr[self.sta_lidx:self.end_lidx]
        self.features = nn.Sequential(*([self.virtual_layer]+self.working_layer_arr))
    def forward(self, sta_input):
        #print(self.features)
        out = sta_input
        cnt = 0
        #print("Features ",self.features)
        #print(self.features)
        for layer in self.features:
            cnt += 1
            out = layer(out)
            #print(type(layer), "size:", out.size())

        if self.end_lidx == -1 or self.end_lidx == self.conv_layer_num:
            #print("fc")    
            out = out.view(out.size(0), -1)
            out = self.fc_layers(out)
            out = self.classifier(out)
            #print("out sz:",out.size())
        #print("vgg forward OK")
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





