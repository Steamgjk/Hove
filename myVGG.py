# -*- coding: utf-8 -*
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.init as init
import os

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Inherit from Function
class HookFunc(torch.autograd.Function):
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
        return input.view_as(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        #pid = os.getpid()
        #print("pid = ", pid)
        #HookFunc.hook_dict[pid] = grad_output
        #print("grad_output.device=",grad_output.device)
        HookFunc.backward_ctx = grad_output
        #print("ctx size = ", HookFunc.backward_ctx.size())
        return grad_output


class HookLayer(torch.nn.Module):
    def __init__(self, pipe_rank = -1):
        super(HookLayer, self).__init__()
        self.pipe_rank = pipe_rank

    def forward(self, input):
        # See the autograd section for explanation of what happens here
        #从此开始，维持和F.linear形参一致，更具体的是跟LinearFunction里面的forward函数的形参一致
        #print("HookLayer forward called:", self.layer_idx)
        return HookFunc.apply(input)
        #exec('return HookFunc_{}.apply(input)'.format(self.pipe_rank))

class myVGG(nn.Module):
    def __init__(self, vgg_name, sta_lidx = -1, end_lidx=-1, op_code = -1):
        super(myVGG, self).__init__()
        self.feature_arr = self._make_layers2(cfg[vgg_name]) 
        self.conv_layer_num = len(self.feature_arr)
        #TODO  only keep one virtual layer
        self.sta_lidx = sta_lidx
        self.end_lidx = end_lidx
        self.op_code = op_code
        if self.sta_lidx is None or self.sta_lidx == -1:
            self.sta_lidx = 0
        if self.end_lidx is None or self.end_lidx == -1:
            self.end_lidx = self.conv_layer_num
        # 1 is FP, 2 is BP, 3 is FP+BP
        self.op_code = op_code   
        self.working_layer_arr = self.feature_arr[self.sta_lidx:self.end_lidx]
        self.virtual_layer = HookLayer()
        self.features = nn.Sequential(*([self.virtual_layer]+self.working_layer_arr))

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

    def forward(self, sta_input):
        out = sta_input
        cnt = 0
        #print("forwarding... sta=", self.sta_lidx, "  end_lidx=", self.end_lidx)
        for layer in self.features:
            cnt += 1
            out = layer(out)
            #print(type(layer), "size:", out.size())

        if self.end_lidx == -1 or self.end_lidx == self.conv_layer_num:
            #print("fc")    
            out = out.view(out.size(0), -1)
            out = self.fc_layers(out)
            out = self.classifier(out)
        #print("forward fin")
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
                layers += [nn.ReLU(inplace=True)] #try False
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return layers        


class myVGGconv(nn.Module):
    def __init__(self, vgg_name):
        super(myVGGconv, self).__init__()
        self.feature_arr = self._make_layers2(cfg[vgg_name]) 
        self.virtual_layer = HookLayer()
        self.features = nn.Sequential(*([self.virtual_layer]+self.feature_arr))

    def forward(self, sta_input):
        out = sta_input
        cnt = 0
        
        for layer in self.features:
            cnt += 1
            out = layer(out)
            #print("Layer=",cnt, "\t", type(layer))
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



class myVGGfc(nn.Module):
    def __init__(self, vgg_name):
        super(myVGGfc, self).__init__()
        self.virtual_layer = HookLayer()
        input_dim = 512*7*7
        output_dim = 4096
        class_num = 1000
        self.fc_layers = nn.Sequential(self.virtual_layer, nn.Linear(input_dim, output_dim),nn.Linear(output_dim, output_dim),nn.Linear(output_dim, output_dim))
        self.classifier = nn.Linear(output_dim, class_num)


    def forward(self, sta_input):
        out = sta_input  
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        out = self.classifier(out)
        #print("forward fin")
        return out

class FCLayer(nn.Module):
    def __init__(self, vgg_name):
        super(FCLayer, self).__init__()
        self.virtual_layer = HookLayer()
        output_dim = 4096
        self.fc_layers = nn.Sequential( nn.Linear(output_dim, output_dim),nn.Linear(output_dim, output_dim))
        self.classifier = nn.Linear(output_dim, class_num)


    def forward(self, sta_input):
        out = sta_input  
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        out = self.classifier(out)
        #print("forward fin")
        return out

