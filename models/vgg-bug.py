# -*- coding: utf-8 -*
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import json

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

  
# Inherit from Function
class HookFunc(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(self, input):
        return input.view_as(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        print("Come ")
        return grad_output


class HookLayer(torch.nn.Module):
    def __init__(self):
        super(HookLayer, self).__init__()

    def forward(self, input):
        # See the autograd section for explanation of what happens here
        # 从此开始，维持和F.linear形参一致，更具体的是跟LinearFunction里面的forward函数的形参一致
        return HookFunc.apply(input)




def grad_hook(x):
    return HookFunc()(x)

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * 1.0)

def grad_reverse(x):
    return GradReverse()(x)

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        #self.features = self._make_layers(cfg[vgg_name])
        self.layers = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)  #one FC layer

    def forward(self, x):
        #out = self.features(x)
        #print("layers num =%d" %(len(self.layers)))
        out = x
        #NCHW
        #print("Input")
        #print(out.size())
        for ll in self.layers:
            #print("in:"+str(out.size()))
            #print("layer: "+str(ll))
            #grad_hook(out)
            out = ll(out)
            #for name, param in ll.named_parameters():
            #    print(name)
            #print("out dim =%d" %(out.dim()))
            #print("out:"+str(out.size()))
        #print("out:"+str(out.size())+"\t"+str(out.size(0)))
        #out = out.view(out.size(0), -1)  #the size -1 is inferred from other 
        #print("out:"+str(out.size()))
        #raw_input("...")
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
        #for ll in layers:
        #    print(ll)
        #    raw_input("...")
        #return nn.Sequential(*layers)
        return layers


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
