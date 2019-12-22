# -*- coding: utf-8 -*
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.init as init
import os

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
        print("input sz=",input.size())
        return input.view_as(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):
        #pid = os.getpid()
        #print("pid = ", pid)
        #HookFunc.hook_dict[pid] = grad_output
        #print("grad_output.sz=",grad_output.size())
        HookFunc.backward_ctx = grad_output
        print("ctx size = ", HookFunc.backward_ctx.size())
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
