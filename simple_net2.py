# -*- coding: utf-8 -*
#覆写forward和backward，但是继承之前的conv module，在里面调用，但是
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import torch.nn as nn

ori = torch.linspace(-1, 1, 20)
for i in range(0, len(ori)):
  ori[i]= i*0.05
#print(ori)
x = torch.unsqueeze(ori, dim=1) # 将1维的数据转换为2维数据 
y = x.pow(2) + 0.2 
#* torch.rand(x.size()) 
#print(x)
#print(y)
# 将tensor置入Variable中 
x, y = Variable(x), Variable(y) 
  
# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        #print(input.view(1, -1))
        #print(weight.view(1,-1))
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        print("grad_output", grad_output.size(), grad_output)
        print("ctx.saved_tensors", ctx.saved_tensors)
        raw_input("...\n")

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight) #点乘
            print("grad_input:")
            #print(grad_input.view(1,-1))
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            print("grad_weight:")
            #print(grad_weight.view(1,-1))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            print("grad_bias")
            #print(grad_bias.view(1,-1))
        #print("hhh", grad_bias)
        return grad_input, grad_weight, grad_bias

class MyLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(MyLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        self.weight.data = torch.ones_like(self.weight.data)*0.01
        print("weight", self.weight)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
            self.bias.data = torch.ones_like(self.bias)*0.2
            print("bias",self.bias)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        #从此开始，维持和F.linear形参一致，更具体的是跟LinearFunction里面的forward函数的形参一致
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )



in_dim = 5
out_dim = 2
linear_layer =  MyLinear(in_dim, out_dim)
l2 = torch.nn.Linear(in_dim, out_dim) # 输出层线性输出 


  