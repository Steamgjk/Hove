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
print(x)
print(y)
# 将tensor置入Variable中 
x, y = Variable(x), Variable(y) 
  
#plt.scatter(x.data.numpy(), y.data.numpy()) 
#plt.show() 

class MyReLUFunc(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    """
 In the forward pass we receive a context object and a Tensor containing the
 input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.
 """
    ctx.save_for_backward(x)
    return x.clamp(min=0)
  def backward(ctx, grad_output):
    """
 In the backward pass we receive the context object and a Tensor containing
    the gradient of the loss with respect to the output produced during the
    forward pass. We can retrieve cached data from the context object, and must
    compute and return the gradient of the loss with respect to the input to the
    forward function.
 """
    x, = ctx.saved_tensors
    grad_x = grad_output.clone()
    grad_x[x < 0] = 0
    return grad_x
class MyReLUMdl(torch.nn.Module):
    def __init__(self):
      super(MyReLUMdl, self).__init__()
    def forward(self, input):
      return MyReLUFunc.apply(input)

# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        #print(input.view(1, -1))
        print(weight.view(1,-1))
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        print("Linear:",grad_output)
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight) #点乘
            print("grad_input:")
            #print(grad_input.view(1,-1))
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            print("grad_input:")
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
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

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
        print("Come grad_out:",grad_output)
        return grad_output


class HookLayer(torch.nn.Module):
    def __init__(self):
        super(HookLayer, self).__init__()

    def forward(self, input):
        # See the autograd section for explanation of what happens here
        # 从此开始，维持和F.linear形参一致，更具体的是跟LinearFunction里面的forward函数的形参一致
        return HookFunc.apply(input)


# 定义一个构建神经网络的类 
class Net(torch.nn.Module): # 继承torch.nn.Module类 
  def __init__(self, n_feature, n_hidden, n_output): 
    super(Net, self).__init__() # 获得Net类的超类（父类）的构造方法 
    # 定义神经网络的每层结构形式 
    # 各个层的信息都是Net类对象的属性 
    #self.hidden = torch.nn.Linear(n_feature, n_hidden) # 隐藏层线性输出 
    self.hidden = MyLinear(n_feature, n_hidden) # 隐藏层线性输出 

    #self.relu1 = torch.nn.ReLU(inplace=True)
    #self.hidden2 = torch.nn.Linear(n_hidden, n_hidden) # 隐藏层线性输出 
    #self.relu2 = torch.nn.ReLU(inplace=True)
    self.hook_layer = HookLayer()

    #self.predict = torch.nn.Linear(n_hidden, n_output) # 输出层线性输出 
    self.predict = MyLinear(n_hidden, n_output) # 输出层线性输出 
  
  # 将各层的神经元搭建成完整的神经网络的前向通路 
  def forward(self, x): 
    x = self.hidden(x)
    print(x)
    x = self.hook_layer(x)
    #print("after hook ", x)
    #x = F.relu(x) # 对隐藏层的输出进行relu激活
    #x = self.relu1(x) 
    #x = self.hidden2(x)
    #x = F.relu(x)
    #x = self.relu2(x)
    x = self.predict(x) 
    return x 
  
# 定义神经网络 
net = Net(1, 5, 1) 
print(net) # 打印输出net的结构 
  
# 定义优化器和损失函数 
optimizer = torch.optim.SGD(net.parameters(), lr=0.5) # 传入网络参数和学习率 
loss_function = torch.nn.MSELoss() # 最小均方误差 
  
# 神经网络训练过程 
plt.ion()  # 动态学习过程展示 
plt.show() 
  
for t in range(300): 
  prediction = net(x) # 把数据x喂给net，输出预测值 
  #print("Prediction:")
  #print(prediction)
  loss = loss_function(prediction, y) # 计算两者的误差，要注意两个参数的顺序 
  print("loss=%.3f" %(loss.data) )
  optimizer.zero_grad() # 清空上一步的更新参数值 
  loss.backward() # 误差反相传播，计算新的更新参数值 
  optimizer.step() # 将计算得到的更新值赋给net.parameters() 
  raw_input("...\n")
  