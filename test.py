# -*- coding: utf-8 -*
#覆写forward和backward，但是继承之前的conv module，在里面调用，但是
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import torch.nn as nn

grad_list = []
def print_grad(grad):
	print("Coming",grad)
	grad_list.append(grad)
dim1= 1
dim2 = 5
x_tensor = torch.zeros(dim1,dim2)
y_tensor = torch.zeros(dim1,dim2)
for i in range(dim1):
	for j in range(dim2):
		x_tensor[i,j]=i*dim2 + j

x = Variable(x_tensor, requires_grad = True)
y = x**2
z = y*2

#z=2*x^2
print(x.data)
print(y.data)
print(z.data)
x.register_hook(print_grad)
y.register_hook(print_grad)
z.register_hook(print_grad)
#s=1/N (2y1+2y2+)
p=z
s = z.mean()

print(s.grad)
print("ppp", p.grad_fn)
print(s.grad_fn)
print(z.grad_fn)
print(y.grad_fn)
print(x.grad_fn)
s.backward(torch.ones(1))
print(grad_list)
print(x.grad)

weights = torch.ones(dim1,dim2)
weights = 1.0 / (dim1*dim2) * weights

#print(weights)
#z.backward(weights)
#print(y.requires_grad)
#print(y.grad)
#print(x.grad)
#s.backward()
#print(x.grad)

