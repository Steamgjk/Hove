# -*- coding: utf-8 -*
#覆写forward和backward，但是继承之前的conv module，在里面调用，但是
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import torch.nn as nn

x = torch.zeros(2,5)
y = torch.zeros(2,5)
x[0,0]