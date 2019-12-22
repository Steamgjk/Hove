'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F



# Inherit from Function
class GoogleNetHookFunc(torch.autograd.Function):
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
        GoogleNetHookFunc.backward_ctx = grad_output
        #print("ctx size = ", ResNetHookFunc.backward_ctx.size())
        return grad_output


class GoogleNetHookLayer(torch.nn.Module):
    def __init__(self, pipe_rank = -1):
        super(GoogleNetHookLayer, self).__init__()
        self.pipe_rank = pipe_rank

    def forward(self, input):
        # See the autograd section for explanation of what happens here
        #从此开始，维持和F.linear形参一致，更具体的是跟LinearFunction里面的forward函数的形参一致
        #print("HookLayer forward called:", self.layer_idx)
        return GoogleNetHookFunc.apply(input)
        #exec('return HookFunc_{}.apply(input)'.format(self.pipe_rank))



class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, sta_lidx = -1,ed_lidx = -1):
        super(GoogLeNet, self).__init__()



        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                            
        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 1000)
        #self.linear = nn.Linear(2458624, 1000)

        self.features_arr = [self.pre_layers,self.a3, self.b3, self.maxpool,self.a4,self.b4,self.c4,self.d4,self.e4, self.maxpool, self.a5, self.b5,self.avgpool]
        self.conv_num = len(self.features_arr)
        self.sta_lidx = 0
        self.ed_lidx = self.conv_num
        if sta_lidx > -1:
            self.sta_lidx = sta_lidx
        if ed_lidx > -1:
            self.ed_lidx = ed_lidx
        self.work_arr = self.features_arr[self.sta_lidx:self.ed_lidx]
        self.virtual_layer = GoogleNetHookLayer()
        self.features = nn.Sequential(*([self.virtual_layer]+self.work_arr))



    def forward(self, x):
        '''
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        '''
        out = x
        for layer in self.features:
            #print("layer_type=",type(layer))
            out = layer(out)
        
        if self.ed_lidx == self.conv_num:
            #print("1-outsize=",out.size())
            out = out.view(out.size(0), -1)
            #print("out size=",out.size())
            out = self.linear(out)
        return out



class GoogLeNetConv(nn.Module):
    def __init__(self, sta_lidx = -1,ed_lidx = -1):
        super(GoogLeNetConv, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                            
        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.features_arr = [self.pre_layers,self.a3, self.b3, self.maxpool,self.a4,self.b4,self.c4,self.d4,self.e4, self.maxpool, self.a5, self.b5,self.avgpool]
        self.virtual_layer = GoogleNetHookLayer()
        self.features = nn.Sequential(*([self.virtual_layer]+self.features_arr))



    def forward(self, x):
        out = x
        for layer in self.features:
            #print("layer_type=",type(layer))
            out = layer(out)
            #print("layer type=", type(layer))
            #print("out size = ", out.size())
        
        return out



class GoogLeNetFC(nn.Module):
    def __init__(self, sta_lidx = -1,ed_lidx = -1):
        super(GoogLeNetFC, self).__init__()
        self.virtual_layer = GoogleNetHookLayer()
        self.linear = nn.Linear(1024, 1000)
        #self.linear = nn.Linear(2458624, 1000)

    def forward(self, x):
        out = self.virtual_layer(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out





def test():
    net = GoogLeNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

#test()
