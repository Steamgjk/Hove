# -*- coding: utf-8 -*
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.init as init
import os
from myVGG import HookLayer
from myVGG import HookFunc

class myAlexNet(nn.Module):
    def __init__(self, vgg_name, sta_lidx = -1, end_lidx=-1, op_code = -1):
        super(myAlexNet, self).__init__()
        
        self.feature_arr = [nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1),nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2)]
        self.conv_layer_num = len(self.feature_arr)
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

        class_num = 1000
        if self.end_lidx == self.conv_layer_num:
            self.fc_layers = nn.Sequential(nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 4096),nn.ReLU(inplace=True),)
            self.classifier = nn.Linear(4096, class_num)

    def forward(self, sta_input):
        out = sta_input
        cnt = 0
        #print("forwarding... sta=", self.sta_lidx, "  end_lidx=", self.end_lidx)
        for layer in self.features:
            cnt += 1
            out = layer(out)
            #print(type(layer), "size:", out.size())
        #print("self.end_lidx=",self.end_lidx, "  cnv=",self.conv_layer_num)
        if self.end_lidx == -1 or self.end_lidx == self.conv_layer_num:
            #print("fc")    
            out = out.view(out.size(0), -1)
            out = self.fc_layers(out)
            out = self.classifier(out)
        return out     


class myAlexNetconv(nn.Module):
    def __init__(self, vgg_name):
        super(myAlexNetconv, self).__init__()
        self.feature_arr = [nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1),nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2)]
        self.virtual_layer = HookLayer()
        self.features = nn.Sequential(*([self.virtual_layer]+self.feature_arr))

    def forward(self, sta_input):
        out = sta_input
        cnt = 0
        for layer in self.features:
            cnt += 1
            out = layer(out)
            #print("Layer=",cnt, "\t", type(layer))
        #print("out sz =",out.size())
        return out


class myAlexNetfc(nn.Module):
    def __init__(self, vgg_name):
        super(myAlexNetfc, self).__init__()
        self.virtual_layer = HookLayer()
        class_num = 1000
        self.fc_layers = nn.Sequential(self.virtual_layer, nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 4096),nn.ReLU(inplace=True),)
        self.classifier = nn.Linear(4096, class_num)

    def forward(self, sta_input):
        #print("sta_input sz =", sta_input.size())
        out = sta_input  
        out = out.view(out.size(0), 256 * 6 * 6)
        out = self.fc_layers(out)
        out = self.classifier(out)
        #print("forward fin")
        return out
