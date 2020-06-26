# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from modules.bn import InPlaceABNSync as BatchNorm2d
from resnet import Resnet18
from torch.nn.parameter import Parameter
#BatchNorm2d=nn.BatchNorm2d
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, bn_eps=1e-5,
                 has_relu=True, inplace=False, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = BatchNorm2d(out_planes, activation='none')
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.PReLU(out_planes)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, BatchNorm2d):
                ly.eps = 1e-5
                ly.momentum = 0.1
                nn.init.constant_(ly.weight, 1)

class OutputLayer(nn.Module):
    def __init__(self, in_chan, n_classes, *args, **kwargs):
        super(OutputLayer, self).__init__()
        self.drop = nn.Dropout2d(0.1)
        self.conv = ConvBnRelu(in_chan, n_classes, 1, 1, 0,has_bn=True,has_relu=False, has_bias=False,inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.drop(x)
        x = self.conv(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, BatchNorm2d):
                ly.eps = 1e-5
                ly.momentum = 0.1
                nn.init.constant_(ly.weight, 1)
                nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        # print("-----------------------------output")
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print('con: ',module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                # print('bn: ', module)
                nowd_params += list(module.parameters())
        return wd_params, nowd_params        

class mynet(nn.Module):
    def __init__(self, out_planes):
        super(mynet, self).__init__()
        self.context_path = Resnet18()
        # self.global_context = GlobalContext(512, 128)
        self.last_conv3=OutputLayer(64, out_planes)
        self.init_weight()

    def forward(self, data, label=None):
        feat, feat32 = self.context_path(data)
        print(feat.shape, feat32.shape)
        feat = self.last_conv3(feat)
        out_mer = F.interpolate(feat,data.size()[2:],mode='bilinear', align_corners=True)
        return out_mer 
 
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, BatchNorm2d):
                ly.eps = 1e-5
                ly.momentum = 0.1
                nn.init.constant_(ly.weight, 1)
                nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (OutputLayer)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
 
