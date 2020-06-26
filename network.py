# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from modules.bn import InPlaceABNSync as BatchNorm2d
from resnet import Resnet18
from torch.nn.parameter import Parameter

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
            self.relu = nn.ReLU(inplace=inplace)
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
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 16):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class SE(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes
        self.init_weight()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                print(ly)
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

def fixed_padding(inputs, kernel_size, dilation=1):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = nn.functional.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes,kernel_size=3, stride=1,  bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0,groups=inplanes, bias=bias)
        self.bn = BatchNorm2d(inplanes, activation='none')
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class sdBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_eps=1e-5,  stride=1, downsample=None):
        super(sdBlock, self).__init__()
        # if planes != inplanes:
        #    self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
        #    self.skipbn = BatchNorm2d(planes)
        # else:
        #    self.skip = None
        self.conv1 = SeparableConv2d(inplanes, planes, 3, 1)
        self.bn1 = BatchNorm2d(planes, activation='none')
        self.relu = nn.ReLU()
        self.conv2 = SeparableConv2d(planes, planes, 3, 1)
        self.bn2 = BatchNorm2d(planes, activation='none')
        # self.sge    = SpatialGroupEnhance()
        self.init_weight()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.sge(out)
        out = self.relu(out)
        # if self.skip is not None:
        #     skip = self.skip(x)
        #     skip = self.skipbn(skip)
        # else:
        #     skip = x
        return out 
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_eps=1e-5,  stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes,  activation='none')
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, activation='none')
        self.init_weight()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class GlobalContext(nn.Module):
    def __init__(self, inc, outc, *args, **kwargs):
        super(GlobalContext, self).__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.conv_global = ConvBnRelu(inc, outc, 1, 1, 0) #all global context 512->128
        self.conv_last = ConvBnRelu(inc, outc, 1, 1, 0)
        self.se_channel_att = SE(outc, outc)
        self.init_weight()

    def forward(self, x):
        last = x
        out = self.global_avg(x)
        out = self.conv_global(out)
        out = F.interpolate(out,size=x.size()[2:],mode='bilinear', align_corners=True)
        last = self.conv_last(last)
        glob = out + last
        se = self.se_channel_att(glob).expand_as(glob)
        glob = glob * se
        return glob

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        # print("-----------------------------global")
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

class KeepRes(nn.Module):
    def __init__(self, inc, outc, *args, **kwargs):
        super(KeepRes, self).__init__()
        self.block1 = sdBlock(inplanes=inc, planes=outc)
        self.block2 = sdBlock(inplanes=inc, planes=outc)
        self.dim_reduction=ConvBnRelu(outc, 32, 1, 1, 0)
        self.init_weight()

    def forward(self, x, global_context):
        x = torch.cat((x, global_context),1)
        highres1 = self.block1(x)
        x = torch.cat((highres1, global_context),1)
        highres2 = self.block2(x)
        highres = highres1 + highres2
        highres = self.dim_reduction(highres)
        return highres

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        # print("-----------------------------keepres")
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

class MergeAttention(nn.Module):
    def __init__(self, inc, outc, *args, **kwargs):
        super(MergeAttention, self).__init__()
        self.merge_att = SE(inc, outc)
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_se = nn.Sequential(nn.Conv2d(64, 32,kernel_size=1, stride=1, padding=0),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(outc, outc,
                                                  kernel_size=1, stride=1, padding=0),
                                        BatchNorm2d(outc))
        self.dim_reduction = ConvBnRelu(128, 32, 1, 1, 0)
        #self.dim_reduction_merge = ConvBnRelu(64, 32, 1, 1, 0)
        # self.spatial_se = nn.Sequential(nn.Conv2d(inc, 1, kernel_size=1, stride=1,
        #                                           padding=0, bias=False),
        #                                 BatchNorm2d(1))
        self.init_weight()

    def forward(self, context, highres):
        context = self.dim_reduction(context)
        merge = torch.cat((context, highres), 1)
        context_se = self.channel_se(self.ch_avg_pool(merge) + self.ch_max_pool(merge)).sigmoid().exp()
        #merge = self.dim_reduction_merge(merge)
        # sp = self.spatial_se(highres).sigmoid().exp()
        # merge = torch.mul(merge_org,context_se) 
        # hr = torch.mul(highres,sp)
        return context, context
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        # print("-----------------------------merge")
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

class OutputLayer(nn.Module):
    def __init__(self, in_chan, n_classes, *args, **kwargs):
        super(OutputLayer, self).__init__()
        self.conv = ConvBnRelu(in_chan, n_classes, 1, 1, 0,has_bn=True,has_relu=False, has_bias=False,inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

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
        self.global_context = GlobalContext(512, 128)
        self.keep_res = KeepRes(192, 64)
        self.merge_att = MergeAttention(32,32)
        self.last_conv1=OutputLayer(32, out_planes)
        self.last_conv2=OutputLayer(32, out_planes)
        self.last_conv3=OutputLayer(32, out_planes)
        self.init_weight()

    def forward(self, data, label=None):
        feat, feat32 = self.context_path(data)
        # print(feat.shape, feat32.shape)
        global_context = self.global_context(feat32)
        global_context = F.interpolate(global_context,
                                       size=feat.size()[2:],
                                       mode='bilinear', align_corners=True)

        highres = self.keep_res(feat, global_context)
        context, merge_att = self.merge_att(global_context, highres)
        co = self.last_conv1(context)
        hr = self.last_conv2(highres)
        mer = self.last_conv3(merge_att)
        out_co = F.interpolate(co,data.size()[2:],mode='bilinear', align_corners=True)
        out_hr = F.interpolate(hr,data.size()[2:],mode='bilinear', align_corners=True)
        out_mer = F.interpolate(mer,data.size()[2:],mode='bilinear', align_corners=True)
        return out_mer,out_co,out_hr
    # def init_weight(self):
    #     state_dict = torch.load('./res1024-up/model_79200.pth')
    #     state_dict = state_dict['model']
    #     self_state_dict = self.state_dict()
    #     for k, v in state_dict.items():
    #         # if 'fc' in k: continue
    #         self_state_dict.update({k: v})
    #     self.load_state_dict(self_state_dict)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (OutputLayer, MergeAttention)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

if __name__ == "__main__":
    model = mynet(19, None)
    # print(model)
