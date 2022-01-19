#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import dataset
from resnet.resnet50 import resnet50
from mindspore import load_checkpoint,save_checkpoint,load_param_into_net

#此函数先不进行修改
"""
def weight_init(module):
    for n, m in module.cells():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.PReLU):
            pass
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()
"""

"""
class Bottleneck(nn.Cell):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               has_bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def construct(self, x):
        relu = nn.ReLU()
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return relu(out + x)


class ResNet(nn.Cell):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.SequentialCell([nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4)])
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.SequentialCell(*layers)

    def construct(self, x):
        relu = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=3,stride=2,pad_mode="valid")
        out0 = relu(self.bn1(self.conv1(x)))
        out1 = maxpool(out0)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5
    
    
    
#####未考察
    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)
"""
"""
class ResNet(nn.Cell):
    def __init__(self):
        super(ResNet, self).__init__()
        net = resnet50()
        # 加载预训练模型
        param_dict = load_checkpoint('resnet/resnet50.ckpt')
        # 给网络加载参数
        load_param_into_net(net,param_dict)
        self.mynet = net
"""

class MM(nn.Cell):
    def __init__(self):
        super(MM, self).__init__()
        relu = nn.ReLU()
        self.conv1 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,pad_mode="pad"), nn.BatchNorm2d(64),
                                   nn.ReLU()])
        self.conv2 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,pad_mode="pad"), nn.BatchNorm2d(64),
                                   nn.ReLU()])
        self.conv3 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2,pad_mode="pad"),
                                   nn.BatchNorm2d(64), nn.ReLU()])
        self.conv4 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4, padding=4,pad_mode="pad"),
                                   nn.BatchNorm2d(64), nn.ReLU()])
        self.conv5 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=8, padding=8,pad_mode="pad"),
                                   nn.BatchNorm2d(64), nn.ReLU()])

######未考察
#    def initialize(self):
#        weight_init(self)

    def construct(self, out1, out2):
        resize_bilinear = nn.ResizeBilinear()
        avgpool = nn.AvgPool2d(kernel_size=352//4)
        x = self.conv1(out1 + resize_bilinear(out2, size=out1.shape[2:]))
        out = x
        out = avgpool(x)
        # print( out.size())
        out = out * x
        # print( out.size())

        out = self.conv2(out) + self.conv3(out) + self.conv4(out) + self.conv5(out)
        return out


class NN(nn.Cell):
    def __init__(self):
        super(NN, self).__init__()
        self.conv_l2 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, padding=2,dilation=2,pad_mode="pad"),
                                     nn.BatchNorm2d(64), nn.ReLU()])
        self.conv_l3 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, padding=4,dilation=4,pad_mode="pad"),
                                     nn.BatchNorm2d(64), nn.ReLU()])
        self.conv_l4 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, padding=6,dilation=6,pad_mode="pad"),
                                     nn.BatchNorm2d(64), nn.ReLU()])
        self.conv_l5 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, padding=8,dilation=8,pad_mode="pad"),
                                     nn.BatchNorm2d(64), nn.ReLU()])

#    def initialize(self):
#        weight_init(self)

    def construct(self, input):
        resize_bilinear = nn.ResizeBilinear()
        out5 = self.conv_l5(input[0])
        out4 = self.conv_l4(
            input[1] + resize_bilinear(out5, size=input[1].shape[2:],align_corners=True))
        out3 = self.conv_l3(
            input[2] + resize_bilinear(out4, size=input[2].shape[2:],align_corners=True))
        out2 = self.conv_l2(input[3] + resize_bilinear(out3,size=input[3].shape[2:],align_corners=True))

        return (out2, out3, out4, out5)


class WW(nn.Cell):
    def __init__(self):
        super(WW, self).__init__()
        self.conv12 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1,pad_mode="pad"), nn.BatchNorm2d(64),
                                    nn.ReLU()])
        self.conv13 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1,pad_mode="pad"), nn.BatchNorm2d(64),
                                    nn.ReLU()])
        self.conv14 = nn.SequentialCell([nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1,pad_mode="pad"), nn.BatchNorm2d(64),
                                    nn.ReLU()])
####未考察
#    def initialize(self):
#        weight_init(self)

    def construct(self, edg, input):
        resize_bilinear = nn.ResizeBilinear()
        out12 = self.conv12(edg + resize_bilinear(input[0], size=edg.shape[2:],align_corners=True))
        out13 = self.conv13(
            edg + resize_bilinear(input[1], size=edg.shape[2:],align_corners=True) + out12)
        out14 = self.conv14(edg+resize_bilinear(input[2],size=edg.shape[2:],align_corners=True)+out13)

        return (out12, out13,out14)


class FDDE(nn.Cell):
    def __init__(self, cfg):
        super(FDDE, self).__init__()
        self.cfg = cfg

        net = resnet50()
        # 加载预训练模型
        param_dict = load_checkpoint('resnet/resnet50.ckpt')
        # 给网络加载参数
        load_param_into_net(net,param_dict)

        self.bkbone = net
        self.conv5 = nn.SequentialCell(nn.Conv2d(2048, 64, kernel_size=1,pad_mode="valid"),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1,pad_mode="pad"),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.SequentialCell(nn.Conv2d(1024, 64, kernel_size=1,pad_mode="valid"),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1,pad_mode="pad"),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.SequentialCell(nn.Conv2d(512, 64, kernel_size=1,pad_mode="valid"),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1,pad_mode="pad"),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.SequentialCell(nn.Conv2d(256, 64, kernel_size=1,pad_mode="valid"),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1,pad_mode="pad"),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.conv1 = nn.SequentialCell(nn.Conv2d(64, 64, kernel_size=1,pad_mode="valid"), 
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1,pad_mode="pad"),
                                   nn.BatchNorm2d(64), nn.ReLU())

        self.convf = nn.SequentialCell(nn.Conv2d(128, 128, kernel_size=3, padding=1,pad_mode="pad"),
                                   nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1,pad_mode="valid"))
        self.conv_edg = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1,pad_mode="valid")

        self.NN = NN()
        self.MM = MM()
        self.WW = WW()

        
#        self.initialize()

####未考察
#    def initialize(self):
#        if self.cfg.snapshot:
#            self.load_state_dict(torch.load(self.cfg.snapshot))
#        else:
#            weight_init(self)

    def construct(self, x, shape=None):
        resize_bilinear = nn.ResizeBilinear()
        conc = ops.Concat(axis = 1)
        if shape is None:
            shape = x.shape[2:]
        out1, out2, out3, out4, out5 = self.bkbone(x)
        out1, out2, out3, out4, out5 = self.conv1(out1),self.conv2(out2), self.conv3(out3), self.conv4(out4), self.conv5(out5)

        (out_l2,out_l3, out_l4, out_l5) = self.NN([out5, out4, out3,out2])

        out_l1 = self.MM(out1, out_l2)
        (out12,out13, out14) = self.WW(edg=out_l1, input=[out_l2, out_l3, out_l4])

        out14_5 = out14 + resize_bilinear(out5, size=out14.shape[2:],align_corners=True)
        #out14_5 = resize_bilinear(out14_5, size=out_l1.shape[2:],align_corners=True)
        out = self.convf(conc((out_l1,out14_5)))

        edg = self.conv_edg(out_l1)
        edg = resize_bilinear(edg, size=shape,align_corners=True)
        out = resize_bilinear(out, size=shape,align_corners=True)
        return edg, out

"""
#####未考察
if __name__ == '__main__':
    cfg = dataset.Config(datapath='../data/DUTS', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9,
                         decay=5e-4, epoch=40)
    net = FDDE(cfg)
    im = torch.rand(size=(1, 3, 352, 352))
    # edg, out = net(im)
    list = net(im)


    total_num = sum(p.numel() for p in net.parameters())/1e6
    train_num = sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6
    print(total_num)
    print(train_num)
"""


