from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torch.autograd import Function

        
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=True, bn=False, dilation=1, groups=1):
        super(Conv2d, self).__init__()
        padding = (int((kernel_size - 1) / 2) + dilation-1) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups = groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class new_disc(nn.Module):
    def __init__(self,Nch):
        super(new_disc,self).__init__()        
        self.conv1 = Conv2d(Nch, 64, kernel_size=1, stride=1)
        self.conv2 = Conv2d(64,64,kernel_size=3,stride=1)
        self.conv3 = Conv2d(64,64,kernel_size=3,stride=1)
        self.conv4 = Conv2d(64, 3,kernel_size=3,stride=1,relu=False)
     
        
    def forward(self, x):
        x=grad_reverse(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        
        return x

class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)


class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop)
        return x,label


class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(2048, 512)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(256,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))
        label = self.LabelResizeLayer(x, need_backprop)
        return x,label

class InstanceLabelResizeLayer(nn.Module):


    def __init__(self):
        super(InstanceLabelResizeLayer, self).__init__()
        self.minibatch=256

    def forward(self, x,need_backprop):
        feats = x.data.cpu().numpy()
        lbs = need_backprop.data.cpu().numpy()
        resized_lbs = np.ones((feats.shape[0], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            resized_lbs[i*self.minibatch:(i+1)*self.minibatch] = lbs[i]

        y=torch.from_numpy(resized_lbs).cuda()

        return y


