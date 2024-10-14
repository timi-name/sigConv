# -*- coding: utf-8 -*-
'''
This file is modified from:
https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import sys


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x, is_pad=True):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        if is_pad:
            x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._out_channels,
                                kernel_size=self._kernel_size,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._out_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_size[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_size[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class TemporalInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name='temporal', kernel_size=3):
        super(TemporalInceptionModule, self).__init__()

        self.pool = MaxPool3dSamePadding(kernel_size=(kernel_size, 1, 1), stride=(1, 1, 1), padding=0)

        self.conv1 = Unit3D(in_channels=in_channels, out_channels=out_channels[0], kernel_size=(kernel_size, 1, 1), padding=0)

        self.conv2a_1_2 = Unit3D(in_channels=in_channels, out_channels=out_channels[1], kernel_size=(kernel_size, 1, 1), padding=0)
        self.conv2b_1_2 = Unit3D(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(kernel_size, 1, 1), padding=0)
        self.conv2a__2 = Unit3D(in_channels=in_channels, out_channels=out_channels[2], kernel_size=(kernel_size, 1, 1), padding=0)

        self.conv4a_3_4 = Unit3D(in_channels=in_channels, out_channels=out_channels[3], kernel_size=(kernel_size, 1, 1), padding=0)
        self.conv4b_3_4 = Unit3D(in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=(kernel_size, 1, 1), padding=0)
        self.conv4a_4_4 = Unit3D(in_channels=in_channels, out_channels=out_channels[4], kernel_size=(kernel_size, 1, 1), padding=0)
        self.conv4b_4_4 = Unit3D(in_channels=out_channels[4], out_channels=out_channels[4], kernel_size=(kernel_size, 1, 1), padding=0)

        self.conv6 = Unit3D(in_channels=in_channels, out_channels=out_channels[5], kernel_size=(kernel_size, 1, 1), padding=0)

        # self.conv_2 = nn.Conv3d(in_channels=2*out_channels[2], out_channels=out_channels[2], kernel_size=(1, 1, 1), padding=0)
        # self.conv_4 = nn.Conv3d(in_channels=2*out_channels[4], out_channels=out_channels[4], kernel_size=(1, 1, 1), padding=0)

        self.conv_x = nn.Conv3d(in_channels=in_channels, out_channels=out_channels[0]+out_channels[2]+out_channels[4]+out_channels[5], kernel_size=(1, 1, 1), padding=0)

        self.name = name

    def forward(self, x):
        residual = x

        out0 = self.conv1(x)  # (b, 512, t, w, h) -> (b, out_channels[0], t, w, h)

        out2_1_2_1x1 = self.conv2b_1_2(self.conv2a_1_2(x))
        out2_1_2_3x3 = self.conv2a__2(x)
        # out2 = self.conv_2(torch.cat([out2_1_2_1x1, out2_1_2_3x3], dim=1))
        out2 = self.pool(out2_1_2_1x1 + out2_1_2_3x3)

        out4_3_4_1x1 = self.conv4b_3_4(self.conv4a_3_4(x))
        out4_4_4_3x3 = self.conv4b_4_4(self.conv4a_4_4(x))
        # out4 = self.conv_4(torch.cat([out4_3_4_1x1, out4_4_4_3x3], dim=1))
        out4 = self.pool(out4_3_4_1x1 + out4_4_4_3x3)

        out5 = self.conv6(x)

        return torch.cat([out0, out2, out4, out5], dim=1) + self.conv_x(residual)


class SpatialInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(SpatialInceptionModule, self).__init__()

        self.pool = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)

        self.conv1 = Unit3D(in_channels=in_channels, out_channels=out_channels[0], kernel_size=(1, 1, 1), padding=0, name=name + '/Branch_0/Conv3d_conv1_1x1x1')

        self.conv2a_1_2 = Unit3D(in_channels=in_channels, out_channels=out_channels[1], kernel_size=(1, 1, 1), padding=0, name=name + '/Branch_0/Conv3d_conv2_1x1x1')
        self.conv2b_1_2 = Unit3D(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(1, 3, 3), padding=0, name=name + '/Branch_0/Conv3d_conv3a_1x3x3')
        self.conv2a__2 = Unit3D(in_channels=in_channels, out_channels=out_channels[2], kernel_size=(1, 3, 3), padding=0, name=name + '/Branch_0/Conv3d_conv3b_1x3x3')

        self.conv4a_3_4 = Unit3D(in_channels=in_channels, out_channels=out_channels[3], kernel_size=(1, 1, 1), padding=0, name=name + '/Branch_0/Conv3d_conv4a_1x1x1')
        self.conv4b_3_4 = Unit3D(in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=(1, 1, 1), padding=0, name=name + '/Branch_0/Conv3d_conv4b_1x1x1')
        self.conv4a_4_4 = Unit3D(in_channels=in_channels, out_channels=out_channels[4], kernel_size=(1, 3, 3), padding=0, name=name + '/Branch_0/Conv3d_conv5a_1x3x3')
        self.conv4b_4_4 = Unit3D(in_channels=out_channels[4], out_channels=out_channels[4], kernel_size=(1, 3, 3), padding=0, name=name + '/Branch_0/Conv3d_conv5b_1x3x3')

        self.conv6 = Unit3D(in_channels=in_channels, out_channels=out_channels[5], kernel_size=(1, 3, 3), padding=0, name=name + '/Branch_0/Conv3d_conv6_1x3x3')

        self.conv_2 = nn.Conv3d(in_channels=2*out_channels[2], out_channels=out_channels[2], kernel_size=(1, 1, 1), padding=0)
        self.conv_4 = nn.Conv3d(in_channels=2*out_channels[4], out_channels=out_channels[4], kernel_size=(1, 1, 1), padding=0)
        self.conv_x = nn.Conv3d(in_channels=in_channels, out_channels=out_channels[0]+out_channels[2]+out_channels[4]+out_channels[5], kernel_size=(1, 1, 1), padding=0)
        self.name = name

    def forward(self, x):
        residual = x

        out0 = self.conv1(x)    # (b, 512, t, w, h) -> (b, out_channels[0], t, w, h)

        out2_1_2_1x1 = self.conv2b_1_2(self.conv2a_1_2(x))
        out2_1_2_3x3 = self.conv2a__2(x)
        out2 = self.pool(self.conv_2(torch.cat([out2_1_2_1x1, out2_1_2_3x3], dim=1)))

        out4_3_4_1x1 = self.conv4b_3_4(self.conv4a_3_4(x))   # (b, out_channels[3], t, w, h) -> (b, out_channels[3], t, w, h)
        out4_4_4_3x3 = self.conv4b_4_4(self.conv4a_4_4(x))    # (b, out_channels[4], t, w, h) -> (b, out_channels[4], 3t, w, h)
        out4 = self.pool(self.conv_4(torch.cat([out4_3_4_1x1, out4_4_4_3x3], dim=1)))

        out5 = self.conv6(x)    # (4, out_channels[4], t, 1, 1) -> (4, out_channels[5], t, 1, 1 )

        # out0 = self.pool(self.conv1(x))    # (b, 512, t, w, h) -> (b, out_channels[0], t, w, h)
        #
        # out2_1_2_1x1 = self.conv2b_1_2(self.conv2a_1_2(x))
        # out2_1_2_3x3 = self.conv2a__2(x)
        # out2 = self.pool(self.conv_2(torch.cat([out2_1_2_1x1, out2_1_2_3x3], dim=1)))
        #
        # out4_3_4_1x1 = self.conv4b_3_4(self.conv4a_3_4(x))   # (b, out_channels[3], t, w, h) -> (b, out_channels[3], t, w, h)
        # out4_4_4_3x3 = self.conv4b_4_4(self.conv4a_4_4(x))    # (b, out_channels[4], t, w, h) -> (b, out_channels[4], 3t, w, h)
        # out4 = self.pool(self.conv_4(torch.cat([out4_3_4_1x1, out4_4_4_3x3], dim=1)))

        # out5 = self.conv6(x)    # (4, out_channels[4], t, 1, 1) -> (4, out_channels[5], t, 1, 1 )

        return torch.cat([out0, out2, out4, out5], dim=1) + self.conv_x(residual)



if __name__ == "__main__":
    inputs = torch.rand(4, 192, 16, 14, 14)    # 1 表示批次大小（batch size），这里有一个样本。

    net = SpatialInceptionModule(512, [160, 112, 224, 24, 64, 64], name="c3d")
    print('Total params: %.4fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))

    output = net(inputs)
    print(inputs.size())
    print(output.size())
