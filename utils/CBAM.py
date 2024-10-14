# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, rotio=16):
#         super(ChannelAttention, self).__init__()
#         self.rotio = rotio
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.max_pool = nn.AdaptiveMaxPool3d(1)
#         self.sharedMLP = nn.Sequential(
#             nn.Conv3d(in_planes, in_planes // self.rotio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv3d(in_planes // self.rotio, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = self.sharedMLP(self.avg_pool(x))
#         maxout = self.sharedMLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, t, h, w = x.size()
        x = self.conv(x)
        y = self.max_pool(x)
        out = y + x
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.avg_pool3d = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv = nn.Conv3d(2 * in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = self.max_pool3d(x)
        avg_pool = self.avg_pool3d(x)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes)
        self.spatial_attention = SpatialAttention(in_planes)

    def forward(self, x):
        out = self.channel_attention(x)*x
        out = self.spatial_attention(out)*out
        return out


if __name__ == '__main__':
    img = torch.randn(4, 192, 8, 14, 14)
    print(img.shape)
    net = CBAM(192)
    output = net(img)
    print(output.shape)
