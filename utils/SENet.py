# -*- coding: utf-8 -*-
from functools import reduce

import torch
from torch import nn

import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SENet(nn.Module):
    def __init__(self, num_classes=400, reduction=16):
        super(SENet, self).__init__()
        self.conv1 = nn.Conv3d(64, 64, kernel_size=(1, 7, 7), stride=2, padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(64, reduction=reduction)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.se(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    img = torch.randn(1, 64, 16, 512, 512)
    model = SENet(400)
    out = model(img)
    # criterion = nn.L1Loss()
    # loss = criterion(out, img)
    # loss.backward()
    print("out shape:{}".format(out.shape))
    # print('loss value:{}'.format(loss))
