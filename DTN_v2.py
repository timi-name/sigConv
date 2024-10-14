'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import torch
from torch.autograd import Variable
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import random, math
from backbone.C3D import *
# from backbone.utils import *
from trans_module import *
from utils import uniform_sampling

import matplotlib.pyplot as plt  # For graphics
import seaborn as sns
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(123)
random.seed(123)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., apply_transform=False, knn_attention=0.7):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.cls_embed = [None for _ in range(depth)]
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, apply_transform=apply_transform, knn_attention=knn_attention)),  # LayerNorm + self-Attantion
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 前馈
            ]))

    def forward(self, x):
        for ii, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            self.cls_embed[ii] = x[:, 0]
        return x

    def get_classEmbd(self):
        return self.cls_embed


class clsToken(nn.Module):
    def __init__(self, frame_rate, inp_dim):
        super().__init__()
        self.frame_rate = frame_rate  # 帧率
        self.inp_dim = inp_dim  # 输入特征维度

        # 初始化分类标识符 (cls_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, inp_dim))
        # 初始化绝对位置嵌入
        self.abs_pos_embedding = nn.Parameter(torch.randn(1, frame_rate + 1, inp_dim))
        # 初始化相对位置嵌入
        self.rel_pos_embedding = nn.Parameter(torch.randn(2 * frame_rate - 1, inp_dim))

    def forward(self, x):
        B, N, C = x.shape  # 获取输入的批次大小(B)、序列长度(N)和特征维度(C)

        # 将分类标识符复制到与批次大小相同的数量
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        # 将分类标识符与输入x在第一个维度上拼接
        x = torch.cat((cls_token, x), dim=1)  # 形状为: (B, N+1, C)

        # 添加绝对位置嵌入
        x += self.abs_pos_embedding[:, :(N + 1)]

        # 计算相对位置嵌入

        seq_len = max(B, N) + 1  # 序列长度增加1（因为添加了cls_token）
        range_vec = torch.arange(seq_len)  # 生成一个序列范围向量
        distance_mat = range_vec[:, None] - range_vec[None, :]  # 计算距离矩阵
        # 将距离矩阵裁剪到合法范围
        distance_mat_clipped = torch.clamp(distance_mat + self.frame_rate - 1, 0, 2 * self.frame_rate - 2)
        # 使用裁剪后的距离矩阵来索引相对位置嵌入
        rel_pos_embedding = self.rel_pos_embedding[distance_mat_clipped[:B, :N+1]]

        # 添加相对位置嵌入
        x += rel_pos_embedding

        return x  # 返回添加了位置编码的输入


# class clsToken(nn.Module):
#     def __init__(self, frame_rate, inp_dim):
#         super().__init__()
#         self.frame_rate = frame_rate
#         num_patches = frame_rate
#         self.cls_token = nn.Parameter(torch.randn(1, 1, inp_dim))
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, inp_dim))
#
#     def forward(self, x):
#         B, N, C = x.shape
#         cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
#         x = torch.cat((cls_token, x), dim=1)
#         x += self.pos_embedding[:, :(N + 1)]
#         return x


class DTNNet(nn.Module):
    def __init__(self, args, num_classes=249, inp_dim=512, dim_head=64, hidden_dim=768,
                 heads=8, pool='cls', dropout=0.1, emb_dropout=0.1, mlp_dropout=0.0, branch_merge='pool',
                 init: bool = False, warmup_temp_epochs: int = 30, branchs=3, dynamic_tms=True):
        super().__init__()

        self._args = args

        print('Temporal Resolution:')
        frame_rate = args.sample_duration // args.intar_fatcer
        # names = self.__dict__
        # assert heads == frame_rate
        self.cls_tokens = nn.ModuleList([])
        dynamic_kernel = []

        # 绝对+相对位置编码：
        self.cls_tokens.append(clsToken(frame_rate, inp_dim))
        print(frame_rate)
        dynamic_kernel.append(int(frame_rate ** 0.5))
        frame_rate += args.sample_duration // args.intar_fatcer

        trans_depth = args.N
        self.multi_scale_transformers = nn.ModuleList([])  # TMS Module
        self.multi_scale_transformers.append(
            nn.ModuleList([
                TemporalInceptionModule(inp_dim, [160, 112, 224, 24, 64, 64], kernel_size=dynamic_kernel[0] if dynamic_tms else 3),  #
                MaxPool3dSamePadding(kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=0),
                Transformer(inp_dim, trans_depth, heads, dim_head, mlp_dim=hidden_dim, dropout=emb_dropout, knn_attention=args.knn_attention),  # LayerNorm + multi-hand attention
                nn.Sequential(               # mlp
                    nn.LayerNorm(inp_dim),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(inp_dim, num_classes))
            ]))

        # num_patches = args.sample_duration
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, inp_dim))

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.branch_merge = branch_merge
        warmup_temp, temp = map(float, args.temp)
        self.temp_schedule = np.concatenate((
            np.linspace(warmup_temp, temp, warmup_temp_epochs),
            np.ones(args.epochs - warmup_temp_epochs) * temp
        ))
        # self.show_res = Rearrange('b t (c p1 p2) -> b t c p1 p2', p1=int(small_dim ** 0.5), p2=int(small_dim ** 0.5))

        if init:
            self.init_weights()

    def TC_forward(self):
        return self.tc_feat

    # @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    def forward(self, x):  # x size: [2, 64, 512]
        B, N, C = x.shape

        # Add position embedding
        # x += self.pos_embedding

        # Local-Global features capturing
        outputs, tem_feat = [], []
        temp = self.temp_schedule[
            self._args.epoch]  # 记录了训练过程中每个轮次对应的温度值。在训练初期进行线性升温（warm-up），从warmup_temp逐步升至temp，之后在剩余训练轮次中保持温度temp不变。
        # 这种温度调度策略有助于模型在训练初期更加平滑地收敛，同时在后期保持稳定的决策边界或输出分布。

        for cls_token, (TC3D, MaxPool, TransBlock, mlp) in zip(self.cls_tokens, self.multi_scale_transformers):  # cls_token -> embedding
            # cls_token = self.__dict__['cls_token_{}'.format(i)]

            sl = uniform_sampling(x.size(1), cls_token.frame_rate, random=self.training)
            sub_x = x[:, sl, :]
            sub_x = sub_x.permute(0, 2, 1).view(B, C, -1, 1, 1)
            sub_x = MaxPool(TC3D(sub_x))
            sub_x = sub_x.permute(0, 2, 1, 3, 4).view(B, -1, C)

            sub_x = cls_token(sub_x)
            sub_x = TransBlock(sub_x)
            sub_x = sub_x[:, 0, :]

            out = mlp(sub_x)
            outputs.append(out / temp)  # temp变量在这个代码片段中的作用是对模型的输出进行温度调整，通过改变概率分布的陡峭程度来影响模型的决策行为和输出多样性。
            # 这种调整有助于控制模型在训练过程中的探索与 exploitation 平衡，以及在推理阶段生成结果的确定性与创新性。

        # Multi-branch fusion
        if self.branch_merge == 'sum':
            x = torch.zeros_like(out)
            for out in outputs:
                x += out
        elif self.branch_merge == 'pool':
            x = torch.cat([out.unsqueeze(-1) for out in outputs], dim=-1)
            x = self.max_pool(x).squeeze()
        return x
