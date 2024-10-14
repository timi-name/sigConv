'''
This file is modified from:
https://github.com/zhoubenjia/RAAR3DNet/blob/master/Network_Train/lib/datasets/base.py
'''
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, set_image_backend
import torch.nn.functional as F

from PIL import Image
from PIL import ImageFilter, ImageOps
import os, glob
import math, random
import numpy as np
import logging
from tqdm import tqdm as tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import cv2
import json
from scipy.ndimage.filters import gaussian_filter

from timm.data.random_erasing import RandomErasing
# from vidaug import augmentors as va
from .augmentation import *

# import functools
import matplotlib.pyplot as plt  # For graphics
from torchvision.utils import save_image, make_grid

np.random.seed(123)


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x


class Datasets(Dataset):
    global kpt_dict

    def __init__(self, args, ground_truth, modality, phase='train'):

        self.dataset_root = args.data
        self.sample_duration = args.sample_duration
        self.sample_size = args.sample_size
        self.phase = phase
        self.typ = modality
        self.args = args
        self._w = args.w

        if phase == 'train':
            self.transform = transforms.Compose([
                Normaliztion(),
                transforms.ToTensor(),
                RandomErasing(args.reprob, mode=args.remode, max_count=args.recount, num_splits=0, device='cuda')
            ])
        else:
            self.transform = transforms.Compose([Normaliztion(), transforms.ToTensor()])

        self.inputs, self.video_apth = self.prepropose(ground_truth)

    def prepropose(self, ground_truth, min_frames=16):
        def get_data_list_and_label(data_df):
            return [(lambda arr: (arr[0], int(arr[1]), int(arr[2])))(i.strip().split(' '))
                    for i in open(data_df).readlines()]

        self.inputs = list(filter(lambda x: x[1] > min_frames, get_data_list_and_label(ground_truth)))
        self.inputs = list(self.inputs)
        self.batch_check()
        random.shuffle(self.inputs)
        self.video_apth = dict([(self.inputs[i][0], i) for i in range(len(self.inputs))])
        return self.inputs, self.video_apth

    def batch_check(self):
        if self.phase == 'train':
            while len(self.inputs) % (self.args.batch_size * self.args.nprocs) != 0:
                sample = random.choice(self.inputs)
                self.inputs.append(sample)
        else:
            while len(self.inputs) % (self.args.test_batch_size * self.args.nprocs) != 0:
                sample = random.choice(self.inputs)
                self.inputs.append(sample)

    def __str__(self):
        if self.phase == 'train':
            frames = [n[1] for n in self.inputs]
            return 'Training Data Size is: {} \n'.format(
                len(self.inputs)) + 'Average Train Data frames are: {}, max frames: {}, min frames: {}\n'.format(
                sum(frames) // len(self.inputs), max(frames), min(frames))
        else:
            frames = [n[1] for n in self.inputs]
            return 'Validation Data Size is: {} \n'.format(
                len(self.inputs)) + 'Average validation Data frames are: {}, max frames: {}, min frames: {}\n'.format(
                sum(frames) // len(self.inputs), max(frames), min(frames))

    def transform_params(self, resize=(320, 240), crop_size=224, flip=0.5):
        if self.phase == 'train':
            left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
            is_flip = True if np.random.uniform(0, 1) < flip else False
        else:
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2
            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def get_path(self, imgs_path, a):
        return os.path.join(imgs_path, "%06d.jpg" % a)

    def depthProposess(self, img):
        h2, w2 = img.shape

        mask = img.copy()
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((10, 10), np.uint8))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find Max Maxtri
        Idx = []
        for i in range(len(contours)):
            Area = cv2.contourArea(contours[i])
            if Area > 500:
                Idx.append(i)
        centers = []

        for i in Idx:
            rect = cv2.minAreaRect(contours[i])
            center, (h, w), degree = rect
            centers.append(center)

        finall_center = np.int0(np.array(centers))
        c_x = min(finall_center[:, 0])
        c_y = min(finall_center[:, 1])
        center = (c_x, c_y)

        crop_x, crop_y = 320, 240
        left = center[0] - crop_x // 2 if center[0] - crop_x // 2 > 0 else 0
        top = center[1] - crop_y // 2 if center[1] - crop_y // 2 > 0 else 0
        crop_w = left + crop_x if left + crop_x < w2 else w2
        crop_h = top + crop_y if top + crop_y < h2 else h2
        rect = (left, top, crop_w, crop_h)
        image = Image.fromarray(img)
        image = image.crop(rect)
        return image

    def image_propose(self, data_path, sl):
        sample_size = self.sample_size
        resize = eval(self.args.resize)
        crop_rect, is_flip = self.transform_params(resize=resize, crop_size=self.args.crop_size, flip=self.args.flip)
        if np.random.uniform(0, 1) < self.args.rotated and self.phase == 'train':
            r, l = eval(self.args.angle)
            rotated = np.random.randint(r, l)
        else:
            rotated = 0

        sometimes = lambda aug: Sometimes(0.5, aug)  # Used to apply augmentor with 50% probability
        self.seq_aug = Sequential([
            RandomResize(self.args.resize_rate),
            RandomCrop(resize),
            # RandomTranslate(self.args.translate, self.args.translate),
            # sometimes(Salt()),
            # sometimes(GaussianBlur()),
        ])

        def transform(img):
            img = np.asarray(img)
            if img.shape[-1] != 3:
                img = np.uint8(255 * img)
                img = self.depthProposess(img)
                img = cv2.applyColorMap(np.asarray(img), cv2.COLORMAP_JET)
            img = self.rotate(np.asarray(img), rotated)
            img = Image.fromarray(img)
            if self.phase == 'train' and self.args.strong_aug:
                img = self.seq_aug(img)

            img = img.resize(resize)
            img = img.crop(crop_rect)

            if is_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img.resize((sample_size, sample_size)))

        def Sample_Image(imgs_path, sl):
            frams = []
            for a in sl:
                ori_image = Image.open(self.get_path(imgs_path, a))
                img = transform(ori_image)
                frams.append(self.transform(img).view(3, sample_size, sample_size, 1))
            if self.args.frp:
                skgmaparr = DynamicImage(frams, dynamic_only=False)  # [t, c, h, w]
            else:
                skgmaparr = torch.ones(*img.shape, 1)
            return torch.cat(frams, dim=3).type(torch.FloatTensor), skgmaparr

        def DynamicImage(frames, dynamic_only):  # frames: [[3, 224, 224, 1], ]
            def tensor_arr_rp(arr):
                l = len(arr)
                statics = []

                def tensor_rankpooling(video_arr, lamb=1.):
                    def get_w(N):
                        return [float(i) * 2 - N - 1 for i in range(1, N + 1)]

                    re = torch.zeros(*video_arr[0].size()[:-1])
                    for a, b in zip(video_arr, get_w(len(video_arr))):
                        re += a.squeeze() * b
                    re = (re - re.min()) / (re.max() - re.min())
                    re = np.uint8(255 * np.float32(re.numpy())).transpose(1, 2, 0)
                    warnings.filterwarnings("ignore", category=RuntimeWarning,
                                            message="invalid value encountered in cast")

                    re = self.transform(np.array(re))
                    return re.unsqueeze(-1)

                return [tensor_rankpooling(arr[i:i + self._w]) for i in range(l)]

            arrrp = tensor_arr_rp(frames)
            arrrp = torch.cat(arrrp[:], dim=-1).type(torch.FloatTensor)
            return arrrp

        return Sample_Image(data_path, sl)

    def equal_part_sample(self, frames, sample_duration):
        lst = list(range(frames))

        clip = len(lst)
        list_temp = list(lst)
        temp_point = []
        temp_remainder = []
        sample_point_index_list = []
        # 计算等分点
        n_equal_point = sample_duration
        equal_parts = n_equal_point + 1  # 除端点外, 利用 k 个点把序列分成k + 1段
        part_length = clip // equal_parts

        def mult_Sampling(sample_duration):
            n_equal_point = int(np.ceil(sample_duration * 0.3))
            part_length = clip // (n_equal_point + 1)

            sample_point = (sample_duration - (n_equal_point + 2)) // ((n_equal_point + 1) * 2)
            remainder = (sample_duration - (n_equal_point + 2)) % (2 * (n_equal_point + 1))
            temp_point.append(lst[0])
            temp_point.append(lst[-1])
            for point_index in range(1, n_equal_point + 1):  # 添加等分点 对应值
                sample_point_index = part_length * point_index
                sample_point_index_list.append(sample_point_index)  # 储存等分点索引
                temp_point.append(lst[sample_point_index])

            center_equal_point = (len(sample_point_index_list) - 1) // 2

            for sample_point_cnts in range(1, sample_point + 1):  # 添加 等分点两侧 对应值
                if len(temp_point) == (n_equal_point + 2) + sample_point * (2 * (sample_point + 1)):
                    break
                """
                为了优先从中间位置的等分点处（关键区域）, 向两侧的等分点偏移取点
                """
                for step in range(center_equal_point + 1):

                    center_equal_point_val = sample_point_index_list[center_equal_point]

                    center_equal_point_rightbias = sample_point_index_list[center_equal_point - (step + 1)]
                    if step == center_equal_point:
                        center_equal_point_leftbias = sample_point_index_list[center_equal_point + (step + 1) - 1]
                    else:
                        center_equal_point_leftbias = sample_point_index_list[center_equal_point + (step + 1)]

                    temp_point.append(lst[center_equal_point_rightbias + 1])
                    temp_point.append(lst[center_equal_point_leftbias - 1])
                    if step < 2:
                        continue
                    temp_point.append(lst[0 + step * 2 + 1])
                    temp_point.append(lst[-1 - step * 2 - 1])

            temp_point.sort()
            # return temp_point
            temp_remainder = equal_Sampling(sample_duration - len(temp_point))
            result = temp_point + temp_remainder
            result.sort()
            return result

        def equal_Sampling(points):
            # 对于小于的点 采用等分点采样法
            part_length_temp = clip // (points + 1)
            temp = []
            while len(temp) != points:  # 循环条件：是否装满目标数量的点
                for step in range(1, points + 1):  # 循环取points个点 range（0,1）不包括1，所以 +1
                    if len(temp) == points:  # 判断是否装满
                        break
                    else:
                        index = part_length_temp * step - 1  # 找到等分点位置
                        temp.append(lst[index])  # 索引对应的值加入列表
                for step in (0, -1):  # 遍历两个 端点
                    if len(temp) == points:  # 判断是否需要添加
                        break
                    temp.append(lst[step])
            temp.sort()
            return temp

        def remove_duplicate_points(existing_list):
            """
            给定一个整数列表，此函数返回一个新的无重复点的列表。
            遇到重复点时，将其替换为原列表范围内最近的唯一邻居整数。
            返回的列表确保最大值不超过原列表的最大值，最小值也不超过原列表的最小值。

            参数:
                existing_list (list[int]): 可能包含重复整数的原始列表。

            返回:
                list[int]: 新列表，无重复点且新加入的点均在原列表的最大值和最小值范围内。
            """
            existing_set = set(existing_list)  # 创建一个不重复元素的集合
            new_list = []
            center_index = len(existing_list) // 2
            center_point = existing_list[center_index]

            # 对列表进行排序，以便从中心点开始向两侧遍历
            sorted_list = sorted(existing_list, key=lambda x: abs(x - center_point))
            existing_set = set()  # 用于记录已经出现过的元素
            new_list = []

            for point in sorted_list:
                if point not in existing_set:
                    # 如果点是独一无二的，直接添加到新列表中
                    new_list.append(point)
                    existing_set.add(point)  # 将新元素添加到集合中
                else:
                    # 如果点是重复的，则寻找范围内的最近的唯一邻居整数
                    neighbor = point
                    while neighbor in existing_set:
                        neighbor += 1
                        # 如果超出了原列表的范围，重置为最小值
                        if neighbor > max(existing_list):
                            neighbor = min(existing_list)
                    new_list.append(neighbor)
                    existing_set.add(neighbor)  # 将新加入的点添加到集合中，以确保不会再次添加重复点

            new_list.sort()
            return new_list

        if sample_duration > clip:
            result = []
            integer_multiple = sample_duration // clip
            # sample_duration > 序列长度
            integer_multiple_ = integer_multiple
            while integer_multiple:
                result.extend(list_temp)
                integer_multiple -= 1
            residue = sample_duration - clip * integer_multiple_
            if residue:
                center_index = len(list_temp) // 2
                center_point = list_temp[center_index]

                # 对列表进行排序，以便从中心点开始向两侧遍历
                sorted_list = sorted(list_temp, key=lambda x: abs(x - center_point))
                for step, val in enumerate(sorted_list):
                    if step == residue:
                        break
                    else:
                        result.append(val)
            result.sort()
            return result
        else:
            result = []
            # 当采样数训练大于采样点加1时
            if sample_duration >= len(lst) // 2:  # 判断是否大于点数（等分点 + 两个端点）
                result = mult_Sampling(sample_duration)  # 该函数：需要分有没有余数  如果有  需要从等分点处向两侧取点
            else:
                result = equal_Sampling(sample_duration)

            return remove_duplicate_points(result)


    def get_sl(self, clip):
        sn = self.sample_duration if not self.args.frp else self.sample_duration
        if self.phase == 'train':
            f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn, range(int(n * i / sn),
                                                                max(int(n * i / sn) + 1, int(n * (i + 1) / sn)))) for i in range(sn)]
        else:
            f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                max(int(n * i / sn) + 1, int(n * (i + 1) / sn))))
                           for i in range(sn)]
        sample_clips = f(int(clip) - self.args.sample_window)
        start = random.sample(range(0, self.args.sample_window), 1)[0]
        if self.phase == 'train':
            return [l + start for l in sample_clips]
        else:
            return f(int(clip))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])

        if self.args.Network == 'FusionNet':
            assert self.typ != 'rgbd', "Please specify '--type rgbd'."
            self.data_path = os.path.join(self.dataset_root, 'rgb', self.inputs[index][0])
            self.clip, skgmaparr = self.image_propose(self.data_path, sl)

            self.data_path = os.path.join(self.dataset_root, 'depth', self.inputs[index][0])
            self.clip1, skgmaparr1 = self.image_propose(self.data_path, sl)
            return (self.clip.permute(0, 3, 1, 2), self.clip1.permute(0, 3, 1, 2)), (skgmaparr, skgmaparr1), \
                   self.inputs[index][2], self.data_path

        self.data_path = os.path.join(self.dataset_root, self.typ, self.inputs[index][0])
        self.clip, skgmaparr = self.image_propose(self.data_path, sl)

        return self.clip.permute(0, 3, 1, 2), skgmaparr.permute(0, 3, 1, 2), self.inputs[index][2], self.inputs[index][
            0]

    def __len__(self):
        return len(self.inputs)


if __name__ == '__main__':
    import argparse
    from config import Config
    from lib import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='Place config Congfile!')
    parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--nprocs', type=int, default=1)

    parser.add_argument('--save_grid_image', action='store_true', help='Save samples?')
    parser.add_argument('--save_output', action='store_true', help='Save logits?')
    parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')

    parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
    parser.add_argument('--save', type=str, default='Checkpoints/', help='experiment name')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    args = parser.parse_args()
    args = Config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.dist = False
    args.eval_only = True
    args.test_batch_size = 1

    valid_queue, valid_sampler = build_dataset(args, phase='val')
    for step, (inputs, heatmap, target, _) in enumerate(valid_queue):
        print(inputs.shape)
        input()
