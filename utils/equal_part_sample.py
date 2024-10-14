import numpy as np


def equal_part_sample(frames, sample_duration):
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
        n_equal_point = int(np.ceil(sample_duration * 0.2))
        part_length = clip // (n_equal_point+1)

        sample_point = (sample_duration-(n_equal_point + 2))//((n_equal_point+1)*2)
        remainder = (sample_duration - (n_equal_point + 2)) % (2 * (n_equal_point + 1))
        temp_point.append(lst[0])
        temp_point.append(lst[-1])
        for point_index in range(1, n_equal_point + 1):             # 添加等分点 对应值
            sample_point_index = part_length * point_index
            sample_point_index_list.append(sample_point_index)      # 储存等分点索引
            temp_point.append(lst[sample_point_index])

        center_equal_point = (len(sample_point_index_list)-1)//2

        for sample_point_cnts in range(1, sample_point + 1):        # 添加 等分点两侧 对应值
            if len(temp_point) == (n_equal_point + 2) + sample_point * (2 * (sample_point + 1)):
                break
            """
            为了优先从中间位置的等分点处（关键区域）, 向两侧的等分点偏移取点
            """
            for step in range(center_equal_point+1):

                center_equal_point_val = sample_point_index_list[center_equal_point]

                center_equal_point_rightbias = sample_point_index_list[center_equal_point - (step + 1)]
                if step == center_equal_point:
                    center_equal_point_leftbias = sample_point_index_list[center_equal_point + (step + 1) - 1]
                else:
                    center_equal_point_leftbias = sample_point_index_list[center_equal_point + (step + 1)]

                temp_point.append(lst[center_equal_point_rightbias+1])
                temp_point.append(lst[center_equal_point_leftbias-1])
                if step < 2:
                    continue
                temp_point.append(lst[0 + step*2+1])
                temp_point.append(lst[-1 - step*2-1])

        temp_point.sort()
        # return temp_point
        temp_remainder = equal_Sampling(sample_duration - len(temp_point))
        result = temp_point + temp_remainder
        result.sort()
        return result

    def equal_Sampling(points):
        # 对于小于的点 采用等分点采样法
        part_length_temp = clip // (points+1)
        temp = []
        while len(temp) != points:                  # 循环条件：是否装满目标数量的点
            for step in range(1, points+1):         # 循环取points个点 range（0,1）不包括1，所以 +1
                if len(temp) == points:             # 判断是否装满
                    break
                else:
                    index = part_length_temp * step-1    # 找到等分点位置
                    temp.append(lst[index])         # 索引对应的值加入列表
            for step in (0, -1):                    # 遍历两个 端点
                if len(temp) == points:             # 判断是否需要添加
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
        integer_multiple = sample_duration//clip
        # sample_duration > 序列长度
        integer_multiple_ = integer_multiple
        while integer_multiple:
            result.extend(list_temp)
            integer_multiple -= 1
        residue = sample_duration-clip*integer_multiple_
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
        if sample_duration >= len(lst)//2:     # 判断是否大于点数（等分点 + 两个端点）
            result = mult_Sampling(sample_duration)                                               # 该函数：需要分有没有余数  如果有  需要从等分点处向两侧取点
        else:
            result = equal_Sampling(sample_duration)

        return remove_duplicate_points(result)


# 使用示例
seq = 55
for i in range(1, 301):
    result = equal_part_sample(i, 16)
    print(i)
    print(len(result))
    print(result)
