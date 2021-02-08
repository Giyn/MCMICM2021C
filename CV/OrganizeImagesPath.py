"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/8 11:45:28
# @File    : OrganizeImagesPath.py
# @Software: PyCharm
-------------------------------------
"""

import os
import json
import shutil
import random

from operator import itemgetter

with open('../data/categorization/negative_samples_categorize.txt', 'r') as file:
    species_dict = json.loads(file.read().replace('\'', '\"'))

with open('../data/categorization/negative_species.txt', 'r') as file:
    species_list = list(map(lambda x: x.strip(), file.readlines()))

species_num_dict = {}

# 初始化
for name in species_list:
    species_num_dict[name] = 0

# 统计
for key, value in species_dict.items():
    if value in species_list:
        species_num_dict[value] += 1

# 去除数量为0的种类
num_zero_species = []
for key, value in species_num_dict.items():
    if value == 0:
        num_zero_species.append(key)

for i in num_zero_species:
    del species_num_dict[i]

species_images_dict = {}
for key, value in species_num_dict.items():
    species_images_dict[key] = []

species_images_dict['negative'] = []

for image, species in species_dict.items():
    species_images_dict[species].append(image)

# print(species_images_dict)

"""对图片进行文件夹分类以及划分训练集和测试集"""


# for species, img_list in species_images_dict.items():
#     os.makedirs('images_data/train_images/{}'.format(species))
#     os.makedirs('images_data/test_images/{}'.format(species))

def split_list(full_list, shuffle=True, ratio=0.8):
    """

    split list

    Args:
        full_list: raw list
        shuffle  : whether to shuffle(bool)
        ratio    : split ratio

    Returns:
        sublist_train: train list
        sublist_test : test list

    """
    length = len(full_list)
    offset = int(length * ratio)

    if length == 0 or offset < 1:
        return full_list, []

    if shuffle:
        random.shuffle(full_list)

    sublist_train = full_list[:offset]
    sublist_test = full_list[offset:]

    return sublist_train, sublist_test


# 按图片种类分配到文件夹
# for species, img_list in species_images_dict.items():
#     if species == 'poliste':
#         train_samples = img_list
#         test_samples = img_list
#         shutil.copyfile('images_data/{}'.format(img_list[0]),
#                         'images_data/train_images/{}/{}'.format(species, img_list[0]))
#         shutil.move('images_data/{}'.format(img_list[0]),
#                     'images_data/test_images/{}/{}'.format(species, img_list[0]))
#     else:
#         train_samples, test_samples = split_list(img_list)
#         for i in train_samples:
#             shutil.move('images_data/{}'.format(i),
#                         'images_data/train_images/{}/{}'.format(species, i))
#         for j in test_samples:
#             shutil.move('images_data/{}'.format(j),
#                         'images_data/test_images/{}/{}'.format(species, j))

# positive样本处理
with open('../data/categorization/positive_samples_categorize.txt', 'r') as file:
    positive_species_dict = json.loads(file.read().replace('\'', '\"'))

# print(positive_species_dict)

positive_species_images_dict = {}

for key, value in positive_species_dict.items():
    positive_species_images_dict[value] = []

for key, value in positive_species_dict.items():
    positive_species_images_dict[value].append(key)

# print(positive_species_images_dict)

# for species, img_list in positive_species_images_dict.items():
#     train_samples, test_samples = split_list(img_list)
#     for i in train_samples:
#         shutil.move('images_data/{}'.format(i),
#                     'images_data/train_images/{}/{}'.format(species, i))
#     for j in test_samples:
#         shutil.move('images_data/{}'.format(j),
#                     'images_data/test_images/{}/{}'.format(species, j))
