"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/8 19:46:18
# @File    : DivideValidationSet.py
# @Software: PyCharm
-------------------------------------
"""

import os
import shutil
import random


def split_list(full_list, shuffle=True, ratio=0.75):
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


species_list = os.listdir('images_data/train_images')

for species in species_list:
    images_list = os.listdir('images_data/train_images/{}'.format(species))

    if len(images_list) == 1:
        train_data = images_list
        validation_data = images_list
        # for i in validation_data:
        #     shutil.copy('images_data/train_images/{}/{}'.format(species, i),
        #                 'images_data/validation_images/{}/{}'.format(species, i))
    else:
        train_data, validation_data = split_list(images_list)
        # for j in validation_data:
        #     shutil.move('images_data/train_images/{}/{}'.format(species, j),
        #                 'images_data/validation_images/{}/{}'.format(species, j))
