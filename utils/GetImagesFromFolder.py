"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/5 18:18:29
# @File    : GetImagesFromFolder.py
# @Software: PyCharm
-------------------------------------
"""

import os
import shutil

image_folders_list = [i for i in os.listdir('../data/images_data') if ('.png' not in i)]

print(image_folders_list)

for folder in image_folders_list:
    old_name = os.listdir('../data/images_data/{}'.format(folder))[0]
    new_name = folder + '.png'
    shutil.move('../data/images_data/{}/{}'.format(folder, old_name),
                '../data/images_data/{}'.format(new_name))
    os.rmdir('../data/images_data/{}'.format(folder))
    print(old_name, '======>', new_name)
