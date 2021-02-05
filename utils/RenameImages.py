"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/5 16:37:21
# @File    : RenameImages.py
# @Software: PyCharm
-------------------------------------
"""

import os
import shutil

images_file_list = [i for i in os.listdir('../data/raw_media_files') if
                    ('.jpg' in i) or ('.jfif' in i) or ('.png' in i)]

print(len(images_file_list))

for image_name in images_file_list:
    old_name = image_name
    new_name = image_name.split('_')[0] + '.png'
    shutil.copyfile('../data/raw_media_files/{}'.format(old_name),
                    '../data/images_data/{}'.format(new_name))
    print(old_name, '======>', new_name)
