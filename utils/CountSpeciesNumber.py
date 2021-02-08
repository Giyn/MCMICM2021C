"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/8 9:22:55
# @File    : CountSpeciesNumber.py
# @Software: PyCharm
-------------------------------------
"""

import json

with open('../data/categorization/negative_samples_categorize.txt', 'r') as file:
    species_dict = json.loads(file.read().replace('\'', '\"'))

with open('../data/categorization/negative_species.txt', 'r') as file:
    species_list = list(map(lambda x: x.strip(), file.readlines()))


print(species_dict)
print(species_list)

species_num_dict = {}

# 初始化
for name in species_list:
    species_num_dict[name] = 0

print(species_num_dict)

# 统计
for key, value in species_dict.items():
    if value in species_list:
        species_num_dict[value] += 1

print(species_num_dict)
