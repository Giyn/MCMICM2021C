"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/8 18:48:43
# @File    : predict.py
# @Software: PyCharm
-------------------------------------
"""

from __future__ import print_function

import os
import pickle

import cv2
import numpy as np
import torch

image_size = (224, 224, 3)
label_dict = {'0': 'bee',
              '1': 'cicada',
              '2': 'hornet',
              '3': 'horntail',
              '4': 'moth',
              '5': 'negative',
              '6': 'poliste',
              '7': 'positive',
              '8': 'sawfly',
              '9': 'wasp',
              '10': 'yellowjacket'
              }


def predict(model_, img_arr, device_):
    in_image = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    in_image = cv2.resize(in_image, (image_size[0], image_size[1]))
    in_image = np.transpose(in_image, (2, 0, 1)) / 255
    in_image = np.expand_dims(in_image, axis=0)
    in_image = torch.tensor(in_image.astype('float32'))
    in_image = in_image.to(device_)
    net_out = model_(torch.tensor(in_image))
    label = str(net_out.argmax().item())

    return label_dict[label], net_out[0][7].item()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running device : {}'.format(device))
model = torch.load('model.pb')
model.to(device)
model.eval()

if __name__ == '__main__':
    test_dir = '../images_data/unverified_images'
    positive_degree_dict = {}
    for filename in os.listdir(test_dir):
        filepath = os.path.join(test_dir, filename)
        img = cv2.imread(filepath)
        # rec_c = cv2.cvtColor(rec_c, cv2.COLOR)
        # print(img.shape)
        sol, positive_degree = predict(model, img, device)
        print(filename + ' prediction: {}'.format(sol))
        positive_degree_dict[filename] = positive_degree

    with open('positive_degree', 'wb') as file:
        pickle.dump(positive_degree_dict, file)
