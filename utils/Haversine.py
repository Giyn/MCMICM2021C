"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/5 10:00:54
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : Haversine.py
# @Software: PyCharm
-------------------------------------
"""

from math import radians
from sklearn.metrics.pairwise import haversine_distances


def get_distance(point_1: list, point_2: list) -> float:
    """

    Get the distance between two latitude and longitude points

    Args:
        point_1: [lat, lng]
        point_2: [lat, lng]

    Returns:
        distance between two points(unit: kilometers)

    """
    point_1_radians = [radians(_) for _ in point_1]
    point_2_radians = [radians(_) for _ in point_2]
    distance = (haversine_distances([point_1_radians, point_2_radians]) * 6371000 / 1000)[0][1]

    return distance


if __name__ == '__main__':
    lat1 = 48.980994
    lng1 = -122.688503
    lat2 = 48.93401
    lng2 = -122.48545

    a = [lat1, lng1]
    b = [lat2, lng2]

    print(get_distance(a, b), 'km')
