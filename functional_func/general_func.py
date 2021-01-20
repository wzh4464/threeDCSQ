import math

import nibabel as nib
import os
from nibabel.viewers import OrthoSlicer3D

import numpy as np


def load_nitf2_img(path):
    return nib.load(path)


def show_nitf2_img(path):
    img = load_nitf2_img(path)
    OrthoSlicer3D(img.dataobj).show()
    return img


def deal_all(this_dir):
    for (root, dirs, files) in os.walk(this_dir):
        # print(files)
        for file in files:
            print(root, file)
            this_file_path = os.path.join(root, file)
            this_img = nib.load(this_file_path)
            print(this_img.shape)


# the code reference: https://www.thinbug.com/q/4116658
def descartes2spherical2(points_xyz):
    """

    :param points_xyz:
    :return: [r,co-latitude(from z),longitude]
    """
    pts_sph = np.zeros(points_xyz.shape)

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # be careful, this is my transform base formula. There are many different transformation methods I thinks.
    xy = points_xyz[:, 0] ** 2 + points_xyz[:, 1] ** 2
    pts_sph[:, 0] = np.sqrt(xy + points_xyz[:, 2] ** 2)
    pts_sph[:, 1] = np.arctan2(np.sqrt(xy), points_xyz[:, 2])  # lat phi
    # pts_sph[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    pts_sph[:, 2] = np.arctan2(points_xyz[:, 1], points_xyz[:, 0]) % (2 * math.pi)  # lon theta
    return pts_sph


def descartes2spherical(points_xyz):
    """

    :param points_xyz:
    :return: [r,latitude(inclination angle,from y),longitude]
    """
    pts_sph = np.zeros(points_xyz.shape)

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # be careful, this is my transform base formula. There are many different transformation methods I thinks.
    xy = points_xyz[:, 0] ** 2 + points_xyz[:, 1] ** 2
    pts_sph[:, 0] = np.sqrt(xy + points_xyz[:, 2] ** 2)
    pts_sph[:, 1] = np.arctan2(np.sqrt(xy), points_xyz[:, 2]) - math.pi / 2  # lat phi
    # pts_sph[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    pts_sph[:, 2] = np.arctan2(points_xyz[:, 1], points_xyz[:, 0]) % (2 * math.pi)  # lon theta
    return pts_sph


def sph2descartes2(points_sph):
    """

    :param points_sph: latitude is used in co-latitude 0-180
    :return:
    """
    points_xyz = np.zeros(points_sph.shape)

    radius = points_sph[:, 0]
    lat = points_sph[:, 1]
    lon = points_sph[:, 2]

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # be careful, this is my transform base formula. There are many different transformation methods I thinks.
    points_xyz[:, 0] = np.cos(lon) * np.sin(lat) * radius
    points_xyz[:, 1] = np.sin(lon) * np.sin(lat) * radius
    points_xyz[:, 2] = np.cos(lat) * radius

    return points_xyz


def sph2descartes(points_sph):
    """

    :param points_sph: latitude is used in latitude -90-90 degrees
    :return:
    """
    points_xyz = np.zeros(points_sph.shape)

    radius = points_sph[:, 0]
    lat = points_sph[:, 1] - math.pi / 2
    lon = points_sph[:, 2]

    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # be careful, this is my transform base formula. There are many different transformation methods I thinks.
    points_xyz[:, 0] = np.cos(lon) * np.sin(lat) * radius
    points_xyz[:, 1] = np.sin(lon) * np.sin(lat) * radius
    points_xyz[:, 2] = np.cos(lat) * radius

    return -points_xyz


def to_the_power_expand(modified_array):
    """

    :param modified_array:
    :return:
    """
    rows = modified_array.shape[0]
    columns = modified_array.shape[1]
    return_array = np.zeros(modified_array.shape)
    for i in range(rows):
        for j in range(columns):
            if modified_array[i][j] > 0:
                return_array[i][j] = (modified_array[i][j] / 10) ** 2
            elif modified_array[i][j] < 0:
                return_array[i][j] = -(-modified_array[i][j] / 10) ** 2
            else:
                return_array[i][j] = 0

    return return_array


def sqrt_expand(original_array):
    """

    :param original_array:
    :return:
    """
    rows = original_array.shape[0]
    columns = original_array.shape[1]
    return_array = np.zeros(original_array.shape)
    for i in range(rows):
        for j in range(columns):
            if original_array[i][j] > 0:
                return_array[i][j] = np.sqrt(original_array[i][j]) * 10
            elif original_array[i][j] < 0:
                return_array[i][j] = -np.sqrt(-original_array[i][j]) * 10
            else:
                return_array[i][j] = 0

    return return_array


def log_expand_offset(original_array, offset):
    """
    many problem!!!!!!!!!!!!!!!!!!!!!!!!!!!
    offset plus and minus is very bad for log
    in math , it is not good.
    :param original_array:
    :param offset:
    :return:
    """
    rows = original_array.shape[0]
    columns = original_array.shape[1]
    return_array = np.zeros(original_array.shape)
    for i in range(rows):
        for j in range(columns):
            if original_array[i][j] > 0:
                return_array[i][j] = math.log(original_array[i][j]) + offset
            elif original_array[i][j] < 0:
                return_array[i][j] = -math.log(-original_array[i][j]) - offset
            else:
                return_array[i][j] = 0

    return return_array


def exp_expand_offset(modified_array, offset):
    rows = modified_array.shape[0]
    columns = modified_array.shape[1]
    return_array = np.zeros(modified_array.shape)
    for i in range(rows):
        for j in range(columns):
            if modified_array[i][j] > 0:
                return_array[i][j] = math.exp(modified_array[i][j] - offset)
            elif modified_array[i][j] < 0:
                return_array[i][j] = -math.exp(-(modified_array[i][j] + offset))
            else:
                return_array[i][j] = 0

    return return_array
