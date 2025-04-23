#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import dependency library
import numpy as np
import math
import pyshtools as pysh


# import user defined library

from .general_func import sph2descartes


def flatten_clim(sh_coefficient_array):
    """
    # -----------------------------------
    # cilm coefficient:
    # [0,:,:]---->m >= 0
    # [1,:,:]---->m <0
    # -----------------------------------
    :param sh_coefficient_array:
    :return:
    """
    flatten_array = []
    coefficient_degree = sh_coefficient_array.shape[1]
    # print(coefficient_degree)
    for l_degree in range(coefficient_degree):
        # m<0
        for m_order in np.arange(l_degree, 0, step=-1):
            flatten_array.append(sh_coefficient_array[1, l_degree, m_order])
            # print(1, l_degree, m_order)
        # m>=0
        for m_order in np.arange(0, l_degree + 1):
            flatten_array.append(sh_coefficient_array[0, l_degree, m_order])
            # print(0, l_degree, m_order)

    return np.array(flatten_array)


def collapse_flatten_clim(flatten_clim):
    l_degree = int(math.sqrt(len(flatten_clim)))
    # print(l_degree)
    clim_array = np.zeros((2, l_degree, l_degree))
    for l_i in range(l_degree):
        # enumerate_times = 2 * l_i + 1
        for m_j in np.arange(-l_i, l_i + 1, step=1):
            if m_j < 0:
                # print(l_i, m_j, 2 * l_i + m_j)
                clim_array[1, l_i, np.abs(m_j)] = flatten_clim[l_i ** 2 + m_j + l_i]
            else:
                # print(l_i, m_j, 2 * l_i + m_j)
                clim_array[0, l_i, m_j] = flatten_clim[l_i ** 2 + m_j + l_i]
    return clim_array


def get_flatten_ldegree_morder(degree):
    """

    :param degree: see the l_degree explanation, that's why I need to plus one in func:get_flatten_ldegree_morder
    :return:  index slice : in dataframe it's called columns
    """
    index_slice = []
    # print(coefficient_degree)
    for l_degree in range(degree + 1):
        # m<0
        for m_order in np.arange(l_degree, 0, step=-1):
            index_slice.append('l' + str(l_degree) + '-m' + str(-m_order))
            # print(1, l_degree, m_order)
        # m>=0
        for m_order in np.arange(0, l_degree + 1):
            index_slice.append('l' + str(l_degree) + '-m' + str(m_order))
            # print(0, l_degree, m_order)
    # in dataframe it's called columns
    return index_slice



def do_reconstruction_from_SH(sample_N: int, sh_coefficient_instance: pysh.SHCoeffs):
    """
    latitude!!
    :param sample_N: sample N, total samples will be 2*sample_N**2
    :param sh_coefficient_instance: the SH transform result
    :param average_sampling: np.mean(array(shape=average_sampling))
    :return:  SH transform xyz reconstruction
    """
    plane_representation_lat = np.arange(-90, 90, 180 / sample_N)
    plane_representation_lon = np.arange(0, 360, 360 / (2 * sample_N))
    plane_LAT, plane_LON = np.meshgrid(plane_representation_lat, plane_representation_lon)

    plane_LAT_FLATTEN = plane_LAT.flatten(order='F')
    plane_LON_FLATTEN = plane_LON.flatten(order='F')
    grid = sh_coefficient_instance.expand(lat=plane_LAT_FLATTEN, lon=plane_LON_FLATTEN)

    plane_LAT_FLATTEN = plane_LAT_FLATTEN / 180 * math.pi
    plane_LON_FLATTEN = plane_LON_FLATTEN / 180 * math.pi

    reconstruction_matrix = []
    for i in range(grid.data.shape[0]):
        reconstruction_matrix.append([grid.data[i], plane_LAT_FLATTEN[i], plane_LON_FLATTEN[i]])

    reconstruction_xyz = sph2descartes(np.array(reconstruction_matrix))
    reconstruction_xyz[:, 2] = -reconstruction_xyz[:, 2]
    return reconstruction_xyz
