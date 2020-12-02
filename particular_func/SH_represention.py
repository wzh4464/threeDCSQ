import math
import os
import config
import time

import numpy as np
import pyshtools as pysh
from matplotlib import pyplot as plt
from multiprocessing import Process

import functional_func.general_func as general_f
import functional_func.cell_func as cell_f
import functional_func.spherical_func as sph_f

import particular_func.SH_analyses as sh_a_f


def do_sampling_with_interval(N, points_surface, average_num):
    radian_interval = math.pi / N
    griddata = np.zeros((N, 2 * N))

    points_surface = general_f.descartes2spherical2(points_surface)
    # points_at_spherical_lat_phi, points_at_spherical_lon_theta = sph_f.sort_by_phi_theta(points_surface)
    # interval_tmp = int(len(points_surface) / (N * N * 2))
    # interval_of_sample_and_all_points = interval_tmp if interval_tmp > 0 else 1
    spherical_matrix = []
    for i in range(N):
        for j in range(2 * N):
            griddata[i][j] = sph_f.calculate_R_with_sortation(points_surface,
                                                              radian_interval * i, radian_interval * j,
                                                              average_num)
            print("\r Loading  ", end='row   ' + str(i) + " and column  " + str(j) + " of all  " + str(N ** 2 * 2))
            # griddata[i][j]=1
            spherical_matrix.append([griddata[i][j], radian_interval * i, radian_interval * j])
    # print(griddata)
    # draw_f.draw_3D_points(general_f.sph2descartes(sperical_matrix))

    return griddata, np.array(spherical_matrix)


def get_SH_coeffient_from_surface_points(file_name, sample_N, lmax, embryo_path,
                                         surface_average_num=10):
    """

    :param sample_N: how many samples we need
    :param embryo_path: the shape img path
    :param file_name: file name
    :param surface_average_num: how many points to get average num to calculate R=f(phi,theta)
    :return:
    """
    if os.path.exists(os.path.join(config.dir_my_data, 'membrane' + file_name)):
        img = general_f.load_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + file_name))
    else:
        img = cell_f.nii_get_cell_surface(general_f.load_nitf2_img(os.path.join(embryo_path, file_name)),
                                          file_name)  # calculate membrane and save automatically

    dict_img_membrane_calculate = {}
    img_membrane_data = img.get_fdata().astype(np.int16)
    x_num, y_num, z_num = img_membrane_data.shape
    # -------------get each cell membrane----------------
    for x in range(x_num):
        for y in range(y_num):
            for z in range(z_num):
                dict_key = img_membrane_data[x][y][z]
                if dict_key != 0:
                    if dict_key in dict_img_membrane_calculate:
                        dict_img_membrane_calculate[dict_key].append([x, y, z])
                    else:
                        dict_img_membrane_calculate[dict_key] = [[x, y, z]]
    # ----------------------

    # -------------get each full cell----------------
    if os.path.exists(os.path.join(embryo_path, file_name)):
        img = general_f.load_nitf2_img(os.path.join(embryo_path, file_name))
    else:
        return EOFError  # calculate cell and save automatically
    dict_img_cell_calculate = {}
    img_cell_data = img.get_fdata().astype(np.int16)

    for x in range(x_num):
        for y in range(y_num):
            for z in range(z_num):
                dict_key = img_cell_data[x][y][z]
                if dict_key != 0:
                    if dict_key in dict_img_cell_calculate:
                        dict_img_cell_calculate[dict_key].append([x, y, z])
                    else:
                        dict_img_cell_calculate[dict_key] = [[x, y, z]]
    # ---------------------------------------------------

    cell_name_affine_table = cell_f.get_cell_name_affine_table()
    # THE DEGREE OF SH. we can specify it or less than N/2-1, or 10 -- 2*(10/2-1)+1=9 coefficients
    # lmax = int(sample_N / 2 - 1)

    this_embryo_dir = os.path.join(embryo_path, 'SH_C_folder_OF' + file_name)
    print('placing SH file in====>',this_embryo_dir)
    if not os.path.exists(this_embryo_dir):
        os.makedirs(this_embryo_dir)

    cilm = {}

    # --------------------------calculate all and save as npy
    for dict_key in dict_img_membrane_calculate.keys():
        # get center point
        dict_img_membrane_calculate[dict_key] = np.array(dict_img_membrane_calculate[dict_key])
        center_points = np.sum(dict_img_cell_calculate[dict_key], axis=0) / len(dict_img_cell_calculate[dict_key])
        if center_points is None:
            center_points = [0, 0, 0]
        points_membrane_local = dict_img_membrane_calculate[dict_key] - center_points

        save_file_name = os.path.join(this_embryo_dir, cell_name_affine_table[dict_key - 1])

        if not os.path.exists(save_file_name):
            # do sampling to fit [ Driscoll and Healy's (1994) sampling theorem. ]
            # ---https://www.sciencedirect.com/science/article/pii/S0196885884710086?via%3Dihub
            griddata, samples_result = do_sampling_with_interval(sample_N, points_membrane_local, surface_average_num)

            # do fourier transform and convolution on SPHERE
            print('---------dealing with cell ' + str(dict_key) + '-----' + cell_name_affine_table[
                dict_key - 1] + '   ---coefficient --------------------')
            # calculate coefficients from points
            cilm[dict_key] = pysh.shtools.SHExpandDH(griddata, sampling=2, lmax_calc=lmax)
            # build sh tools coefficient class instance
            sh_coefficient_instance = pysh.SHCoeffs.from_array(cilm[dict_key])

            sh_coefficient_instance.to_file(filename=save_file_name)
        else:
            sh_coefficient_instance = pysh.SHCoeffs.from_file(save_file_name)


