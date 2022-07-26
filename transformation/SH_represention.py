import math
import os
from static import config

import numpy as np
import pandas as pd
import pyshtools as pysh

import utils.general_func as general_f
import utils.cell_func as cell_f
import utils.spherical_func as sph_f
from utils.sh_cooperation import get_flatten_ldegree_morder, flatten_clim


def do_sampling_with_lat_lon(points_surface, lat_lon, average_num=5, is_return_xyz=False):
    """

    :param points_surface:
    :param lat_lon: co-lat points list, should be same as lon
    :param average_num:
    :param is_return_xyz:
    :return: R and coordinate array
    """

    points_surface = general_f.descartes2spherical2(points_surface)

    list_return = []
    for item in lat_lon:
        R = sph_f.spherical_R_with_lat_lon(points_surface, item[0], item[1], average_num)
        list_return.append([R, item[0], item[1]])
    list_return = np.array(list_return)
    if is_return_xyz:
        return list_return[:, 0], np.array(general_f.sph2descartes2(list_return))
    return list_return[:, 0], np.array(list_return)


def do_sampling_with_interval(N, points_surface, average_num, is_return_xyz=False):
    """
    co-latitude
    :param N: lat N, lon 2N
    :param points_surface: surface points in xyz
    :param average_num: how many average closest number to do calculate R
    :return: R distance 2D grid matrix, spherical coordinate
    """
    radian_interval = math.pi / N
    griddata = np.zeros((N, 2 * N))

    points_surface = general_f.descartes2spherical2(points_surface)

    # points_at_spherical_lat_phi, points_at_spherical_lon_theta = sph_f.sort_by_phi_theta(points_surface)
    # interval_tmp = int(len(points_surface) / (N * N * 2))
    # interval_of_sample_and_all_points = interval_tmp if interval_tmp > 0 else 1
    spherical_matrix = []
    for i in range(N):
        for j in range(2 * N):
            griddata[i][j] = sph_f.spherical_R_with_lat_lon(points_surface,
                                                            radian_interval * i, radian_interval * j,
                                                            average_num)
            # print("\r Loading  ", end='row   ' + str(i) + " and column  " + str(j) + " of all  " + str(N ** 2 * 2))
            spherical_matrix.append([griddata[i][j], radian_interval * i, radian_interval * j])

    if is_return_xyz:
        return griddata, np.array(general_f.sph2descartes2(spherical_matrix))
    return griddata, np.array(spherical_matrix)


def get_nib_embryo_membrane_dict(embryo_path, file_name):
    '''

    :param file_name: with timepoint like Embryo04_010_segCell.nii.gz
    :return:
    '''
    if os.path.exists(os.path.join(config.dir_my_data, 'membrane' + file_name)):
        img = general_f.load_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + file_name))
    else:
        img = cell_f.nii_get_cell_surface(general_f.load_nitf2_img(os.path.join(file_name, file_name)),
                                          file_name)  # calculate membrane and save automatically

    dict_img_membrane = {}
    img_membrane_data = img.get_fdata().astype(np.int16)
    x_num, y_num, z_num = img_membrane_data.shape
    # -------------get each cell membrane----------------
    for x in range(x_num):
        for y in range(y_num):
            for z in range(z_num):
                dict_key = img_membrane_data[x][y][z]

                if dict_key != 0:
                    # print(file_name,dict_key)

                    if dict_key in dict_img_membrane:
                        dict_img_membrane[dict_key].append([x, y, z])
                    else:
                        dict_img_membrane[dict_key] = [[x, y, z]]
    # ----------------------
    # print(dict_img_membrane)

    # -------------get each full cell----------------
    if os.path.exists(os.path.join(embryo_path, file_name)):
        img = general_f.load_nitf2_img(os.path.join(embryo_path, file_name))
    else:
        print('reading ， not embryo file!!!!!error')
        return EOFError  # calculate cell and save automatically

    dict_img_cell_calculate = {}
    img_cell_data = img.get_fdata().astype(np.int16)
    for x in range(x_num):
        for y in range(y_num):
            for z in range(z_num):
                dict_key = img_cell_data[x][y][z]
                if dict_key != 0:
                    # print(file_name,dict_key)

                    if dict_key in dict_img_cell_calculate:
                        dict_img_cell_calculate[dict_key].append([x, y, z])
                    else:
                        dict_img_cell_calculate[dict_key] = [[x, y, z]]
    # ---------------------------------------------------
    # print(dict_img_cell_calculate.keys())
    dict_center_points = {}
    for dict_key in dict_img_cell_calculate.keys():
        center_point = np.sum(dict_img_cell_calculate[dict_key], axis=0) / len(dict_img_cell_calculate[dict_key])
        if center_point is None:
            center_point = [0, 0, 0]
        # print(center_point)
        dict_center_points[dict_key] = center_point
    # print(dict_img_membrane)

    # print(dict_center_points)
    return dict_img_membrane, dict_center_points


def get_SH_coefficient_of_embryo(embryo_path, file_name, sample_N, lmax,
                                 surface_average_num=3):
    """

    :param sample_N: how many samples we need
    :param embryo_path: the shape img path
    :param file_name: file name
    :param surface_average_num: how many points to get average num to calculate R=f(phi,theta)
    :return:
    """
    embryo_array=general_f.load_nitf2_img(os.path.join(embryo_path, file_name)).get_fdata().astype(int)
    cell_keys=np.unique(embryo_array)

    number_cell_affine_table, _ = cell_f.get_cell_name_affine_table()
    # THE DEGREE OF SH. we can specify it or less than N/2-1, or 10 -- 2*(10/2-1)+1=9 coefficients
    # lmax = int(sample_N / 2 - 1)

    # this_embryo_dir = os.path.join(embryo_path, 'SH_C_folder_OF' + file_name)
    # print('placing SH file in====>', this_embryo_dir)
    # if not os.path.exists(this_embryo_dir):
    #     os.makedirs(this_embryo_dir)
    print(os.path.basename(file_name))

    folder_tmp = os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv/SPHARM', os.path.basename(embryo_path))
    if not os.path.exists(folder_tmp):
        os.mkdir(folder_tmp)


    column_indices=get_flatten_ldegree_morder(lmax)
    pd_embryo=pd.DataFrame(columns=column_indices)

    # --------------------------calculate all and save as csv --------------------------
    for dict_key in cell_keys:
        if dict_key:
            cell_surface,center= cell_f.nii_get_cell_surface(embryo_array, dict_key)
            # get center point
            points_membrane_local = cell_surface - center
            griddata, _ = do_sampling_with_interval(sample_N, points_membrane_local, surface_average_num)

            # do fourier transform and convolution on SPHERE
            print('---------dealing with cell ' + str(dict_key) + '-----' + number_cell_affine_table[
                dict_key] + '   ---coefficient --------------------')
            # calculate coefficients from points
            cilm = flatten_clim(pysh.shtools.SHExpandDH(griddata, sampling=2, lmax_calc=lmax))
            # build sh tools coefficient class instance
            sh_coefficient = pysh.SHCoeffs.from_array(cilm)
            print(cilm)
            # pd_embryo.iloc[]



def sample_and_SHc_with_surface(surface_points, sample_N, lmax, surface_average_num=5):
    center_points = np.sum(surface_points, axis=0) / len(surface_points)
    if center_points is None:
        center_points = [0, 0, 0]
    points_surface_local = surface_points - center_points
    griddata, _ = do_sampling_with_interval(sample_N, points_surface_local, surface_average_num)
    # do fourier transform and convolution on SPHERE
    print('---------dealing with surface point coefficient --------------------')
    # calculate coefficients from points
    cilm = pysh.shtools.SHExpandDH(griddata, sampling=2, lmax_calc=lmax)
    # build sh tools coefficient class instance
    sh_coefficient_instance = pysh.SHCoeffs.from_array(cilm)
    return sh_coefficient_instance
