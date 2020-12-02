import math

import functional_func.general_func as general_f
import functional_func.cell_func as cell_f
import functional_func.draw_func as draw_f
import functional_func.spherical_func as spherical_f
import config
import os
import numpy as np
import time
from multiprocessing import Process


def build_R_array_for_embryo(ray_num=1000, img_path=config.dir_segemented_tmp1, file_name='Embryo04_001_segCell.nii.gz',
                             csv_path=config.dir_my_data_R_matrix_csv,
                             save_name='Embryo04_001_segCell.nii.gz' + str(time.time()), surface_average_num=10):
    """
    the file read must have been calculate with erosion to find out surface
    :param surface_average_num: the number of points to count the average R of the ray
    :param ray_num: the numbers of fibonacci points on surface of sphere -- the ray from center, directions to fibonacci points
    :param img_path: the cell surface path
    :param file_name: the cell surface filename
    :param csv_path: the cell R matrix surface path
    :param save_name: the cell R matrix saved surface filename
    :return:
    """

    if os.path.exists(os.path.join(config.dir_my_data, 'membrane' + file_name)):
        img = general_f.load_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + file_name))
    else:
        img = cell_f.nii_get_cell_surface(img_path, file_name)  # calculate membrane and save automatically

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

    # -------------get each cell volume----------------
    if os.path.exists(os.path.join(img_path, file_name)):
        img = general_f.load_nitf2_img(os.path.join(img_path, file_name))
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
    # ----------------------
    #
    # dict_key = list(dict_img_membrane_calculate.keys())[3]

    # for dict_key in dict_img_calculate.keys():
    #     points_num_ = len(dict_img_calculate[dict_key])
    #     dict_img_calculate[dict_key] = np.array(dict_img_calculate[dict_key])
    #     center_points = np.sum(dict_img_calculate[dict_key], axis=0) / points_num_
    #     print(center_points)
    #     p = Process(target=draw_pack.draw_3D_points_in_new_coordinate,
    #                 args=(dict_img_calculate[dict_key], center_points,))
    #     p.start()
    for dict_key in dict_img_membrane_calculate.keys():
        points_num_this_membrane = len(dict_img_membrane_calculate[dict_key])
        # points_num_this_cell=
        dict_img_membrane_calculate[dict_key] = np.array(dict_img_membrane_calculate[dict_key])
        center_points = np.sum(dict_img_cell_calculate[dict_key], axis=0) / len(dict_img_cell_calculate[dict_key])
        # print(center_points)
        # draw_f.draw_3D_points_in_new_coordinate(dict_img_calculate[dict_key], center_points)
        if center_points is None:
            center_points = [0, 0, 0]
        points_self = dict_img_membrane_calculate[dict_key] - center_points
        p = Process(target=build_R_matrix_with_ray_sampling,
                    args=(
                        points_self, ray_num, points_num_this_membrane, surface_average_num, file_name + str(dict_key)))
        p.start()
        # draw_f.draw_3D_points(points_self)
        # representation_matrix_xyz=build_R_matrix_for_each_cell(points_self,ray_num,points_num_this_cell,surface_average_num)


def build_R_matrix_with_ray_sampling(points_cell_membrane, ray_num, points_num_this_cell, surface_average_num,
                                     cell_name):
    points_self = general_f.descartes2spherical2(points_cell_membrane)
    points_at_spherical_lat_phi, points_at_spherical_lon_theta = spherical_f.sort_by_phi_theta(points_self)
    # print(points_self)
    # print(points_at_spherical_lat_phi)
    # print(points_at_spherical_lon_theta)

    # ray_xyz = spherical_f.fibonacci_sphere(ray_num)
    ray_sph = general_f.descartes2spherical2(spherical_f.fibonacci_sphere(ray_num))

    # let's find the point closest with the ray!
    probable_interval_num = int(points_num_this_cell / ray_num)
    # print(probable_interval_num)

    ray_representation_matrix = []
    for i in range(ray_sph.shape[0]):
        # pass
        # if math.fabs(i[1] - i[2]) <= 0.01:
        #     print(i)

        ray_lat = ray_sph[i][1]
        ray_lon = ray_sph[i][2]

        average_R = spherical_f.calculate_R_with_average_with_my_locate_method(points_at_spherical_lat_phi,
                                                                               points_at_spherical_lon_theta,
                                                                               ray_lat, ray_lon, probable_interval_num,
                                                                               surface_average_num)
        print(average_R)
        ray_representation_matrix.append([average_R, ray_lat, ray_lon])

    ray_representation_matrix = np.array(ray_representation_matrix)
    representation_matrix_xyz = general_f.sph2descartes2(ray_representation_matrix)
    draw_f.draw_3D_points(representation_matrix_xyz, fig_name=cell_name)

    # # -------------draw R matrix curve on 2D way-------
    # R_curve = ray_representation_matrix.copy()
    # R_curve[:, [0, 2]] = R_curve[:, [2, 0]]
    # draw_f.draw_3D_curve_with_triangle(R_curve, fig_name=cell_name)

    return representation_matrix_xyz
    # -------------------------------------

    # 先把surface r phi theta 按照 phi thera 排序算出来， 根据 ray 的 theta phi 取surface的一个点距（折算到index中，取3-4个 最近的点做R的平均，它就是intersection
    # 今天先把这个xyz-> r phi theta 写出来，检查一下ray的theta phi是怎么回事，会不会超过2pi
