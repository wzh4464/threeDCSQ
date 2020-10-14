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


def build_R_array_for_embryo(ray_num=1000, img_path=config.dir_segemented, file_name='Embryo04_001_segCell.nii.gz',
                             csv_path=config.dir_my_data_R_matrix_csv,
                             save_name='Embryo04_001_segCell.nii.gz' + str(time.time()), surface_average_num=5):
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

    dict_img_calculate = {}
    img_2_data = img.get_fdata().astype(np.int16)
    x_num, y_num, z_num = img_2_data.shape
    # -------------get each cell----------------
    for x in range(x_num):
        for y in range(y_num):
            for z in range(z_num):
                dict_key = img_2_data[x][y][z]
                if dict_key != 0:
                    if dict_key in dict_img_calculate:
                        dict_img_calculate[dict_key].append([x, y, z])
                    else:
                        dict_img_calculate[dict_key] = [[x, y, z]]

    #
    dict_key = list(dict_img_calculate.keys())[3]

    # for dict_key in dict_img_calculate.keys():
    #     points_num_ = len(dict_img_calculate[dict_key])
    #     dict_img_calculate[dict_key] = np.array(dict_img_calculate[dict_key])
    #     center_points = np.sum(dict_img_calculate[dict_key], axis=0) / points_num_
    #     print(center_points)
    #     p = Process(target=draw_pack.draw_3D_points_in_new_coordinate,
    #                 args=(dict_img_calculate[dict_key], center_points,))
    #     p.start()
    points_num_this_cell = len(dict_img_calculate[dict_key])
    dict_img_calculate[dict_key] = np.array(dict_img_calculate[dict_key])
    center_points = np.sum(dict_img_calculate[dict_key], axis=0) / points_num_this_cell
    # print(center_points)
    # draw_f.draw_3D_points_in_new_coordinate(dict_img_calculate[dict_key], center_points)
    if center_points is None:
        center_points = [0, 0, 0]
    points_self = dict_img_calculate[dict_key] - center_points
    p = Process(target=build_R_matrix_for_each_cell,
                args=(points_self, ray_num, points_num_this_cell, surface_average_num,))
    p.start()
    draw_f.draw_3D_points(points_self)
    # representation_matrix_xyz=build_R_matrix_for_each_cell(points_self,ray_num,points_num_this_cell,surface_average_num)


def build_R_matrix_for_each_cell(points_cell, ray_num, points_num_this_cell, surface_average_num):
    points_self = general_f.descartes2spherical(points_cell)
    points_at_spherical_lat_phi, points_at_spherical_lon_theta = sort_by_phi_theta(points_self)
    # print(points_self)
    # print(points_at_spherical_lat_phi)
    # print(points_at_spherical_lon_theta)

    # ray_xyz = spherical_f.fibonacci_sphere(ray_num)
    ray_sph = general_f.descartes2spherical(spherical_f.fibonacci_sphere(ray_num))

    # let's find the point closest with the ray!
    probable_interval_num = int(points_num_this_cell / ray_num)
    # print(probable_interval_num)

    ray_representation_matrix = []
    for i in range(ray_sph.shape[0]):
        # pass
        # if math.fabs(i[1] - i[2]) <= 0.01:
        #     print(i)
        FLAG_AS_GOTO = False
        ray_lat = ray_sph[i][1]
        ray_lon = ray_sph[i][2]
        # flag to find the ray lat in points
        prob_ray_lat_index_in_points = int(ray_lat / math.pi * points_num_this_cell)
        print(prob_ray_lat_index_in_points)
        # -----------------deal with the border issues---------------------------- #
        if prob_ray_lat_index_in_points + probable_interval_num >= points_num_this_cell:
            prob_ray_lat_index_in_points -= (probable_interval_num + 1)
            FLAG_AS_GOTO = True
        elif prob_ray_lat_index_in_points - probable_interval_num <= 0:
            prob_ray_lat_index_in_points += (probable_interval_num + 1)
            FLAG_AS_GOTO = True
        elif ray_lon <= points_at_spherical_lat_phi[0][1]:
            prob_ray_lat_index_in_points = probable_interval_num
            FLAG_AS_GOTO = True
        elif ray_lon >= points_at_spherical_lat_phi[points_num_this_cell - 1][1]:
            prob_ray_lat_index_in_points = points_num_this_cell - (probable_interval_num + 1)
            FLAG_AS_GOTO = True
        # ---------------------------------------------------------------------- #

        if FLAG_AS_GOTO is False:
            # print(prob_ray_lat_index_in_points)
            pro_ray_index_pos = (0, 0)
            if points_at_spherical_lat_phi[prob_ray_lat_index_in_points - probable_interval_num][1] < ray_lat < \
                    points_at_spherical_lat_phi[prob_ray_lat_index_in_points + probable_interval_num][1]:
                pro_ray_index_pos = (0, 0)
                prob_ray_lat_index_in_points = prob_ray_lat_index_in_points - probable_interval_num
            elif points_at_spherical_lat_phi[prob_ray_lat_index_in_points + probable_interval_num][1] <= ray_lat:
                pro_ray_index_pos = (1, 0)
                prob_ray_lat_index_in_points = prob_ray_lat_index_in_points + probable_interval_num
            elif points_at_spherical_lat_phi[prob_ray_lat_index_in_points - probable_interval_num][1] >= ray_lat:
                pro_ray_index_pos = (0, 1)
                prob_ray_lat_index_in_points = prob_ray_lat_index_in_points - probable_interval_num

            while 1:
                if pro_ray_index_pos == (0, 0):
                    if points_at_spherical_lat_phi[prob_ray_lat_index_in_points][1] > ray_lat:
                        break
                    else:
                        prob_ray_lat_index_in_points += 1
                elif pro_ray_index_pos == (1, 0):
                    if points_at_spherical_lat_phi[prob_ray_lat_index_in_points][1] > ray_lat:
                        break
                    else:
                        prob_ray_lat_index_in_points += 1
                elif pro_ray_index_pos == (0, 1):
                    if points_at_spherical_lat_phi[prob_ray_lat_index_in_points][1] < ray_lat:
                        break
                    else:
                        prob_ray_lat_index_in_points -= 1
                else:
                    print('======lat====finding---ray----====error====================================')
        # print(ray_lat)
        # print(points_at_spherical_lat_phi[prob_ray_lat_index_in_points])
        # flag to find the ray lon in points
        FLAG_AS_GOTO = False
        prob_ray_lon_index_in_points = int(ray_lon / (math.pi * 2) * points_num_this_cell)
        # -----------------deal with the border issues---------------------------- #
        if prob_ray_lon_index_in_points + probable_interval_num >= points_num_this_cell:
            prob_ray_lon_index_in_points -= (probable_interval_num + 1)
            FLAG_AS_GOTO = True
        elif prob_ray_lon_index_in_points - probable_interval_num <= 0:
            prob_ray_lon_index_in_points += (probable_interval_num + 1)
            FLAG_AS_GOTO = True
        elif ray_lon <= points_at_spherical_lon_theta[0][2]:
            prob_ray_lon_index_in_points = probable_interval_num
            FLAG_AS_GOTO = True
        elif ray_lon >= points_at_spherical_lon_theta[points_num_this_cell - 1][2]:
            prob_ray_lon_index_in_points = points_num_this_cell - (probable_interval_num + 1)
            FLAG_AS_GOTO = True
        # ---------------------------------------------------------------------- #

        if FLAG_AS_GOTO is False:
            pro_ray_index_pos = (0, 0)
            if points_at_spherical_lon_theta[prob_ray_lon_index_in_points - probable_interval_num][2] < ray_lon < \
                    points_at_spherical_lon_theta[prob_ray_lon_index_in_points + probable_interval_num][2]:
                pro_ray_index_pos = (0, 0)
                prob_ray_lon_index_in_points = prob_ray_lon_index_in_points - probable_interval_num
            elif points_at_spherical_lon_theta[prob_ray_lon_index_in_points + probable_interval_num][2] <= ray_lon:
                pro_ray_index_pos = (1, 0)
                prob_ray_lon_index_in_points = prob_ray_lon_index_in_points + probable_interval_num
            elif points_at_spherical_lon_theta[prob_ray_lon_index_in_points - probable_interval_num][2] >= ray_lon:
                pro_ray_index_pos = (0, 1)
                prob_ray_lon_index_in_points = prob_ray_lon_index_in_points - probable_interval_num

            while 1:
                if pro_ray_index_pos == (0, 0):
                    if points_at_spherical_lon_theta[prob_ray_lon_index_in_points][2] > ray_lon:
                        break
                    else:
                        prob_ray_lon_index_in_points += 1
                elif pro_ray_index_pos == (1, 0):
                    if points_at_spherical_lon_theta[prob_ray_lon_index_in_points][2] > ray_lon:
                        break
                    else:
                        prob_ray_lon_index_in_points += 1
                elif pro_ray_index_pos == (0, 1):
                    if points_at_spherical_lon_theta[prob_ray_lon_index_in_points][2] < ray_lon:
                        break
                    else:
                        # if prob_ray_lon_index_in_points==0:
                        #     print('----------------------')
                        #     print(points_at_spherical_lon_theta[prob_ray_lon_index_in_points][2])
                        #     print(ray_lon)
                        #     print('----------------------')

                        prob_ray_lon_index_in_points -= 1
                else:
                    print('======lon====finding---ray----====error====================================')
        print(ray_lon)
        print(points_at_spherical_lon_theta[prob_ray_lon_index_in_points])

        prob_points_set = np.vstack((points_at_spherical_lon_theta[
                                     prob_ray_lon_index_in_points - probable_interval_num:prob_ray_lon_index_in_points + probable_interval_num,
                                     :], points_at_spherical_lat_phi[
                                         prob_ray_lat_index_in_points - probable_interval_num:prob_ray_lat_index_in_points + probable_interval_num,
                                         :]))
        # print(prob_points_set.shape[0])
        prob_set_distance = (prob_points_set[:, 1] - ray_lat) ** 2 + (prob_points_set[:, 2] - ray_lon) ** 2
        prob_points_set = np.hstack((prob_points_set, prob_set_distance.reshape((prob_set_distance.shape[0], 1))))
        # print(prob_points_set[:, 3].argsort())
        prob_points_set = prob_points_set[prob_points_set[:, 3].argsort()]
        # print(prob_points_set)
        # print(prob_points_set[:surface_average_num, 0])
        average_R = np.average(prob_points_set[:surface_average_num, 0])
        print(average_R)
        ray_representation_matrix.append([average_R, ray_lat, ray_lon])

    representation_matrix_xyz = general_f.sph2descartes(np.array(ray_representation_matrix))
    draw_f.draw_3D_points(representation_matrix_xyz)

    return representation_matrix_xyz


def sort_by_phi_theta(points_at_spherical):
    # from small-> large
    points_at_spherical_phi = points_at_spherical[points_at_spherical[:, 1].argsort()]
    points_at_spherical_theta = points_at_spherical[points_at_spherical[:, 2].argsort()]
    return points_at_spherical_phi, points_at_spherical_theta

    # 先把surface r phi theta 按照 phi thera 排序算出来， 根据 ray 的 theta phi 取surface的一个点距（折算到index中，取3-4个 最近的点做R的平均，它就是intersection
    # 今天先把这个xyz-> r phi theta 写出来，检查一下ray的theta phi是怎么回事，会不会超过2pi
