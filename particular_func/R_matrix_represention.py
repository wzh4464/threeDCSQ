import functional_func.general_func as general_f
import functional_func.cell_func as cell_f
import functional_func.draw_func as draw_f
import config
import os
import numpy as np
import time


def build_R_array_for_embryo(ray_num=1000, img_path=config.dir_segemented, file_name='Embryo04_001_segCell.nii.gz',
                             csv_path=config.dir_my_data_R_matrix_csv,
                             save_name='Embryo04_001_segCell.nii.gz' + str(time.time())):
    """
    the file read must have been calculate with erosion to find out surface
    :param ray_num: the numbers of fibonacci points on surface of sphere -- the ray from center, directions to fibonacci points
    :param img_path: the cell surface path
    :param file_name: the cell surface filename
    :param csv_path: the cell R matrix surface path
    :param save_name: the cell R matrix saved surface filename
    :return:
    """
    img = general_f.load_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + 'Embryo04_001_segCell.nii.gz'))

    dict_img_calculate = {}
    img_2_data = img.get_fdata().astype(np.int16)
    x_num, y_num, z_num = img_2_data.shape
    for x in range(x_num):
        for y in range(y_num):
            for z in range(z_num):
                dict_key = img_2_data[x][y][z]
                if dict_key != 0:
                    if dict_key in dict_img_calculate:
                        dict_img_calculate[dict_key].append([x, y, z])
                    else:
                        dict_img_calculate[dict_key] = [[x, y, z]]

    #         test           --------------------------build all and single----------------------------------------
    dict_key = list(dict_img_calculate.keys())[0]

    # for dict_key in dict_img_calculate.keys():
    #     points_num_ = len(dict_img_calculate[dict_key])
    #     dict_img_calculate[dict_key] = np.array(dict_img_calculate[dict_key])
    #     center_points = np.sum(dict_img_calculate[dict_key], axis=0) / points_num_
    #     print(center_points)
    #     p = Process(target=draw_pack.draw_3D_points_in_new_coordinate,
    #                 args=(dict_img_calculate[dict_key], center_points,))
    #     p.start()
    points_num_ = len(dict_img_calculate[dict_key])
    dict_img_calculate[dict_key] = np.array(dict_img_calculate[dict_key])
    center_points = np.sum(dict_img_calculate[dict_key], axis=0) / points_num_
    print(center_points)
    draw_f.draw_3D_points_in_new_coordinate(dict_img_calculate[dict_key], center_points)

    先把surface r phi theta 按照 phi thera 排序算出来， 根据 ray 的 theta phi 取surface的一个点距（折算到index中，取3-4个 最近的点做R的平均，它就是intersection
    今天先把这个xyz-> r phi theta 写出来，检查一下ray的theta phi是怎么回事，会不会超过2pi
