import functional_func.draw_func as draw_pack
import functional_func.spherical_func as sphe_pack
import functional_func.general_func as general_f
import functional_func.cell_func as cell_f
import config
import numpy as np
import os
from multiprocessing import Process

def test_06_10_2020_2():
    img_2 = general_f.load_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + 'Embryo04_001_segCell.nii.gz'))

    cnt_volume, cnt_surface = cell_f.nii_count_volume_surface(img_2)
    dict_img_2_calculate = {}
    img_2_data = img_2.get_fdata().astype(np.int16)
    x_num, y_num, z_num = img_2_data.shape
    # print(img_2_data.dtype)
    for x in range(x_num):
        for y in range(y_num):
            for z in range(z_num):
                dict_key = img_2_data[x][y][z]
                if dict_key != 0:
                    if dict_key in dict_img_2_calculate:
                        dict_img_2_calculate[dict_key].append([x, y, z])
                    else:
                        dict_img_2_calculate[dict_key] = [[x, y, z]]

    #         test           --------------------------build all and single----------------------------------------
    dict_key = list(dict_img_2_calculate.keys())[0]

    for dict_key in dict_img_2_calculate.keys():
        points_num_ = len(dict_img_2_calculate[dict_key])
        dict_img_2_calculate[dict_key] = np.array(dict_img_2_calculate[dict_key])
        center_points = np.sum(dict_img_2_calculate[dict_key], axis=0) / points_num_
        print(center_points)
        p=Process(target=draw_pack.draw_3D_points_in_new_coordinate,args=(dict_img_2_calculate[dict_key],center_points,))
        p.start()

    # points_num_ = len(dict_img_2_calculate[dict_key])
    # dict_img_2_calculate[dict_key] = np.array(dict_img_2_calculate[dict_key])
    # center_points = np.sum(dict_img_2_calculate[dict_key], axis=0) / points_num_
    # print(center_points)
    # draw_pack.draw_3D_points_in_new_coordinate(dict_img_2_calculate[dict_key], center_points)


def test_06_10_2020_1():
    spherical_fibonacci_1, _ = sphe_pack.fibonacci_sphere(500)
    # draw_pack.draw_3D_curve(spherical_fibonacci)
    p1=Process(target=draw_pack.draw_3D_points,args=(spherical_fibonacci_1,))
    p1.start()

    sphere_points = sphe_pack.average_lat_lon_sphere()
    p2 = Process(target=draw_pack.draw_3D_points, args=(sphere_points,))
    p2.start()
