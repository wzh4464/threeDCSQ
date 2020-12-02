import functional_func.general_func as general_f
import functional_func.cell_func as cell_f
import functional_func.draw_func as draw_f

from matplotlib import pyplot as plt
import pyshtools as pysh
import numpy as np
import multiprocessing
import config
import os
import math


def analysis_with_img(embryo_path, file_name):
    _, cell_name_to_No_dict = cell_f.get_cell_name_affine_table()
    this_embryo_dir = os.path.join(embryo_path, 'SH_C_folder_OF' + file_name)

    # find all  cells in this embryos dir
    cells_list = os.walk(this_embryo_dir)

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

    # -------------get each full cell ----------------
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

    # -----------------------------------draw to see ----------------------------------------
    # fig = plt.figure(figsize=(6, 6))
    # _, ax = plt.subplots(2, 3)

    for _, _, file_name_list in cells_list:
        for cell_name in file_name_list:
            cell_SH_path = os.path.join(this_embryo_dir, cell_name)
            sh_coefficient_instance = pysh.SHCoeffs.from_file(cell_SH_path, lmax=2)
            # print()
            print(sh_coefficient_instance.errors)

            # ------------------display coefficients distribution--------------------
            # if cell_name[0] == 'A':
            #     sh_coefficient_instance.plot_spectrum(ax=ax[0, 0], legend=cell_name)
            # if cell_name[0] == 'C':
            #     sh_coefficient_instance.plot_spectrum(ax=ax[0, 1], legend=cell_name)
            # if cell_name[0] == 'D':
            #     sh_coefficient_instance.plot_spectrum(ax=ax[0, 2], legend=cell_name)
            # if cell_name[0] == 'E':
            #     sh_coefficient_instance.plot_spectrum(ax=ax[1, 0], legend=cell_name)
            # if cell_name[0] == 'M':
            #     sh_coefficient_instance.plot_spectrum(ax=ax[1, 1], legend=cell_name)
            # if cell_name[0] == 'P':
            #     sh_coefficient_instance.plot_spectrum(ax=ax[1, 2], legend=cell_name)

            # # ------------------2D representation-------------------------------
            # # grid.plot(cmap='RdBu', cmap_reverse=True, show=False,
            # #           title=cell_name_affine_table[dict_key] + 'regular')
            #   ------------------------------------------------------------------

            dict_key = cell_name_to_No_dict[cell_name]

            tmp_this_membrane = np.array(dict_img_membrane_calculate[dict_key])
            center_points = np.sum(dict_img_cell_calculate[dict_key], axis=0) / len(dict_img_cell_calculate[dict_key])
            if center_points is None:
                center_points = [0, 0, 0]
            points_membrane_local = tmp_this_membrane - center_points

            # # -----------------3D represent 2D curvature -----------------------
            # plane_representation_lat = np.arange(-90, 90, 180 / grid.data.shape[0])
            # plane_representation_lon = np.arange(0, 360, 360 / grid.data.shape[1])
            # plane_LAT, plane_LON = np.meshgrid(plane_representation_lat, plane_representation_lon)
            # fig = plt.figure(figsize=(6, 6))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot_surface(plane_LAT, plane_LON, grid.data.T)

            # # ------------------3D representation-------------------------------
            # do_contraction_image(sh_coefficient_instance, 50, points_membrane_local)
            # # -----------------------------------------------------------------

    # ------------------------------------------------------------------------------------------


def do_contraction_image(sh_coefficient_instance, sh_show_N, points_membrane_local):

    plane_representation_lat = np.arange(-90, 90, 180 / sh_show_N)
    plane_representation_lon = np.arange(0, 360, 360 / (2 * sh_show_N))
    plane_LAT, plane_LON = np.meshgrid(plane_representation_lat, plane_representation_lon)

    plane_LAT_FLATTEN=plane_LAT.flatten(order='F')
    plane_LON_FLATTEN=plane_LON.flatten(order='F')
    # print(plane_LAT_FLATTEN)
    # print(plane_LON_FLATTEN)
    grid = sh_coefficient_instance.expand(lat=plane_LAT_FLATTEN, lon=plane_LON_FLATTEN)

    plane_LAT_FLATTEN=plane_LAT_FLATTEN/180*math.pi
    plane_LON_FLATTEN=plane_LON_FLATTEN/180*math.pi

    # grid = sh_coefficient_instance.expand()
    # print(grid.data.shape)

    reconstruction_matrix = []
    # lat_interval = math.pi / grid.data.shape[0]
    # lon_interval = 2 * math.pi / grid.data.shape[1]
    ratio_interval = math.pi / sh_show_N
    for i in range(grid.data.shape[0]):
        # for j in range(grid.data.shape[1]):
        reconstruction_matrix.append([grid.data[i], plane_LAT_FLATTEN[i], plane_LON_FLATTEN[i]])

    p = multiprocessing.Process(target=draw_f.draw_3D_points, args=(points_membrane_local,))
    p.start()
    draw_f.draw_3D_points(general_f.sph2descartes(np.array(reconstruction_matrix)))



