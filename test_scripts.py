from typing import Optional, Any, Union, Tuple

import functional_func.draw_func as draw_pack
import functional_func.spherical_func as sphe_pack
import functional_func.general_func as general_f
import functional_func.cell_func as cell_f
import config
import numpy as np
import os
import pandas as pd

from multiprocessing import Process
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import functional_func.draw_func as draw_f
import multiprocessing
import pyshtools as pysh
import particular_func.SH_represention as sh_represent
import particular_func.PCA as PCA_f
import particular_func.SH_analyses as sh_analysis
import functional_func.geometry as geo_f
import matplotlib.pyplot as plt

from tqdm import tqdm

import math
import random
import numpy.linalg as la


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
        p = Process(target=draw_pack.draw_3D_points_in_new_coordinate,
                    args=(dict_img_2_calculate[dict_key], center_points,))
        p.start()

    # points_num_ = len(dict_img_2_calculate[dict_key])
    # dict_img_2_calculate[dict_key] = np.array(dict_img_2_calculate[dict_key])
    # center_points = np.sum(dict_img_2_calculate[dict_key], axis=0) / points_num_
    # print(center_points)
    # draw_pack.draw_3D_points_in_new_coordinate(dict_img_2_calculate[dict_key], center_points)


def test_06_10_2020_1():
    spherical_fibonacci_1, _ = sphe_pack.fibonacci_sphere(500)
    # draw_pack.draw_3D_curve(spherical_fibonacci)
    p1 = Process(target=draw_pack.draw_3D_points, args=(spherical_fibonacci_1,))
    p1.start()

    sphere_points = sphe_pack.average_lat_lon_sphere()
    p2 = Process(target=draw_pack.draw_3D_points, args=(sphere_points,))
    p2.start()


def test_11_1_2021():
    # ------------------------------sh become smaller or bigger------------------------------

    sh_instance_original = pysh.SHCoeffs.from_zeros(10)
    sh_instance_original.coeffs[0, 10, 0] = 100.
    sh_instance_original.coeffs[1, 3, 0] = 100.
    sh_instance_original.coeffs[0, 9, 0] = 100.

    sh_instance_modified = pysh.SHCoeffs.from_zeros(10)
    sh_instance_modified.coeffs[0, 10, 0] = 10.
    sh_instance_modified.coeffs[1, 3, 0] = 10.
    sh_instance_modified.coeffs[0, 9, 0] = 10.

    sh_instance_modified_reconstruction = sh_analysis.do_reconstruction_for_SH(20, sh_instance_modified)
    p = multiprocessing.Process(target=draw_f.draw_3D_points,
                                args=(sh_instance_modified_reconstruction, '10',))
    p.start()

    sh_instance_original_reconstruction = sh_analysis.do_reconstruction_for_SH(20, sh_instance_original)
    draw_f.draw_3D_points(sh_instance_original_reconstruction, fig_name='1')

    # ---------------------------------------------------------------------------------------------------------


def test_2021_6_15():
    print('hello 2021 6 14')
    points = geo_f.get_sample_on_geometric_object(r'./DATA/template_shape_stl/unit-regular-tetrahedron.solid')
    instance = sh_represent.sample_and_SHc_with_surface(surface_points=points, sample_N=30, lmax=14,
                                                        surface_average_num=3)
    draw_f.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance))


# regular shape testing robustness
def test_2021_6_20():
    print('hello 2021 6 20')
    points = geo_f.get_sample_on_geometric_object(r'./DATA/template_shape_stl/unit-cube.solid')
    instance = sh_represent.sample_and_SHc_with_surface(surface_points=points, sample_N=30, lmax=14,
                                                        surface_average_num=5)
    p1 = Process(target=draw_pack.draw_3D_points,
                 args=(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance),))
    p1.start()
    # p2 = Process(target=draw_pack.draw_3D_points, args=(
    #     general_f.rotate_points_lat(general_f.rotate_points_lon(
    #         sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance),
    #         phi=pi / 8), theta=pi / 8),))
    # p2.start()
    p2 = Process(target=draw_pack.draw_3D_points, args=(
        general_f.rotate_points_lat(general_f.rotate_points_lon(
            sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance),
            phi=math.pi / 8), theta=math.pi / 8),))
    p2.start()

    p3 = Process(target=draw_pack.draw_3D_points,
                 args=(sh_analysis.do_reconstruction_for_SH(sample_N=100,
                                                            sh_coefficient_instance=instance.rotate(
                                                                alpha=-(math.pi / 8),
                                                                beta=0,
                                                                gamma=0,
                                                                convention='x',
                                                                degrees=False).rotate(
                                                                alpha=0,
                                                                beta=math.pi / 8,
                                                                gamma=0,
                                                                convention='x',
                                                                degrees=False)),))
    p3.start()

    p3 = Process(target=draw_pack.draw_3D_points,
                 args=(sh_analysis.do_reconstruction_for_SH(sample_N=100,
                                                            sh_coefficient_instance=instance.rotate(
                                                                alpha=-(math.pi / 8),
                                                                beta=math.pi / 8,
                                                                gamma=0,
                                                                convention='x',
                                                                degrees=False)),))
    p3.start()

    print(sh_analysis.flatten_clim(instance))
    print(sh_analysis.flatten_clim(instance.rotate(alpha=-(math.pi / 8),
                                                   beta=math.pi / 8,
                                                   gamma=0,
                                                   convention='x',
                                                   degrees=False)))

    # p2 = Process(target=draw_pack.draw_3D_points, args=(
    #     general_f.rotate_points_lon(
    #         sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance), phi=pi / 8),))
    # p2.start()
    # p3 = Process(target=draw_pack.draw_3D_points,
    #              args=(sh_analysis.do_reconstruction_for_SH(sample_N=100,
    #                                                         sh_coefficient_instance=instance.rotate(alpha=-(pi / 8),
    #                                                                                                 beta=0,
    #                                                                                                 gamma=0,
    #                                                                                                 convention='x',
    #                                                                                                 degrees=False)),))
    # p3.start()


# Kmeans clustering effective for regular shape
def test_2021_6_21():
    # index would be {list_index}::{theta}::{phi}
    df_regular_polyhedron_sh = pd.DataFrame(columns=sh_analysis.get_flatten_ldegree_morder(degree=17))
    # print(len(sh_analysis.get_flatten_ldegree_morder(degree=17)))
    # df_regular_polyhedron_points
    for regular_shape_index, item_name in enumerate(geo_f.regular_polyhedron_list):
        basic_points = geo_f.get_sample_on_geometric_object(os.path.join(r'./DATA/template_shape_stl', item_name))
        regular_instance = sh_represent.sample_and_SHc_with_surface(surface_points=basic_points, sample_N=36, lmax=17,
                                                                    surface_average_num=3)
        # using degree here
        for theta_tmp in np.arange(start=0, stop=180, step=9):
            for phi_tmp in np.arange(start=0, stop=360, step=9):
                df_regular_polyhedron_sh.loc[
                    str(regular_shape_index) + '::' + str(theta_tmp) + '::' + str(phi_tmp)] = sh_analysis.flatten_clim(
                    regular_instance.rotate(
                        alpha=-(theta_tmp),
                        beta=phi_tmp,
                        gamma=0,
                        convention='x'))  # using degree here
    print(df_regular_polyhedron_sh)
    df_regular_polyhedron_sh.to_csv(os.path.join(config.dir_my_regular_shape_path, '5_regular_gap_9degree_sh.csv'))
    estimator1 = KMeans(n_clusters=5, max_iter=1000)
    # estimator1.fit(df_SHcPCA_coeffs.values)
    result_1 = estimator1.fit_predict(df_regular_polyhedron_sh.values)
    df_kmeans_clustering = pd.DataFrame(index=df_regular_polyhedron_sh.index, columns=['cluster_num'])
    df_kmeans_clustering['cluster_num'] = result_1
    df_kmeans_clustering.to_csv(os.path.join(config.dir_my_regular_shape_path, '5_regular_gap_9degree_sh_cluster.csv'))


# plot the regular shape rotation, rotation invariance
def test_2021_6_21_2():
    print('hello 2021 6 20')
    points = geo_f.get_sample_on_geometric_object(r'./DATA/template_shape_stl/unit-regular-octahedron.solid')
    instance = sh_represent.sample_and_SHc_with_surface(surface_points=points, sample_N=34, lmax=16,
                                                        surface_average_num=3)

    # p1 = Process(target=draw_pack.draw_3D_points,
    #              args=(sh_analysis.do_reconstruction_for_SH(sample_N=100,
    #                                                         sh_coefficient_instance=instance1),))
    # p1.start()
    # pysh.utils.figstyle(rel_width=0.75)

    fig_1d_spectrum = plt.figure()
    fig_2d_spectrum = plt.figure()
    fig_points = plt.figure()

    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 1)
    instance.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 1)
    instance.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 1, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance),
                             ax=axes_tmp)
    # p1 = Process(target=draw_pack.draw_3D_points,
    #              args=(sh_analysis.do_reconstruction_for_SH(sample_N=100,
    #                                                         sh_coefficient_instance=instance),
    #                    "DEFAULT",
    #                    (10, 10),
    #                    ax[0, 0],))
    # p1.start()

    instance1 = instance.rotate(alpha=-(math.pi / 8),
                                beta=math.pi / 8,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 2)
    instance1.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 2)
    instance1.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 2, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance1),
                             ax=axes_tmp)

    instance2 = instance.rotate(alpha=-(math.pi / 8) * 2,
                                beta=(math.pi / 8) * 2,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 3)
    instance2.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 3)
    instance2.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 3, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance2),
                             ax=axes_tmp)

    instance3 = instance.rotate(alpha=-(math.pi / 8) * 3,
                                beta=(math.pi / 8) * 3,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 4)
    instance3.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 4)
    instance3.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 4, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance3),
                             ax=axes_tmp)

    instance4 = instance.rotate(alpha=-(math.pi / 8) * 4,
                                beta=(math.pi / 8) * 4,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 5)
    instance4.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 5)
    instance4.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 5, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance4),
                             ax=axes_tmp)

    instance5 = instance.rotate(alpha=-(math.pi / 8) * 5,
                                beta=(math.pi / 8) * 5,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 6)
    instance5.plot_cross_spectrum(clm=instance5, ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 6)
    instance5.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 6, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance5),
                             ax=axes_tmp)
    plt.show()
    # instance1.plot_cross_spectrum2d()
    # instance1.plot_spectrum2d()
    # instance1.plot_cross_spectrum2d()


# do PCA for all SHc and save the coefficient
def test_2021_6_21_3():
    used_degree = 25

    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_PCA.csv')
    if not os.path.exists(PCA_matrices_saving_path):
        path_saving_csv_normalized = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc.csv')
        df_SHc_norm = general_f.read_csv_to_df(path_saving_csv_normalized)
        print('finish read all embryo cell df_sh_norm_coefficients--------------')

        sh_PCA = PCA()
        sh_PCA.fit(df_SHc_norm.values[:, :(used_degree + 1) ** 2])
        sh_PCA_mean = sh_PCA.mean_
        print('PCA COMPONENTS: ', sh_PCA.n_components_)
        print('PCA EXPLAINED VARIANCE: ', sh_PCA.explained_variance_ratio_)

        df_PCA_matrices = pd.DataFrame(data=sh_PCA.components_,
                                       columns=sh_analysis.get_flatten_ldegree_morder(used_degree))
        df_PCA_matrices.insert(loc=0, column='explained_variation', value=list(sh_PCA.explained_variance_ratio_))
        df_PCA_matrices.loc['mean'] = [0] + list(sh_PCA_mean)
        df_PCA_matrices.to_csv(PCA_matrices_saving_path)

        PCA_f.draw_PCA(sh_PCA)
    else:
        print('PCA exist')
        means, variation, n_components = PCA_f.read_PCA_file(PCA_matrices_saving_path)
        print(means)
        print(variation)
        print(n_components)
    # RECOSTRUCT THE SHcPCA, draw tree fuck


def test_2021_6_22_1():
    for cell_index in np.arange(start=4, stop=21, step=1):
        path_tmp = r'./DATA/SegmentCellUnified04-20/Sample' + f'{cell_index:02}' + 'LabelUnified'
        print(path_tmp)
        sh_analysis.analysis_SHc_Kmeans_One_embryo(embryo_path=path_tmp, used_degree=16, cluster_num=4,
                                                   is_show_cluster=False)
        # sh_analysis.analysis_SHcPCA_KMEANS_clustering(embryo_path=path_tmp, used_degree=16, cluster_num=4)

        # sh_analysis.analysis_SHcPCA_energy_ratio(embryo_path=path_tmp, used_degree=9)
        # sh_analysis.analysis_SHcPCA_maximum_clustering(embryo_path=path_tmp, used_degree=16)


def test_2021_6_30_1():
    sh_analysis.analysis_compare_represent_method(embryo_path=config.dir_segemented_tmp1)


# calculate zk for embryo
def test_2021_6_30_2():
    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_norm_PCA.csv')

    for cell_index in np.arange(start=4, stop=21, step=1):
        path_tmp = r'./DATA/SegmentCellUnified04-20/Sample' + f'{cell_index:02}' + 'LabelUnified'
        print(path_tmp)
        PCA_f.calculate_PCA_zk_norm(embryo_path=path_tmp,
                                    PCA_matrices_saving_path=PCA_matrices_saving_path)


def test_2021_6_30_3():
    general_f.combine_all_embryo_SHc_in_df(dir_my_data_SH_time_domain_csv=config.dir_my_data_SH_time_domain_csv,
                                           is_norm=False)


# draw three methods contraction, figure plot
def test_2021_7_1_1():
    embryo_path = config.dir_segemented_tmp1
    k = 48  # 16+1 square
    degree = 12

    embryo_name = os.path.basename(embryo_path)

    # the whole PCA for embryo is the same one ,read before loop avoiding redundant reading
    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_norm_PCA.csv')
    pca_means, variation, df_n_components = PCA_f.read_PCA_file(PCA_matrices_saving_path)
    # the path need to change to non-norm path
    SHcPCA_path = os.path.join(config.dir_my_data_SH_PCA_csv, embryo_name + '_SHcPCA{}_norm.csv'.format(k))
    df_SHcPCA = general_f.read_csv_to_df(SHcPCA_path)
    # seg_files_path=[]
    No_cell, _ = cell_f.get_cell_name_affine_table()
    # ------------------------draw original, sample, SHc, SHcPCA reconstruction result--------------------
    # for file_name in reversed(os.listdir(embryo_path)):
    for file_name in os.listdir(embryo_path):

        if os.path.isfile(os.path.join(embryo_path, file_name)):
            path_embryo = os.path.join(embryo_path, file_name)
            print(path_embryo)
            tp = file_name.split('_')[1]
            # seg_files_path.append(file_name)
            dict_cell_membrane, dict_center_points = sh_represent.get_nib_embryo_membrane_dict(embryo_path, file_name)
            # print(dict_cell_membrane)
            # print(dict_center_points)
            for keys_tmp in dict_cell_membrane.keys():
                cell_name = No_cell[keys_tmp]
                idx_SHcPCA: str = tp + '::' + cell_name

                local_surface_points = dict_cell_membrane[keys_tmp] - dict_center_points[keys_tmp]
                fig = plt.figure()

                # original cell plot
                ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                draw_f.draw_3D_points(local_surface_points, fig_name='original' + idx_SHcPCA, ax=ax1)

                sh_show_N = 20
                # just for show 20 x 20 x 2 sample
                ax2 = fig.add_subplot(2, 2, 2, projection='3d')
                _, sample_surface = sh_represent.do_sampling_with_interval(sh_show_N, local_surface_points, 3,
                                                                           is_return_xyz=True)
                draw_f.draw_3D_points(sample_surface, fig_name='Sample' + idx_SHcPCA, ax=ax2)

                # SHc cell plot
                SHc_embryo_dir = os.path.join(embryo_path, 'SH_C_folder_OF' + file_name)
                cell_SH_path = os.path.join(SHc_embryo_dir, cell_name)
                sh_instance = pysh.SHCoeffs.from_file(cell_SH_path, lmax=degree)
                reconstruction_xyz = sh_analysis.do_reconstruction_for_SH(sh_show_N, sh_instance)
                ax3 = fig.add_subplot(2, 2, 3, projection='3d')
                draw_f.draw_3D_points(reconstruction_xyz, fig_name='SHc' + idx_SHcPCA, ax=ax3)

                # SHcPCA plot

                zk = df_SHcPCA.loc[idx_SHcPCA]
                # x_hat is a sh coefficients instance
                x_hat = df_n_components.values[:k].T.dot(zk) + pca_means
                sh_instance = pysh.SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(x_hat))
                reconstruction_xyz = sh_analysis.do_reconstruction_for_SH(sh_show_N, sh_instance)
                ax4 = fig.add_subplot(2, 2, 4, projection='3d')
                draw_f.draw_3D_points(reconstruction_xyz, fig_name='SHcPCA' + idx_SHcPCA, ax=ax4)

                plt.show()


def test_2021_7_1_2():
    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_PCA.csv')

    PCA_f.calculate_PCA_zk_norm(embryo_path=config.dir_segemented_tmp1,
                                PCA_matrices_saving_path=PCA_matrices_saving_path, k=12)


# draw three methods contraction, error estimate # TIME CONSUMING AND
def test_2021_7_2_1():
    # draw three methods contraction
    embryo_path = config.dir_segemented_tmp1

    # degree = 16

    embryo_name = os.path.basename(embryo_path)
    # the whole PCA for embryo is the same one ,read before loop avoiding redundant reading
    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_PCA.csv')
    pca_means, variation, df_n_components = PCA_f.read_PCA_file(PCA_matrices_saving_path)
    # the path need to change to non-norm path

    # seg_files_path=[]
    No_cell, _ = cell_f.get_cell_name_affine_table()

    l_degree_range = np.arange(5, 25, 1)

    df_err = pd.DataFrame(columns=['outline', 'outlinePCA', 'SHc', 'SHcPCA'])

    # ------calculate sample, SHc, SHcPCA reconstruction error by random, one cell one average error result--------
    # for file_name in reversed(os.listdir(embryo_path)):
    # for l_degree in l_degree_range:

    # N = 25
    # just for show 20 x 20 x 2 sample
    for file_name in os.listdir(embryo_path):
        if os.path.isfile(os.path.join(embryo_path, file_name)):
            path_embryo = os.path.join(embryo_path, file_name)
            tp = file_name.split('_')[1]
            # seg_files_path.append(file_name)
            dict_cell_membrane, dict_center_points = sh_represent.get_nib_embryo_membrane_dict(embryo_path,
                                                                                               file_name)
            # print(dict_cell_membrane)
            # print(dict_center_points)
            for l_degree in reversed(l_degree_range):

                N: int = int(math.sqrt((l_degree + 1) ** 2 / 2) + 1)
                k: int = l_degree ** 2  # 16+1 square
                SHcPCA_path = os.path.join(config.dir_my_data_SH_PCA_csv, embryo_name + '_SHcPCA{}.csv'.format(k))
                if not os.path.exists(SHcPCA_path):
                    PCA_f.calculate_PCA_zk(embryo_path, PCA_matrices_saving_path, k)
                df_SHcPCA = general_f.read_csv_to_df(SHcPCA_path)
                print(path_embryo, 'degree', l_degree)

                for keys_tmp in tqdm(dict_cell_membrane.keys()):
                    cell_name = No_cell[keys_tmp]
                    idx_ = tp + '::' + cell_name
                    idx = tp + '::' + cell_name + '::' + str(l_degree)

                    # co-latitude 0-math.pi
                    error_test_point_num = 1000
                    map_testing = [[random.uniform(0, math.pi), random.uniform(0, 2 * math.pi)] for i in
                                   range(error_test_point_num)]

                    # --------------------------ground truth from original---------------------------------------
                    local_surface_points = dict_cell_membrane[keys_tmp] - dict_center_points[keys_tmp]
                    R_from_lat_lon, original_xyz = sh_represent.do_sampling_with_lat_lon(local_surface_points,
                                                                                         map_testing,
                                                                                         average_num=10,
                                                                                         is_return_xyz=True)
                    # --------------------------outline extraction------------------------------------------------
                    _, sample_surface = sh_represent.do_sampling_with_interval(N, local_surface_points, 10,
                                                                               is_return_xyz=True)
                    R_sample, sample_xyz = sh_represent.do_sampling_with_lat_lon(sample_surface, map_testing,
                                                                                 average_num=1,
                                                                                 is_return_xyz=True)

                    # SHc cell plot
                    # --------------------------------shc-------------------------------------------------------
                    SHc_embryo_dir = os.path.join(embryo_path, 'SH_C_folder_OF' + file_name)
                    cell_SH_path = os.path.join(SHc_embryo_dir, cell_name)
                    sh_instance = pysh.SHCoeffs.from_file(cell_SH_path, lmax=l_degree)
                    # R_SHc = sh_analysis.do_reconstruction_for_SH(lat_num, sh_instance)
                    # print(type(map_random))
                    R_SHc, shc_sample_xyz = sh_analysis.get_points_with_SHc(sh_instance,
                                                                            colat=np.array(map_testing)[:, 0],
                                                                            lon=np.array(map_testing)[:, 1],
                                                                            is_return_xyz=True)

                    # # SHcPCA
                    # ------------------------------shcpca--------------------------------------------------------
                    zk = df_SHcPCA.loc[idx_]
                    x_hat = df_n_components.values[:k].T.dot(zk) + pca_means
                    sh_instance = pysh.SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(x_hat))
                    R_SHcPCA, shcpca_sample_xyz = sh_analysis.get_points_with_SHc(sh_instance,
                                                                                  colat=np.array(map_testing)[:, 0],
                                                                                  lon=np.array(map_testing)[:, 1],
                                                                                  is_return_xyz=True)

                    err_sample: Union[Tuple[Any, Optional[Any]], Any] = np.average(np.abs(R_sample - R_from_lat_lon))
                    err_SHc = np.average(np.abs(R_SHc - R_from_lat_lon))
                    err_SHcPCA = np.average(np.abs(R_SHcPCA - R_from_lat_lon))

                    # print(err_sample, 0, err_SHc, err_SHcPCA)
                    # err_SHcPCA = 0
                    # print(err_sample, err_SHc, err_SHcPCA)

                    df_err.loc[idx] = [err_sample, 0, err_SHc, err_SHcPCA]

    df_err.to_csv(os.path.join(config.dir_my_data_err_est_dir, embryo_name + 'test2.csv'))


# test sph2des and des2sph ; calculate zk
def test_2021_7_6_1():
    point_lat = [[1, -3 * math.pi / 8, math.pi / 4], [1, 3 * math.pi / 8, math.pi / 4]]
    print(point_lat)
    print(general_f.sph2descartes(point_lat))
    print(general_f.descartes2spherical(general_f.sph2descartes(point_lat)))

    point_lat = [[1, math.pi / 8, math.pi / 4], [1, 7 * math.pi / 8, math.pi / 4]]
    print(point_lat)

    print(general_f.sph2descartes2(point_lat))
    print(general_f.descartes2spherical2(general_f.sph2descartes2(point_lat)))

    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_PCA.csv')

    PCA_f.calculate_PCA_zk(embryo_path=config.dir_segemented_tmp1, PCA_matrices_saving_path=PCA_matrices_saving_path)


# calculate regular shape PCA and shcpca
def test_2021_7_7_1():
    # index would be {list_index}::{theta}::{phi}

    df_regular_polyhedron_sh_path = os.path.join(config.dir_my_regular_shape_path, '5_regular_random_degree_sh.csv')
    if not os.path.exists(df_regular_polyhedron_sh_path):
        df_regular_polyhedron_sh = pd.DataFrame(columns=sh_analysis.get_flatten_ldegree_morder(degree=16))
        # print(len(sh_analysis.get_flatten_ldegree_morder(degree=17)))
        # df_regular_polyhedron_points
        for regular_shape_index, item_name in enumerate(geo_f.regular_polyhedron_list):
            basic_points = geo_f.get_sample_on_geometric_object(os.path.join(r'./DATA/template_shape_stl', item_name))
            regular_instance = sh_represent.sample_and_SHc_with_surface(surface_points=basic_points, sample_N=36,
                                                                        lmax=16,
                                                                        surface_average_num=3)
            sample_num = 4000
            map_testing = [[random.uniform(0, math.pi), random.uniform(0, 2 * math.pi)] for i in
                           range(sample_num)]
            print('dealing with', item_name)
            for item_rotation in tqdm(map_testing):
                df_regular_polyhedron_sh.loc[
                    str(regular_shape_index) + '::' + str(item_rotation[0]) + '::' + str(
                        item_rotation[1])] = sh_analysis.flatten_clim(
                    regular_instance.rotate(alpha=-(item_rotation[0]), beta=item_rotation[1], gamma=0, convention='x',
                                            degrees=False))

        print(df_regular_polyhedron_sh)
        df_regular_polyhedron_sh.to_csv(df_regular_polyhedron_sh_path)
    else:
        df_regular_polyhedron_sh = general_f.read_csv_to_df(df_regular_polyhedron_sh_path)

    estimator1 = KMeans(n_clusters=5, max_iter=1000)
    # estimator1.fit(df_SHcPCA_coeffs.values)
    result_1 = estimator1.fit_predict(df_regular_polyhedron_sh.values)
    df_kmeans_clustering = pd.DataFrame(index=df_regular_polyhedron_sh.index, columns=['cluster_num'])
    df_kmeans_clustering['cluster_num'] = result_1
    df_kmeans_clustering.to_csv(
        os.path.join(config.dir_my_regular_shape_path, '5_regular_random_degree_sh_cluster.csv'))

    used_degree = 16
    k = 12
    PCA_matrices_saving_path = os.path.join(config.dir_my_regular_shape_path, 'SHc_PCA.csv')
    sh_PCA = PCA(n_components=k)
    sh_PCA.fit(df_regular_polyhedron_sh.values)
    sh_PCA_mean = sh_PCA.mean_
    # PCA_f.draw_PCA(sh_PCA)
    print('PCA COMPONENTS: ', sh_PCA.n_components_)
    print('PCA EXPLAINED VARIANCE: ', sh_PCA.explained_variance_ratio_)

    df_PCA_matrices = pd.DataFrame(data=sh_PCA.components_,
                                   columns=sh_analysis.get_flatten_ldegree_morder(used_degree))
    df_PCA_matrices.insert(loc=0, column='explained_variation', value=list(sh_PCA.explained_variance_ratio_))
    df_PCA_matrices.loc['mean'] = [0] + list(sh_PCA_mean)
    df_PCA_matrices.to_csv(PCA_matrices_saving_path)
    print(sh_PCA.components_.shape)

    df_SHcPCA = pd.DataFrame(columns=range(k))
    Q, R = la.qr(sh_PCA.components_.T)
    R_ = np.linalg.inv(R)
    # print(R.shape)
    # print(Q.shape)

    for y_idx in tqdm(df_regular_polyhedron_sh.index, desc='dealing with each cell'):
        y = df_regular_polyhedron_sh.loc[y_idx]
        # print(pca_means)
        y_u = y - sh_PCA_mean
        zk = R_.dot(Q.T.dot(y_u))
        # print(zk)
        # print(zk.shape)
        df_SHcPCA.loc[y_idx] = zk

    cluster_num = 5
    estimator2 = KMeans(n_clusters=cluster_num, max_iter=1000)
    # estimator1.fit(df_SHcPCA_coeffs.values)
    result_2 = estimator2.fit_predict(df_SHcPCA.values)
    df_kmeans_clustering = pd.DataFrame(index=df_SHcPCA.index, columns=['cluster_num'])
    df_kmeans_clustering['cluster_num'] = result_2
    df_kmeans_clustering.to_csv(
        os.path.join(config.dir_my_regular_shape_path,
                     '5_regular_random_degree_shcpca_cluster{}.csv'.format(cluster_num)))


def test_2021_7_8():
    PCA_matrices_saving_path = os.path.join(config.dir_my_regular_shape_path, 'SHc_PCA.csv')
    mean, variance, df_PCA = PCA_f.read_PCA_file(PCA_matrices_saving_path)

    k = 12
    # df_SHcPCA = pd.DataFrame(columns=range(k))
    Q, R = la.qr(df_PCA.values.T)
    R_ = np.linalg.inv(R)

    print('hello 2021 7 8')
    points = geo_f.get_sample_on_geometric_object(r'./DATA/template_shape_stl/unit-regular-tetrahedron.solid')
    instance = sh_represent.sample_and_SHc_with_surface(surface_points=points, sample_N=34, lmax=16,
                                                        surface_average_num=3)
    fig_1d_spectrum = plt.figure()
    fig_2d_spectrum = plt.figure()
    fig_points = plt.figure()

    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 1)
    instance.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 1)
    instance.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 1, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance),
                             ax=axes_tmp)
    # p1 = Process(target=draw_pack.draw_3D_points,
    #              args=(sh_analysis.do_reconstruction_for_SH(sample_N=100,
    #                                                         sh_coefficient_instance=instance),
    #                    "DEFAULT",
    #                    (10, 10),
    #                    ax[0, 0],))
    # p1.start()

    instance1 = instance.rotate(alpha=-(math.pi / 8),
                                beta=math.pi / 8,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 2)
    instance1.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 2)
    instance1.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 2, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance1),
                             ax=axes_tmp)

    instance2 = instance.rotate(alpha=-(math.pi / 8) * 2,
                                beta=(math.pi / 8) * 2,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 3)
    instance2.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 3)
    instance2.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 3, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance2),
                             ax=axes_tmp)

    instance3 = instance.rotate(alpha=-(math.pi / 8) * 3,
                                beta=(math.pi / 8) * 3,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 4)
    instance3.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 4)
    instance3.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 4, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance3),
                             ax=axes_tmp)

    instance4 = instance.rotate(alpha=-(math.pi / 8) * 4,
                                beta=(math.pi / 8) * 4,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 5)
    instance4.plot_spectrum(ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 5)
    instance4.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 5, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance4),
                             ax=axes_tmp)

    instance5 = instance.rotate(alpha=-(math.pi / 8) * 5,
                                beta=(math.pi / 8) * 5,
                                gamma=0,
                                convention='x',
                                degrees=False)
    axes_tmp = fig_1d_spectrum.add_subplot(2, 3, 6)
    instance5.plot_cross_spectrum(clm=instance5, ax=axes_tmp)
    axes_tmp = fig_2d_spectrum.add_subplot(2, 3, 6)
    instance5.plot_spectrum2d(ax=axes_tmp)
    axes_tmp = fig_points.add_subplot(2, 3, 6, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=100, sh_coefficient_instance=instance5),
                             ax=axes_tmp)
    plt.show()


# change shc in to 3D matrix
def test_2021_7_15():
    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_norm_PCA.csv')
    sh_PCA_mean, variance, df_PCA = PCA_f.read_PCA_file(PCA_matrices_saving_path)

    pca_number = 24
    # print(df_PCA.index)
    dense = 100

    for i in range(pca_number):

        # sh_instance = pysh.SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(df_PCA.loc[str(i)]))
        # matrix_3d=sh_analysis.generate_3D_matrix_from_SHc(sh_instance=sh_instance,dense=dense)
        #
        # fig=plt.figure()
        # ax_tmp=fig.add_subplot(1,2,1, projection='3d')
        # draw_f.draw_3D_points(sh_analysis.do_reconstruction_for_SH(100,sh_instance),ax=ax_tmp)
        #
        # tmp_draw=[]
        # for i in range(dense):
        #     for j in range(dense):
        #         for k in range(dense):
        #             if matrix_3d[i][j][k] == 1:
        #                 tmp_draw.append([i,j,k])
        # tmp_draw=np.array(tmp_draw)
        # ax_tmp=fig.add_subplot(1,2,2, projection='3d')
        # draw_f.draw_3D_points(tmp_draw,ax=ax_tmp)
        #
        # plt.show()

        component = df_PCA.loc[str(i)]

        for pca_change in np.arange(start=-5, stop=6, step=2):
            shc_instance = pysh.SHCoeffs.from_array(
                sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + pca_change * component)))
            matrix_3d = sh_analysis.generate_3D_matrix_from_SHc(sh_instance=shc_instance, dense=dense)
            saving_path = os.path.join(r'D:\cell_shape_quantification\DATA\my_data_csv\PCA_matrix',
                                       'norm_pca_' + str(i) + '_change_' + str(pca_change))
            # with open(saving_path, 'wb') as f:
            np.save(saving_path, matrix_3d)

import seaborn as sns

def test_2021_7_15_1():
    embryo_path = config.dir_segemented_tmp1

    # degree = 16

    embryo_name = os.path.basename(embryo_path)
    df_err=general_f.read_csv_to_df(os.path.join(config.dir_my_data_err_est_dir, embryo_name + 'test1.csv'))

    name_list=[]
    degree_list=[]
    for item in df_err.index:
        name_list.append(item.split('::')[1])
        degree_list.append(item.split('::')[2])
    df_err['name']=name_list
    df_err['degree']=degree_list

    reversed_df = df_err.iloc[::-1]
    print(df_err)
    print(reversed_df)

    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    ax.set_yscale("log")
    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="degree", y="SHcPCA", data=reversed_df,
                whis=[0, 100], width=.6, palette="vlag", )

    # Add in points to show each observation
    sns.stripplot(x="degree", y="SHcPCA", data=reversed_df,
                  size=1, color=".5", linewidth=0)

    # Tweak the visual presentation
    ax.yaxis.grid(True)
    ax.set(ylabel="average error")

    ax.set(xlabel="outline degree")
    sns.despine(trim=True, left=True)

    plt.show()

if __name__ == "__main__":
    test_2021_7_15_1()
