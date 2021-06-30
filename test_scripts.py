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

from math import pi


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


def test_2021_6_20():
    print('hello 2021 6 20')
    points = geo_f.get_sample_on_geometric_object(r'./DATA/template_shape_stl/unit-cube.solid')
    instance = sh_represent.sample_and_SHc_with_surface(surface_points=points, sample_N=30, lmax=14,
                                                        surface_average_num=3)
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
            phi=pi / 8), theta=pi / 8),))
    p2.start()

    p3 = Process(target=draw_pack.draw_3D_points,
                 args=(sh_analysis.do_reconstruction_for_SH(sample_N=100,
                                                            sh_coefficient_instance=instance.rotate(alpha=-(pi / 8),
                                                                                                    beta=0,
                                                                                                    gamma=0,
                                                                                                    convention='x',
                                                                                                    degrees=False).rotate(
                                                                alpha=0,
                                                                beta=pi / 8,
                                                                gamma=0,
                                                                convention='x',
                                                                degrees=False)),))
    p3.start()

    p3 = Process(target=draw_pack.draw_3D_points,
                 args=(sh_analysis.do_reconstruction_for_SH(sample_N=100,
                                                            sh_coefficient_instance=instance.rotate(alpha=-(pi / 8),
                                                                                                    beta=pi / 8,
                                                                                                    gamma=0,
                                                                                                    convention='x',
                                                                                                    degrees=False)),))
    p3.start()

    print(sh_analysis.flatten_clim(instance))
    print(sh_analysis.flatten_clim(instance.rotate(alpha=-(pi / 8),
                                                   beta=pi / 8,
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


# test the regular shape rotation, rotation invariance
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

    instance1 = instance.rotate(alpha=-(pi / 8),
                                beta=pi / 8,
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

    instance2 = instance.rotate(alpha=-(pi / 8) * 2,
                                beta=(pi / 8) * 2,
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

    instance3 = instance.rotate(alpha=-(pi / 8) * 3,
                                beta=(pi / 8) * 3,
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

    instance4 = instance.rotate(alpha=-(pi / 8) * 4,
                                beta=(pi / 8) * 4,
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

    instance5 = instance.rotate(alpha=-(pi / 8) * 5,
                                beta=(pi / 8) * 5,
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

    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_norm_PCA.csv')
    if not os.path.exists(PCA_matrices_saving_path):
        path_saving_csv_normalized = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_norm.csv')
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


def test_2021_6_30_2():
    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_norm_PCA.csv')

    print('PCA exist')
    means, variation, n_components = PCA_f.read_PCA_file(PCA_matrices_saving_path)
    print(means)
    print(variation)
    print(n_components)


def test_2021_6_30_3():
    general_f.combine_all_embryo_SHc_in_df(dir_my_data_SH_time_domain_csv=config.dir_my_data_SH_time_domain_csv,
                                           is_norm=False)


if __name__ == "__main__":
    test_2021_6_30_2()
