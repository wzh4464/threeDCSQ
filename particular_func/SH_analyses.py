import random

import functional_func.general_func as general_f
import functional_func.cell_func as cell_f
import functional_func.draw_func as draw_f
import particular_func.SH_represention as SH_represention

import numpy.linalg as la
import scipy.linalg as spla

from matplotlib import pyplot as plt
import pyshtools as pysh
import numpy as np
import pandas as pd
import multiprocessing
import config
import os
import math
import re

from collections import Counter

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

cluster_AB_list = ['ABa', 'ABp', 'ABal', 'ABar', 'ABpl', 'ABpr', 'ABala', 'ABalp', 'ABara', 'ABarp', 'ABpla', 'ABplp',
                   'ABpra', 'ABprp']
cluster_P1_list = ['EMS', 'P2', 'MS', 'E', 'C', 'P3', 'MSa', 'MSp', 'Ea', 'Ep', 'Ca', 'Cp', 'D', 'P4', 'Z']


def analysis_SHcPCA_maximum_clustering(embryo_path, l_degree):
    # read the SHcPCA matrix to get SHcPCA coefficients
    embryo_name = os.path.split(embryo_path)[-1]

    # ---------------read SHcPCA coefficient-------------------
    embryo_time_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                                    embryo_name + '_embryo_SHcPCA_result.csv')
    if not os.path.exists(embryo_time_matrices_saving_path):
        print('error detected! no SHcPCA matrix csv file can be found')
        return
    df_SHcPCA_coeffs = pd.read_csv(embryo_time_matrices_saving_path)
    df_index_tmp = df_SHcPCA_coeffs.values[:, :1]
    df_SHcPCA_coeffs.drop(columns=df_SHcPCA_coeffs.columns[0], inplace=True)
    df_SHcPCA_coeffs.index = list(df_index_tmp.flatten())
    print('finish read ', embryo_name, '--SHcPCA coefficient df!--------------')
    # -----------------------------------------------------------------------
    maximum_SHcPCA = np.max(df_SHcPCA_coeffs.values)
    minimum_SHcPCA = np.min(df_SHcPCA_coeffs.values)

    df_max_abs_clustering = pd.DataFrame(index=list(df_index_tmp.flatten()), columns=['cluster_num', 'abs_max_num'])
    for i_index in df_max_abs_clustering.index:
        abs_array = np.abs(df_SHcPCA_coeffs.loc[i_index])
        # maximum_abs_SHcPCA = np.max(abs_array)
        max_index_indices = np.where(abs_array == np.max(abs_array))[0][0]
        df_max_abs_clustering.at[i_index, 'cluster_num'] = max_index_indices

        df_max_abs_clustering.at[i_index, 'abs_max_num'] = df_SHcPCA_coeffs.loc[i_index][max_index_indices]

    print(df_max_abs_clustering)
    path_max_to_save = os.path.join(config.dir_my_data_SH_PCA_csv,
                                    embryo_name + '_SHcPCA_MAX_CLUSTER.csv')
    df_max_abs_clustering.to_csv(path_max_to_save)

def analysis_SHPCA_One_embryo(embryo_path, l_degree, is_do_PCA=True, is_show_PCA=True):
    """

    :param embryo_path: the 3D image data for embryo, 1-end time points embryos in this folder
    :param l_degree: degree ---- if l_degree is 10, there will be 0-10 degrees, will be 11**2 = 121 coefficients
    :return:
    """
    embryo_name = os.path.split(embryo_path)[-1]

    # --------------------------------get normalized coefficient -----------------------------------
    path_volume_surface_path = os.path.join(config.dir_my_data_volume_surface, embryo_name + '.csv')
    if not os.path.exists(path_volume_surface_path):
        cell_f.count_volume_surface_normalization_tocsv(embryo_path)

    df_embryo_volume_surface_slices = pd.read_csv(path_volume_surface_path)
    df_index_tmp = df_embryo_volume_surface_slices.values[:, :1]
    df_embryo_volume_surface_slices.drop(columns=df_embryo_volume_surface_slices.columns[0], inplace=True)
    # we can get normalization coefficient from df_embryo_volume_surface_slices.at[index_name,'normalized_c']
    df_embryo_volume_surface_slices.index = list(df_index_tmp.flatten())
    # ------------------------------------------------------------

    # --------------------------get sh coefficient df and do normalization---------------------------------
    path_saving_csv = os.path.join(config.dir_my_data_SH_time_domain_csv, embryo_name + '_l_' + str(l_degree) + '.csv')
    if not os.path.exists(path_saving_csv):
        compute_embryo_sh_descriptor_csv(embryo_path, l_degree, path_saving_csv)

    # ======================it's time to do normalization!=====================================
    path_saving_csv_normalized = os.path.join(config.dir_my_data_SH_time_domain_csv,
                                              embryo_name + '_l_' + str(l_degree) + '_norm.csv')
    if not os.path.exists(path_saving_csv_normalized):

        df_embryo_time_slices = pd.read_csv(path_saving_csv)
        df_index_tmp = df_embryo_time_slices.values[:, :1]
        df_embryo_time_slices.drop(columns=df_embryo_time_slices.columns[0], inplace=True)
        df_embryo_time_slices.index = list(df_index_tmp.flatten())

        for index_tmp in df_embryo_volume_surface_slices.index:
            this_normalized_coefficient = df_embryo_volume_surface_slices.loc[index_tmp][2]
            normalization_tmp = df_embryo_time_slices.loc[index_tmp] / this_normalized_coefficient
            # print(normalization_tmp.shape)
            # print(df_embryo_time_slices.loc[index_tmp])
            # print(normalization_tmp)
            df_embryo_time_slices.loc[index_tmp] = normalization_tmp
        df_embryo_time_slices.to_csv(path_saving_csv_normalized)
    # after build it, we can read it directly
    df_embryo_time_slices = pd.read_csv(path_saving_csv_normalized)
    df_index_tmp = df_embryo_time_slices.values[:, :1]
    df_embryo_time_slices.drop(columns=df_embryo_time_slices.columns[0], inplace=True)
    df_embryo_time_slices.index = list(df_index_tmp.flatten())
    print('finish read ', embryo_name, 'df_sh_norm_coefficients--------------')
    # ----------------------------------------------------------------------------------------

    if is_do_PCA:
        # ------------------------------directly pca-------------------------------------------------

        sh_PCA = PCA(n_components=12)

        sh_PCA.fit(df_embryo_time_slices.values)
        sh_PCA_mean = sh_PCA.mean_.flatten()

        print('PCA COMPONENTS: ', sh_PCA.n_components_)
        print('PCA EXPLAINED VARIANCE: ', sh_PCA.explained_variance_ratio_)

        PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                                embryo_name + '_time_single_PCA_result.csv')
        if not os.path.exists(PCA_matrices_saving_path):
            df_PCA_matrices = pd.DataFrame(data=sh_PCA.components_, columns=get_flatten_ldegree_morder(l_degree))
            df_PCA_matrices.insert(loc=0, column='explained_variation', value=list(sh_PCA.explained_variance_ratio_))
            df_PCA_matrices.loc['mean'] = [0] + list(sh_PCA_mean)
            df_PCA_matrices.to_csv(PCA_matrices_saving_path)

        if is_show_PCA:
            component_index = 0
            for component in sh_PCA.components_:
                print('components  ', component[:20])
                # print("inverse log::",inverse_log_expand[:50])

                fig = plt.figure()

                shc_instance_3 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(sh_PCA_mean + -5 * component)))
                shc_instance_2 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(sh_PCA_mean + -3 * component)))
                shc_instance_1 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(sh_PCA_mean + -1 * component)))
                shc_instance_0 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(sh_PCA_mean + 0 * component)))
                shc_instance1 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(sh_PCA_mean + 1 * component)))
                shc_instance2 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(sh_PCA_mean + 3 * component)))
                shc_instance3 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(sh_PCA_mean + 5 * component)))

                sh_reconstruction = do_reconstruction_for_SH(30, shc_instance_3)
                axes_tmp = fig.add_subplot(2, 3, 1, projection='3d')
                draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-5),
                                      ax=axes_tmp)
                sh_reconstruction = do_reconstruction_for_SH(30, shc_instance_2)
                axes_tmp = fig.add_subplot(2, 3, 2, projection='3d')
                draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-3),
                                      ax=axes_tmp)
                sh_reconstruction = do_reconstruction_for_SH(30, shc_instance_1)
                axes_tmp = fig.add_subplot(2, 3, 3, projection='3d')
                draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-1),
                                      ax=axes_tmp)

                sh_reconstruction = do_reconstruction_for_SH(30, shc_instance1)
                axes_tmp = fig.add_subplot(2, 3, 4, projection='3d')
                draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(1), ax=axes_tmp)
                sh_reconstruction = do_reconstruction_for_SH(30, shc_instance2)
                axes_tmp = fig.add_subplot(2, 3, 5, projection='3d')
                draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(3), ax=axes_tmp)
                sh_reconstruction = do_reconstruction_for_SH(30, shc_instance3)
                axes_tmp = fig.add_subplot(2, 3, 6, projection='3d')
                draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(5), ax=axes_tmp)

                plt.show()

                component_index += 1

        print('finish PCA, begin to work on PCA shape SPACE MODELING')
        # inverse_component_matrix = np.linalg.inv(sh_PCA.components_)
        Q, R = la.qr(sh_PCA.components_.T)

        embryo_time_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                                        embryo_name + '_embryo_SHcPCA_result.csv')
        if not os.path.exists(embryo_time_matrices_saving_path):
            df_sh_pca_coefs = pd.DataFrame(index=df_embryo_time_slices.index, columns=np.arange(0, 12))
            for index_tmp in df_embryo_time_slices.index:
                x_pca_shape_space = spla.solve_triangular(R, Q.T.dot(df_embryo_time_slices.loc[index_tmp]), lower=False)
                # print(x_pca_shape_space)
                df_sh_pca_coefs.loc[index_tmp] = list(x_pca_shape_space)
            df_sh_pca_coefs.to_csv(embryo_time_matrices_saving_path)
            print(df_sh_pca_coefs)

        # ----------------------------end:directly pca----------------------------------------------

        # # ----------------------------PCA modified sqrt then multiple 10--------------------------------
        #
        # sqrt_expand_array = general_f.sqrt_expand(df_embryo_time_slices.values)
        # print('finish sqrt expand for embryo values')
        # sh_PCA = PCA(n_components=12)
        #
        # sh_PCA.fit(sqrt_expand_array)
        # sh_PCA_mean = sh_PCA.mean_.flatten()
        #
        # print(sh_PCA.n_components_)
        # print(sh_PCA.explained_variance_ratio_)
        #
        #
        # component_index = 0
        # for component in sh_PCA.components_:
        #     print('components  ', component[:20])
        #     # print("inverse log::",inverse_log_expand[:50])
        #
        #     fig = plt.figure()
        #
        #     shc_instance_3 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(general_f.to_the_power_expand(
        #         (sh_PCA_mean + -30 * component).reshape((sh_PCA_mean.shape[0], 1))).flatten())))
        #     shc_instance_2 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(general_f.to_the_power_expand(
        #         (sh_PCA_mean + -20 * component).reshape((sh_PCA_mean.shape[0], 1))).flatten())))
        #
        #     shc_instance_1 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(general_f.to_the_power_expand(
        #         (sh_PCA_mean + -10 * component).reshape((sh_PCA_mean.shape[0], 1))).flatten())))
        #
        #     shc_instance_0 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(general_f.to_the_power_expand(
        #         sh_PCA_mean.reshape((sh_PCA_mean.shape[0], 1))).flatten())))
        #
        #     shc_instance1 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(general_f.to_the_power_expand(
        #         (sh_PCA_mean + 10 * component).reshape((sh_PCA_mean.shape[0], 1))).flatten())))
        #     shc_instance2 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(general_f.to_the_power_expand(
        #         (sh_PCA_mean + 20 * component).reshape((sh_PCA_mean.shape[0], 1))).flatten())))
        #     shc_instance3 = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(general_f.to_the_power_expand(
        #         (sh_PCA_mean + 30 * component).reshape((sh_PCA_mean.shape[0], 1))).flatten())))
        #
        #     sh_reconstruction = do_reconstruction_for_SH(30, shc_instance_3)
        #     axes_tmp = fig.add_subplot(2, 3, 1, projection='3d')
        #     draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-5), ax=axes_tmp)
        #     sh_reconstruction = do_reconstruction_for_SH(30, shc_instance_2)
        #     axes_tmp = fig.add_subplot(2, 3, 2, projection='3d')
        #     draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-3), ax=axes_tmp)
        #     sh_reconstruction = do_reconstruction_for_SH(30, shc_instance_1)
        #     axes_tmp = fig.add_subplot(2, 3, 3, projection='3d')
        #     draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-1), ax=axes_tmp)
        #
        #     sh_reconstruction = do_reconstruction_for_SH(30, shc_instance1)
        #     axes_tmp = fig.add_subplot(2, 3, 4, projection='3d')
        #     draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(1), ax=axes_tmp)
        #     sh_reconstruction = do_reconstruction_for_SH(30, shc_instance2)
        #     axes_tmp = fig.add_subplot(2, 3, 5, projection='3d')
        #     draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(3), ax=axes_tmp)
        #     sh_reconstruction = do_reconstruction_for_SH(30, shc_instance3)
        #     axes_tmp = fig.add_subplot(2, 3, 6, projection='3d')
        #     draw_f.draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(5), ax=axes_tmp)
        #
        #     plt.show()
        #     component_index += 1

        # # # ----------------------------end :::::: PCA modified sqrt then multiple 10--------------------------------

    return df_embryo_time_slices


def analysis_time_domain_k_means(embryo_path, l_degree):
    """

    :param embryo_path: the 3D image data for embryo, 1-end time points embryos in this folder
    :param l_degree: degree ---- if l_degree is 10, there will be 0-10 degrees, will be 11**2 = 121 coefficients
    :return:
    """
    embryo_name = os.path.split(embryo_path)[-1]

    # --------------------------get sh coefficient df ---------------------------------
    path_saving_csv = os.path.join(config.dir_my_data_SH_time_domain_csv, embryo_name + '_l_' + str(l_degree) + '.csv')
    if not os.path.exists(path_saving_csv):
        compute_embryo_sh_descriptor_csv(embryo_path, l_degree, path_saving_csv)

    df_embryo_time_slices = pd.read_csv(path_saving_csv)
    cell_index_time_this_embryo = df_embryo_time_slices.values[:, :1]
    array_time_this_embryo = df_embryo_time_slices.values[:, 1:]

    # ----------------------------------------------------------------------------------------

    cluster_cell_list = cluster_AB_list + cluster_P1_list
    # print(cluster_cell_list)

    cluster_num = 12

    # --------------------------------get volume and surface -----------------------------------
    path_volume_surface_path = os.path.join(config.dir_my_data_volume_surface, embryo_name + '.csv')
    if not os.path.exists(path_volume_surface_path):
        cell_f.count_volume_surface_normalization_tocsv(embryo_path)
    df_embryo_volume_surface_slices = pd.read_csv(path_volume_surface_path)

    df_index_tmp = df_embryo_volume_surface_slices.values[:, :1]
    df_embryo_volume_surface_slices.drop(columns=df_embryo_volume_surface_slices.columns[0], inplace=True)
    df_embryo_volume_surface_slices.index = list(df_index_tmp.flatten())
    # ------------------------------------------------------------

    # ------------------------   original k-means clustering ----------------------------------------
    estimator1 = KMeans(n_clusters=cluster_num, max_iter=1000)
    estimator1.fit(array_time_this_embryo)
    result_1 = estimator1.predict(array_time_this_embryo)

    path_to_saving = os.path.join(config.dir_my_data_SH_time_domain_csv,
                                  embryo_name + '_l_' + str(l_degree) + 'KMEANS_cluster12.csv')
    deal_with_cluster(result_1, cell_index_time_this_embryo, cluster_cell_list, cluster_num).to_csv(path_to_saving)

    result_1_sort_dict = {}
    for index, value in enumerate(result_1):
        if value in result_1_sort_dict.keys():
            result_1_sort_dict[value].append(index)
        else:
            result_1_sort_dict[value] = [index]

    center_sampling_dict = {}

    figure_rows = 2
    figure_columns = 2
    for index, value in enumerate(estimator1.cluster_centers_):
        log_abs_data = np.log(np.abs(np.float64(value)))
        offset = np.abs(int(np.min(log_abs_data)))
        center_sample_modified = general_f.log_expand_offset(value.reshape((value.shape[0], 1)), offset).flatten()

        shc_instance = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(center_sample_modified)))
        center_sampling_dict[index] = do_reconstruction_for_SH(30, shc_instance)

        fig = plt.figure()
        axes_tmp = fig.add_subplot(figure_rows, figure_columns, 1, projection='3d')
        draw_f.draw_3D_points(center_sampling_dict[index], ax=axes_tmp, fig_name='class' + str(index) + 'center_one')

        random_selection = random.sample(result_1_sort_dict[index], figure_rows * figure_columns - 1)
        for i in np.arange(1, figure_rows * figure_columns):
            this_index = random_selection[i - 1]
            shc_instance = pysh.SHCoeffs.from_array(
                collapse_flatten_clim(list(array_time_this_embryo[this_index])))
            sh_reconstruction = do_reconstruction_for_SH(20, shc_instance)
            axes_tmp = fig.add_subplot(figure_rows, figure_columns, i + 1, projection='3d')
            tp_cell_name_index = cell_index_time_this_embryo[this_index][0]
            # print(tp_cell_name_index)
            draw_f.draw_3D_points(sh_reconstruction, ax=axes_tmp, fig_name=tp_cell_name_index + 'V' + str(
                df_embryo_volume_surface_slices.loc[tp_cell_name_index][0]) + 'S' + str(
                df_embryo_volume_surface_slices.loc[tp_cell_name_index][1]))

    plt.show()
    # -----end:   -------------------   original k-means clustering ----------------------------------------

    # # ---------------------------- abs k-means clustering--------------------------------
    #
    # estimator2 = KMeans(n_clusters=cluster_num)
    # log_abs_data = np.log(np.abs(np.float64(data_frame_time_this_embryo)))
    # estimator2.fit(log_abs_data)
    # result_2 = estimator2.predict(log_abs_data)
    #
    # path_to_saving = os.path.join(config.dir_my_data_SH_time_domain_csv,
    #                               embryo_name + '_l_' + str(l_degree) + 'KMEANS_log_abs8.csv')
    # deal_with_cluster(result_2, cell_index_time_this_embryo, cluster_cell_list, cluster_num).to_csv(path_to_saving)
    #
    # result_2_sort_dict = {}
    # for index, value in enumerate(result_2):
    #     if value in result_2_sort_dict.keys():
    #         result_2_sort_dict[value].append(index)
    #     else:
    #         result_2_sort_dict[value] = [index]
    #
    # center_sampling_dict = {}
    # for index, value in enumerate(estimator2.cluster_centers_):
    #     fig = plt.figure()
    #     shc_instance = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(value)))
    #     center_sampling_dict[index] = do_reconstruction_for_SH(20, shc_instance, 3)
    #
    #     axes_tmp = fig.add_subplot(3, 3, 1, projection='3d')
    #     draw_f.draw_3D_points(center_sampling_dict[index], ax=axes_tmp, fig_name='class' + str(index) + 'center_one')
    #
    #     random_selection = random.sample(result_2_sort_dict[index], 8)
    #     for i in np.arange(1, 9):
    #         this_index = random_selection[i - 1]
    #         shc_instance = pysh.SHCoeffs.from_array(
    #             collapse_flatten_clim(list(data_frame_time_this_embryo[this_index])))
    #         sh_reconstruction = do_reconstruction_for_SH(20, shc_instance, 3)
    #         axes_tmp = fig.add_subplot(3, 3, i + 1, projection='3d')
    #         draw_f.draw_3D_points(sh_reconstruction, ax=axes_tmp,
    #                               fig_name=cell_index_time_this_embryo[this_index])
    #
    # plt.show()
    # # ---------------------------- end::::abs k-means clustering--------------------------------

    # # ----------------------------modified log k-means clustering--------------------------------
    #
    # estimator3 = KMeans(n_clusters=cluster_num)
    # log_abs_data = np.log(np.abs(np.float64(data_frame_time_this_embryo)))
    # offset = abs(int(np.min(log_abs_data)))
    # log_expand_offset_array = general_f.log_expand_offset(data_frame_time_this_embryo, offset)
    # estimator3.fit(log_expand_offset_array)
    # result_3 = estimator3.predict(log_expand_offset_array)
    #
    # path_to_saving = os.path.join(config.dir_my_data_SH_time_domain_csv,
    #                               embryo_name + '_l_' + str(l_degree) + 'KMEANS_logabs_inverse25.csv')
    # deal_with_cluster(result_3, cell_index_time_this_embryo, cluster_cell_list, cluster_num).to_csv(path_to_saving)
    #
    # result_3_sort_dict = {}
    # for index, value in enumerate(result_3):
    #     if value in result_3_sort_dict.keys():
    #         result_3_sort_dict[value].append(index)
    #     else:
    #         result_3_sort_dict[value] = [index]
    #
    # center_sampling_dict = {}
    # for index, value in enumerate(estimator3.cluster_centers_):
    #     fig = plt.figure()
    #     inverse_log_expand = general_f.exp_expand_offset(value.reshape((value.shape[0], 1)), offset).flatten()
    #
    #     shc_instance = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(value)))  # modified
    #     # shc_instance = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(inverse_log_expand))) # original
    #
    #     axes_tmp = fig.add_subplot(2, 2, 1, projection='3d')
    #     center_sampling_dict[index] = do_reconstruction_for_SH(30, shc_instance, 3)
    #     draw_f.draw_3D_points(center_sampling_dict[index], ax=axes_tmp, fig_name='class' + str(index) + 'center_one')
    #     # grid = shc_instance.expand()
    #     # grid.plot3d(cmap='RdBu', cmap_reverse=True, cmap_limits=[-1.5, 1.5], title='class' + str(index) + 'center_one',
    #     #             scale=1, ax=axes_tmp)
    #
    #     random_selection = random.sample(result_3_sort_dict[index], 8)
    #     for i in np.arange(1, 4):
    #         this_index = random_selection[i - 1]
    #         shc_instance = pysh.SHCoeffs.from_array(
    #             collapse_flatten_clim(list(data_frame_time_this_embryo[this_index])))
    #         sh_reconstruction = do_reconstruction_for_SH(20, shc_instance, 3)
    #         axes_tmp = fig.add_subplot(2, 2, i + 1, projection='3d')
    #         draw_f.draw_3D_points(sh_reconstruction, ax=axes_tmp,
    #                               fig_name=cell_index_time_this_embryo[this_index])
    #
    # plt.show()
    # # ----------------------------end :::::::modified log k-means clustering--------------------------------


def compute_embryo_sh_descriptor_csv(embryo_path, l_degree, path_saving_csv):
    # find all  embryos in this time domain by temporal series
    temporal_embryo_list = os.listdir(embryo_path)

    _, cell_name_to_No_dict = cell_f.get_cell_name_affine_table()
    # count = 0
    # see the l_degree explanation, that's why I need to plus one in func:get_flatten_ldegree_morder
    data_embryo_time_slices = pd.DataFrame(columns=get_flatten_ldegree_morder(l_degree))
    # print(data_embryo_time_slices)
    for temporal_embryo in temporal_embryo_list:
        time_point = str.split(temporal_embryo, '_')[1]
        print(temporal_embryo)
        if os.path.isfile(os.path.join(embryo_path, temporal_embryo)):
            # build the SHC table for this embryo
            this_embryo_dir = os.path.join(embryo_path, 'SH_C_folder_OF' + temporal_embryo)
            cells_list = os.listdir(this_embryo_dir)
            for cell_name in cells_list:
                cell_SH_path = os.path.join(this_embryo_dir, cell_name)
                sh_coefficient_instance = pysh.SHCoeffs.from_file(cell_SH_path, lmax=l_degree)
                one_dimension_coefficient = flatten_clim(sh_coefficient_instance)
                # print(cell_SH_path, one_dimension_coefficient[0])
                data_embryo_time_slices.loc[time_point + '::' + cell_name] = list(one_dimension_coefficient)
                print("\r Loading  ", end='tp and cell name ' + time_point + '::' + cell_name)

                # dict_key = cell_name_to_No_dict[cell_name]
    data_embryo_time_slices.to_csv(path_saving_csv)
    # print(count)


def analysis_calculate_error_contrast(embryo_path, file_name, behavior='both'):
    """
    :param embryo_path: the 3D image data for embryo, 1-end time points embryos in this folder
    :param file_name: specular time point embryo
    :param behavior: 'draw_contraction':drawing contraction for each cell;
                    'calculate_error' : calculate error between sh and original
    :return:
    """
    _, cell_name_to_No_dict = cell_f.get_cell_name_affine_table()

    # --------------------------------------file path dealing --------------------------------------
    this_embryo_dir = os.path.join(embryo_path, 'SH_C_folder_OF' + file_name)

    # find all  cells in this embryos dir
    cells_list = os.listdir(this_embryo_dir)

    if os.path.exists(os.path.join(config.dir_my_data, 'membrane' + file_name)):
        img = general_f.load_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + file_name))
    else:
        img = cell_f.nii_get_cell_surface(general_f.load_nitf2_img(os.path.join(embryo_path, file_name)),
                                          file_name)  # calculate membrane and save automatically
    # ----------------------------------------------------------------------------------------------

    # -------------------------initialize variables-------------------------------------------
    dict_img_membrane_calculate = {}
    img_membrane_data = img.get_fdata().astype(np.int16)
    x_num, y_num, z_num = img_membrane_data.shape
    # ----------------------------------------------------------------------------------------------

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

    # -------------get each cell ----------------
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

    # -----------------------analysis :  drawing contraction for each cell --------------------
    if behavior == 'draw_contraction' or 'both':
        # there is actually one loop for this embryo this time each cells
        for cell_name in cells_list:
            cell_SH_path = os.path.join(this_embryo_dir, cell_name)
            sh_coefficient_instance = pysh.SHCoeffs.from_file(cell_SH_path, lmax=11)

            dict_key = cell_name_to_No_dict[cell_name]

            tmp_this_membrane = np.array(dict_img_membrane_calculate[dict_key])
            center_points = np.sum(dict_img_cell_calculate[dict_key], axis=0) / len(
                dict_img_cell_calculate[dict_key])
            if center_points is None:
                center_points = [0, 0, 0]
            points_membrane_local = tmp_this_membrane - center_points

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
            # grid.plot(cmap='RdBu', cmap_reverse=True, show=False,
            #           title=cell_name_affine_table[dict_key] + 'regular')
            #   ------------------------------------------------------------------

            # # -----------------3D represent 2D curvature -----------------------
            # plane_representation_lat = np.arange(-90, 90, 180 / grid.data.shape[0])
            # plane_representation_lon = np.arange(0, 360, 360 / grid.data.shape[1])
            # plane_LAT, plane_LON = np.meshgrid(plane_representation_lat, plane_representation_lon)
            # fig = plt.figure(figsize=(6, 6))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot_surface(plane_LAT, plane_LON, grid.data.T)

            # ------------------3D representation-------------------------------
            do_contraction_image(sh_coefficient_instance, 20, points_membrane_local)
            # -----------------------------------------------------------------
    # --------------end-------analysis : drawing contraction for each cell-----------------------

    # -----------------------------------analysis : error ----------------------------------------
    if behavior == 'calculate_error' or 'both':
        average_error_list = []
        l_degree_range = np.arange(5, 30, 1)
        for l_degree in l_degree_range:
            print("-------------==>>>> degree    ", l_degree)
            average_error_list_this_degree = []

            # there is actually one loop for this embryo this time each cells
            for cell_name in cells_list:
                cell_SH_path = os.path.join(this_embryo_dir, cell_name)
                sh_coefficient_instance = pysh.SHCoeffs.from_file(cell_SH_path, lmax=l_degree)

                dict_key = cell_name_to_No_dict[cell_name]

                tmp_this_membrane = np.array(dict_img_membrane_calculate[dict_key])
                center_points = np.sum(dict_img_cell_calculate[dict_key], axis=0) / len(
                    dict_img_cell_calculate[dict_key])
                if center_points is None:
                    center_points = [0, 0, 0]
                points_membrane_local = tmp_this_membrane - center_points

                reconstruction_xyz = do_reconstruction_for_SH(50, sh_coefficient_instance)
                grid_data_SH, _ = SH_represention.do_sampling_with_interval(50, reconstruction_xyz, 10)

                grid_data_original, _ = SH_represention.do_sampling_with_interval(50, points_membrane_local, 10)

                error_sum = np.sum(np.abs(grid_data_SH - grid_data_original))
                print(error_sum)
                average_error_list_this_degree.append(error_sum)

            sum_error_this_degree = np.sum(np.array(average_error_list_this_degree))
            print("=================>>>>>>>>>>>> degree ", l_degree, ' ------ error----', sum_error_this_degree)
            average_error_list.append(sum_error_this_degree)
        print(average_error_list)
        plt.plot(l_degree_range, average_error_list)
        plt.show()
    # -------------------------------end-------analysis : error----------------------------------------------------


def do_reconstruction_for_SH(sample_N, sh_coefficient_instance):
    """
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

    reconstruction_xyz = general_f.sph2descartes(np.array(reconstruction_matrix))
    reconstruction_xyz[:, 2] = -reconstruction_xyz[:, 2]
    return reconstruction_xyz


def do_contraction_image(sh_coefficient_instance, sh_show_N, points_membrane_local):
    plane_representation_lat = np.arange(-90, 90, 180 / sh_show_N)
    plane_representation_lon = np.arange(0, 360, 360 / (2 * sh_show_N))
    plane_LAT, plane_LON = np.meshgrid(plane_representation_lat, plane_representation_lon)

    plane_LAT_FLATTEN = plane_LAT.flatten(order='F')
    plane_LON_FLATTEN = plane_LON.flatten(order='F')
    # print(plane_LAT_FLATTEN)
    # print(plane_LON_FLATTEN)
    grid = sh_coefficient_instance.expand(lat=plane_LAT_FLATTEN, lon=plane_LON_FLATTEN)

    plane_LAT_FLATTEN = plane_LAT_FLATTEN / 180 * math.pi
    plane_LON_FLATTEN = plane_LON_FLATTEN / 180 * math.pi

    # grid = sh_coefficient_instance.expand()
    # print(grid.data.shape)

    reconstruction_matrix = []
    # lat_interval = math.pi / grid.data.shape[0]
    # lon_interval = 2 * math.pi / grid.data.shape[1]
    ratio_interval = math.pi / sh_show_N
    for i in range(grid.data.shape[0]):
        reconstruction_matrix.append([grid.data[i], plane_LAT_FLATTEN[i], plane_LON_FLATTEN[i]])

    reconstruction_xyz = general_f.sph2descartes(np.array(reconstruction_matrix))
    reconstruction_xyz[:, 2] = -reconstruction_xyz[:, 2]
    p = multiprocessing.Process(target=draw_f.draw_3D_points, args=(reconstruction_xyz, 'shc reconstruction',))
    p.start()
    draw_f.draw_3D_points(points_membrane_local)

    # fig, ax = sh_coefficient_instance.expand().plot3d()
    # plt.show()


def flatten_clim(sh_coefficient_instance):
    """
    # -----------------------------------
    # cilm coefficient:
    # [0,:,:]---->m >= 0
    # [1,:,:]---->m <0
    # -----------------------------------
    :param sh_coefficient_instance:
    :return:
    """
    flatten_array = []
    coefficient_degree = sh_coefficient_instance.coeffs.shape[1]
    # print(coefficient_degree)
    for l_degree in range(coefficient_degree):
        # m<0
        for m_order in np.arange(l_degree, 0, step=-1):
            flatten_array.append(sh_coefficient_instance.coeffs[1, l_degree, m_order])
            # print(1, l_degree, m_order)
        # m>=0
        for m_order in np.arange(0, l_degree + 1):
            flatten_array.append(sh_coefficient_instance.coeffs[0, l_degree, m_order])
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


def deal_with_cluster(cluster_result, cell_index, cluster_cell_list, cluster_num):
    clustering_result_df = pd.DataFrame(index=cluster_cell_list, columns=list(np.arange(0, cluster_num)))

    for match_pattern in cluster_cell_list:
        for index, cell_sample in enumerate(list(cluster_result)):
            cell_name = str.split(cell_index[index][0], '::')[-1]
            if re.match(match_pattern, cell_name):
                # -------------1---------------
                if match_pattern == 'Z' or 'P3' or 'C' or 'P4' or 'D':
                    if pd.isna(clustering_result_df.at['P2', cell_sample]):
                        clustering_result_df.at['P2', cell_sample] = 1
                    else:
                        clustering_result_df.at['P2', cell_sample] += 1
                    # ------------2--------------
                    if match_pattern == 'Z' or 'P4' or 'D':
                        if pd.isna(clustering_result_df.at['P3', cell_sample]):
                            clustering_result_df.at['P3', cell_sample] = 1
                        else:
                            clustering_result_df.at['P3', cell_sample] += 1
                        # -----------3-----------------
                        if match_pattern == 'Z':
                            if pd.isna(clustering_result_df.at['P4', cell_sample]):
                                clustering_result_df.at['P4', cell_sample] = 1
                            else:
                                clustering_result_df.at['P4', cell_sample] += 1
                # -------------1-----------------
                if match_pattern == 'MS':
                    if pd.isna(clustering_result_df.at['EMS', cell_sample]):
                        clustering_result_df.at['EMS', cell_sample] = 1
                    else:
                        clustering_result_df.at['EMS', cell_sample] += 1
                # ------------1-----------------
                if pd.isna(clustering_result_df.at[match_pattern, cell_sample]):
                    clustering_result_df.at[match_pattern, cell_sample] = 1
                else:
                    clustering_result_df.at[match_pattern, cell_sample] += 1

    clustering_result_df.loc['Cluster_sum'] = clustering_result_df.apply(lambda x: x.sum(), axis=0)

    clustering_result_df['Mother_daughter_sum'] = clustering_result_df.apply(lambda x: x.sum(), axis=1)
    Mother_daughter_sum = clustering_result_df['Mother_daughter_sum']

    clustering_result_df = clustering_result_df.div(clustering_result_df['Mother_daughter_sum'], axis=0)
    clustering_result_df.drop(columns='Mother_daughter_sum', axis=1, inplace=True)

    clustering_result_df = clustering_result_df.astype('float64')
    clustering_result_df = clustering_result_df.round(2)
    clustering_result_df['Mother_daughter_sum'] = Mother_daughter_sum

    return clustering_result_df
