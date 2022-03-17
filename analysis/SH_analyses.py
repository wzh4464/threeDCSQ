import random

import utils.general_func as general_f
import utils.cell_func as cell_f
import utils.draw_func as draw_f
import transformation.SH_represention as SH_represention

from matplotlib import pyplot as plt
import pyshtools as pysh
import numpy as np
import pandas as pd
import multiprocessing
from utils import config
import os
import math
import re

from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.spherical_func import normalize_SHc

cluster_AB_list = ['ABa', 'ABp', 'ABal', 'ABar', 'ABpl', 'ABpr', 'ABala', 'ABalp', 'ABara', 'ABarp', 'ABpla', 'ABplp',
                   'ABpra', 'ABprp']
cluster_P1_list = ['EMS', 'P2', 'MS', 'E', 'C', 'P3', 'MSa', 'MSp', 'Ea', 'Ep', 'Ca', 'Cp', 'D', 'P4', 'Z']


def analysis_SHcPCA_energy_ratio(embryo_path, used_degree=16):
    '''

    :param embryo_path:
    :param used_degree:
    :return: just ration of all, same as SHcPCA
    '''
    # read the SHcPCA matrix to get SHcPCA coefficients
    embryo_name = os.path.split(embryo_path)[-1]

    # ---------------read SHcPCA coefficient-------------------
    embryo_time_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                                    embryo_name + '_SHcPCA{}.csv'.format((used_degree + 1) ** 2))
    df_SHcPCA_coeffs = general_f.read_csv_to_df(embryo_time_matrices_saving_path)
    # -----------------------------------------------------------------------
    df_SHcPCA_abs_coeffs = df_SHcPCA_coeffs.abs()
    df_SHcPCA_abs_coeffs['all_energy'] = df_SHcPCA_abs_coeffs.apply(lambda x: x.sum(), axis=1)

    df_energy_distribution = pd.DataFrame(index=df_SHcPCA_coeffs.index, columns=df_SHcPCA_coeffs.columns)
    for df_tmp_index in df_energy_distribution.index:
        # print(df_SHcPCA_abs_coeffs.at[df_tmp_index, 'all_energy'])
        df_energy_distribution.loc[df_tmp_index] = df_SHcPCA_coeffs.loc[df_tmp_index] / \
                                                   df_SHcPCA_abs_coeffs.at[df_tmp_index, 'all_energy']

    # print(df_SHcPCA_coeffs.applymap(lambda x: x / df_SHcPCA_abs_coeffs['all_energy']))
    print(df_energy_distribution)
    path_energy_distribution_to_save = os.path.join(config.dir_my_data_SHcPCA_ratio,
                                                    embryo_name + '_SHcPCA{}_E_D.csv'.format(
                                                        (used_degree + 1) ** 2))
    df_energy_distribution.to_csv(path_energy_distribution_to_save)


def analysis_SHcPCA_KMEANS_clustering(embryo_path, used_degree=16, cluster_num=12):
    # read the SHcPCA matrix to get SHcPCA coefficients
    embryo_name = os.path.split(embryo_path)[-1]

    # ---------------read SHcPCA coefficient-------------------
    embryo_time_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                                    embryo_name + '_SHcPCA{}.csv'.format((used_degree + 1) ** 2))
    df_SHcPCA_coeffs = general_f.read_csv_to_df(embryo_time_matrices_saving_path)
    # -----------------------------------------------------------------------

    # cluster_num = 8
    estimator1 = KMeans(n_clusters=cluster_num, max_iter=10000)
    # estimator1.fit(df_SHcPCA_coeffs.values)
    result_1 = estimator1.fit_predict(df_SHcPCA_coeffs.values)

    df_kmeans_clustering = pd.DataFrame(index=df_SHcPCA_coeffs.index, columns=['cluster_num'])
    df_kmeans_clustering['cluster_num'] = result_1

    print(df_kmeans_clustering)
    path_max_to_save = os.path.join(config.dir_my_data_SH_clustering_csv,
                                    embryo_name + '_SHcPCA' + str(
                                        (used_degree + 1) ** 2) + '_KMEANS_CLUSTER{}.csv'.format(cluster_num))
    df_kmeans_clustering.to_csv(path_max_to_save)


def analysis_SHcPCA_maximum_clustering(embryo_path, used_degree=16):
    # read the SHcPCA matrix to get SHcPCA coefficients
    embryo_name = os.path.split(embryo_path)[-1]

    # ---------------read SHcPCA coefficient-------------------
    embryo_SHcPCA_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                             embryo_name + '_SHcPCA{}.csv'.format((used_degree + 1) ** 2))
    if not os.path.exists(embryo_SHcPCA_saving_path):
        print('error detected! no SHcPCA matrix csv file can be found')
        return
    df_SHcPCA_coeffs = general_f.read_csv_to_df(embryo_SHcPCA_saving_path)
    print('finish read ', embryo_name, '--SHcPCA coefficient df!--------------')
    # -----------------------------------------------------------------------
    # maximum_SHcPCA = np.max(df_SHcPCA_coeffs.values)
    # minimum_SHcPCA = np.min(df_SHcPCA_coeffs.values)

    df_max_abs_clustering = pd.DataFrame(index=df_SHcPCA_coeffs.index, columns=['cluster_num', 'abs_max_num'])
    for i_index in df_max_abs_clustering.index:
        abs_array = np.abs(df_SHcPCA_coeffs.loc[i_index])
        # maximum_abs_SHcPCA = np.max(abs_array)
        # find where is it!
        max_index_indices = np.where(abs_array == np.max(abs_array))[0][0]
        df_max_abs_clustering.at[i_index, 'cluster_num'] = max_index_indices

        df_max_abs_clustering.at[i_index, 'max_value'] = df_SHcPCA_coeffs.loc[i_index][max_index_indices]

    print(df_max_abs_clustering)
    path_max_to_save = os.path.join(config.dir_my_data_SH_clustering_csv,
                                    embryo_name + '_SHcPCA{}_MAX_CLUSTER.csv'.format((used_degree + 1) ** 2))
    df_max_abs_clustering.to_csv(path_max_to_save)


def analysis_SHcPCA_All_embryo(l_degree, is_show_PCA=True, PCA_num=24):
    # embryo_file_name = [f for f in os.listdir(embryo_sub_path) if os.path.isfile(os.path.join(embryo_sub_path, f))]

    SHc_dict = {}
    for embryo_index in np.arange(4, 21):
        embryo_name = 'Sample' + f'{embryo_index:02}' + 'LabelUnified'
        print(embryo_name)

        path_saving_csv_normalized = os.path.join(config.dir_my_data_SH_time_domain_csv,
                                                  embryo_name + '_l_' + str(l_degree) + '_norm.csv')
        df_SHc_norm = general_f.read_csv_to_df(path_saving_csv_normalized)
        SHc_dict[embryo_name] = df_SHc_norm.values
        print('finish read ', embryo_name, 'df_sh_norm_coefficients--------------')


def analysis_SHcPCA_One_embryo(embryo_path, used_degree, l_degree=25, is_do_PCA=True, is_show_PCA=True):
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

        df_SHc_norm = pd.read_csv(path_saving_csv)
        df_index_tmp = df_SHc_norm.values[:, :1]
        df_SHc_norm.drop(columns=df_SHc_norm.columns[0], inplace=True)
        df_SHc_norm.index = list(df_index_tmp.flatten())

        for index_tmp in df_embryo_volume_surface_slices.index:
            this_normalized_coefficient = df_embryo_volume_surface_slices.loc[index_tmp][2]
            normalization_tmp = df_SHc_norm.loc[index_tmp] / this_normalized_coefficient

            df_SHc_norm.loc[index_tmp] = normalization_tmp
        df_SHc_norm.to_csv(path_saving_csv_normalized)
    # after build it, we can read it directly
    df_SHc_norm = general_f.read_csv_to_df(path_saving_csv_normalized)
    print('finish read ', embryo_name, 'df_sh_norm_coefficients--------------')
    # ----------------------------------------------------------------------------------------

    if is_do_PCA:
        # ------------------------------directly pca-------------------------------------------------

        sh_PCA = PCA()

        sh_PCA.fit(df_SHc_norm.values[:, :(used_degree + 1) ** 2])
        # print(sh_PCA.mean_.shape)
        # print(sh_PCA.mean_.flatten().shape)
        #
        sh_PCA_mean = sh_PCA.mean_

        print('PCA COMPONENTS: ', sh_PCA.n_components_)
        print('PCA EXPLAINED VARIANCE: ', sh_PCA.explained_variance_ratio_)

        PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                                embryo_name + '_PCA{}.csv'.format(sh_PCA.n_components_))
        # if not os.path.exists(PCA_matrices_saving_path):
        df_PCA_matrices = pd.DataFrame(data=sh_PCA.components_, columns=get_flatten_ldegree_morder(used_degree))
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

        print('finish PCA, begin to work on SHcPCA shape SPACE MODELING')
        print('LETS compute the SHcPCA vector , now begin.')

        # Q, R = la.qr(sh_PCA.components_)
        inverse_component_matrix = np.linalg.inv(sh_PCA.components_.T)

        embryo_time_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                                        embryo_name + '_SHcPCA{}.csv'.format(sh_PCA.n_components_))
        # if not os.path.exists(embryo_time_matrices_saving_path):
        df_sh_pca_coefs = pd.DataFrame(index=df_SHc_norm.index, columns=list(np.arange(0, sh_PCA.n_components_)))
        for index_tmp in df_SHc_norm.index:
            # x_pca_shape_space = spla.solve_triangular(R, Q.T.dot(df_SHc_norm.loc[index_tmp] - sh_PCA_mean),
            #                                           lower=False)
            # print(x_pca_shape_space)
            df_sh_pca_coefs.loc[index_tmp] = inverse_component_matrix.dot(
                df_SHc_norm.loc[index_tmp][:sh_PCA.n_components_] - sh_PCA_mean)
        df_sh_pca_coefs.to_csv(embryo_time_matrices_saving_path)

        # ----------------------------end:directly pca----------------------------------------------

        # # ----------------------------PCA modified sqrt then multiple 10--------------------------------
        #
        # sqrt_expand_array = general_f.sqrt_expand(df_SHc_norm.values)
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

    return df_SHc_norm


def analysis_SHc_Kmeans_One_embryo(embryo_path, used_degree, l_degree=25, cluster_num=12, is_normalization=True,
                                   is_conclude_in_percentage=False,
                                   is_show_cluster=False):
    """

    :param embryo_path: the 3D image data for embryo, 1-end time points embryos in this folder
    :param l_degree: degree ---- if l_degree is 10, there will be 0-10 degrees, will be 11**2 = 121 coefficients
    :return:
    """
    embryo_name = os.path.split(embryo_path)[-1]

    # --------------------------------get volume and surface -----------------------------------
    path_volume_surface_path = os.path.join(config.dir_my_data_volume_surface, embryo_name + '.csv')
    if not os.path.exists(path_volume_surface_path):
        cell_f.count_volume_surface_normalization_tocsv(embryo_path)

    df_embryo_volume_surface_slices = general_f.read_csv_to_df(path_volume_surface_path)
    # ------------------------------------------------------------

    # -------------------------------read normalized or original choosing------------------------------
    path_original_SHc_saving_csv = os.path.join(config.dir_my_data_SH_time_domain_csv,
                                                embryo_name + '_l_' + str(l_degree) + '.csv')
    path_normalized_SHc_saving_csv = os.path.join(config.dir_my_data_SH_time_domain_csv,
                                                  embryo_name + '_l_' + str(l_degree) + '_norm.csv')
    if not os.path.exists(path_original_SHc_saving_csv):
        compute_embryo_sh_descriptor_csv(embryo_path, l_degree, path_original_SHc_saving_csv)
    if is_normalization:
        # --------------------------get normalized sh coefficient df ---------------------------------
        if not os.path.exists(path_normalized_SHc_saving_csv):
            normalize_SHc(path_normalized_SHc_saving_csv, df_embryo_volume_surface_slices,
                          path_normalized_SHc_saving_csv)
        # after build it, we can read it directly
        df_embryo_SHc = general_f.read_csv_to_df(path_normalized_SHc_saving_csv)
        print('finish read ', embryo_name, 'df_sh_norm_coefficients--------------')
    else:
        # --------------------------get sh coefficient df ---------------------------------
        df_embryo_SHc = general_f.read_csv_to_df(path_original_SHc_saving_csv)
        print('finish read ', embryo_name, 'df_sh_norm_coefficients--------------')

        # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------

    # ==========================================  original k-means clustering =======================================
    # estimator1 = KMeans(n_clusters=cluster_num, max_iter=10000)
    estimator1 = KMeans(n_clusters=cluster_num)

    estimator1.fit(df_embryo_SHc.values[:, :(used_degree + 1) ** 2])
    Kmeans_cluster_label = estimator1.predict(df_embryo_SHc.values[:, :(used_degree + 1) ** 2])

    # ------------------------- save norm k-means clustering result to csv --------------------
    df_SHc_KMEANS_clustering = pd.DataFrame(index=df_embryo_SHc.index, columns=['cluster_num'])
    # maximum_abs_SHcPCA = np.max(abs_array)
    df_SHc_KMEANS_clustering['cluster_num'] = Kmeans_cluster_label

    path_kmeans_to_save = os.path.join(config.dir_my_data_SH_clustering_csv,
                                       embryo_name + '_SHc' + str(
                                           (used_degree + 1) ** 2) + '_KMEANS_CLUSTER{}.csv'.format(cluster_num))
    df_SHc_KMEANS_clustering.to_csv(path_kmeans_to_save)
    print(df_SHc_KMEANS_clustering)
    # ---------------------------------------------------------------------------------------

    # ============================================================================================================

    # ------------------------  is saving the percentage conclusion to csv ?----------------------------
    if is_conclude_in_percentage:
        cluster_cell_list = cluster_AB_list + cluster_P1_list
        path_to_saving = os.path.join(config.dir_my_data_SH_time_domain_csv,
                                      embryo_name + '_l_' + str(l_degree) + 'KMEANS_cluster12.csv')
        conclude_cluster_in_percentage(Kmeans_cluster_label, df_embryo_SHc.values, cluster_cell_list,
                                       cluster_num).to_csv(path_to_saving)
    # ---------------------------------------------------------------------------------------------

    # ----------------------  draw cluster centroid and randomly selective cells' figures ---------
    if is_show_cluster == True:
        result_1_sort_dict = {}
        for index, value in enumerate(Kmeans_cluster_label):
            if value in result_1_sort_dict.keys():
                result_1_sort_dict[value].append(index)
            else:
                result_1_sort_dict[value] = [index]

        center_sampling_dict = {}

        figure_rows = 3
        figure_columns = 3
        for index, value in enumerate(estimator1.cluster_centers_):
            log_abs_data = np.log(np.abs(np.float64(value)))
            offset = np.abs(int(np.min(log_abs_data)))
            center_sample_modified = general_f.log_expand_offset(value.reshape((value.shape[0], 1)), offset).flatten()

            shc_instance = pysh.SHCoeffs.from_array(collapse_flatten_clim(list(center_sample_modified)))
            center_sampling_dict[index] = do_reconstruction_for_SH(30, shc_instance)

            fig = plt.figure()
            axes_tmp = fig.add_subplot(figure_rows, figure_columns, 1, projection='3d')
            draw_f.draw_3D_points(center_sampling_dict[index], ax=axes_tmp,
                                  fig_name='class' + str(index) + 'center_one')

            random_selection = random.sample(result_1_sort_dict[index], figure_rows * figure_columns - 1)
            for i in np.arange(1, figure_rows * figure_columns):
                this_index = random_selection[i - 1]
                shc_instance = pysh.SHCoeffs.from_array(
                    collapse_flatten_clim(list(df_embryo_SHc.values[this_index])))
                sh_reconstruction = do_reconstruction_for_SH(20, shc_instance)
                axes_tmp = fig.add_subplot(figure_rows, figure_columns, i + 1, projection='3d')
                tp_cell_name_index = df_embryo_SHc.index[this_index]
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


def analysis_compare_represent_method(embryo_path):
    embryo_time_list = []
    for file_name in os.listdir(embryo_path):
        if os.path.isfile(os.path.join(embryo_path, file_name)):
            # print(path_tmp)
            embryo_time_list.append(file_name)


def analysis_compare_SHc(embryo_path, file_name, behavior='both'):
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
    # ----------------------------------------------------

    # -------------get each cell all point,just for calculate the centroid---------------
    if os.path.exists(os.path.join(embryo_path, file_name)):
        img = general_f.load_nitf2_img(os.path.join(embryo_path, file_name))
    else:
        return EOFError  # calculate cell and save automatically
    dict_img_cell_calculate = {}
    img_cell_data = img.get_fdata().astype(np.int16)
    # just for calculate the centroid
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
    if behavior == 'draw_contraction' or behavior == 'both':
        # there is actually one loop for this embryo this time each cells
        for cell_name in cells_list:
            cell_SH_path = os.path.join(this_embryo_dir, cell_name)
            sh_coefficient_instance = pysh.SHCoeffs.from_file(cell_SH_path, lmax=16)

            dict_key = cell_name_to_No_dict[cell_name]
            # get centroid
            tmp_this_membrane = np.array(dict_img_membrane_calculate[dict_key])
            center_points = np.sum(dict_img_cell_calculate[dict_key], axis=0) / len(
                dict_img_cell_calculate[dict_key])
            if center_points is None:
                center_points = [0, 0, 0]
            points_membrane_local = tmp_this_membrane - center_points
            N_sh_num = int(math.sqrt(len(points_membrane_local) / 2))

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
            print(cell_name)
            do_contraction_image(sh_coefficient_instance, N_sh_num, points_membrane_local)
            # -----------------------------------------------------------------
    # --------------end-------analysis : drawing contraction for each cell-----------------------

    # -----------------------------------analysis : error ----------------------------------------
    if behavior == 'calculate_error' or behavior == 'both':
        average_error_list = []
        l_degree_range = np.arange(5, 30, 1)
        for l_degree in l_degree_range:
            print("-------------==>>>> degree    ", l_degree)
            average_error_list_this_degree = []

            # there is actually one loop for this embryo this time each cells
            for cell_name in tqdm(cells_list, desc="dealing with {}".format(l_degree)):
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

    # -----------------------------------analysis : error ----------------------------------------

    if behavior == 'compare_method_error':

        average_error_list = []
        l_degree_range = np.arange(5, 25, 1)

        for l_degree in l_degree_range:
            print("dealing with degree  ", l_degree, (l_degree + 1) ** 2 * 2, (l_degree + 1) ** 2, (l_degree + 1) ** 2,
                  "  coefficients for sampling, SPHARM,SPCSM")
            # random latitude and longitude
            map_random = [[random.uniform(0, math.pi), random.uniform(0, 2 * math.pi)] for i in range(15 ** 2 * 2)]

            # there is actually one loop for this embryo this time each cells
            for cell_name in tqdm(cells_list, desc="dealing with {}".format(l_degree)):

                # ------------------------calculate shc error------------------------------------
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



def generate_3D_matrix_from_SHc(sh_instance, dense=100, multiplier=2):
    '''
    not just surface but inside
    :param sh_instance:
    :param dense: x y z dense (number of points)
    :return:
    '''
    matrix_3d = np.zeros((dense, dense, dense))
    # matrix_tmp = np.zeros((dense, dense, dense))

    error_test_point_num = 10
    map_tmp = [[random.uniform(0, math.pi), random.uniform(0, 2 * math.pi)] for i in
               range(error_test_point_num)]
    R_SHc, shc_sample_xyz = get_points_with_SHc(sh_instance,
                                                colat=np.array(map_tmp)[:, 0],
                                                lon=np.array(map_tmp)[:, 1],
                                                is_return_xyz=True)
    max_r_index = np.argmax(R_SHc)
    bound_cube = max(np.abs(shc_sample_xyz[max_r_index])) * multiplier
    interval_coordinate = bound_cube / (dense / 2)

    for i in range(dense):
        for j in range(dense):
            for k in range(dense):
                x_i, y_j, z_k = -bound_cube + i * interval_coordinate, -bound_cube + j * interval_coordinate, -bound_cube + k * interval_coordinate
                r, colat, lon = general_f.descartes2spherical2([[x_i, y_j, z_k]])[0]
                r_shc, _ = get_points_with_SHc(sh_instance, colat=np.array([colat]), lon=np.array([lon]))
                if r < r_shc:
                    matrix_3d[i][j][k] = 1
    return matrix_3d


def get_points_with_SHc(sh_instance, colat, lon, is_return_xyz=False):
    '''
    co latitude
    :param cilm: the sh instance. coeffs
    :param colat: the co-latitude points
    :param lon: the lon points
    :return: the R of above latitude
    '''

    sph_list = []

    lat = -(colat - math.pi / 2)
    f_lm_array = sh_instance.expand(lat=lat, lon=lon, degrees=False)
    for idx, _ in enumerate(lat):
        sph_list.append([f_lm_array[idx], lat[idx], lon[idx]])

    reconstruction_xyz = general_f.sph2descartes(np.array(sph_list))
    reconstruction_xyz[:, 2] = -reconstruction_xyz[:, 2]

    # return the co-latitude!
    sph_list = general_f.descartes2spherical2(reconstruction_xyz)
    # print(sph_list[:,1]-colat)
    # sph_list=np.array(sph_list)
    if is_return_xyz is True:
        return sph_list[:, 0], np.array(reconstruction_xyz)

    return sph_list[:, 0], sph_list


def do_contraction_image(sh_coefficient_instance, sh_show_N, points_membrane_local):
    reconstruction_xyz = do_reconstruction_for_SH(sh_show_N, sh_coefficient_instance)
    p = multiprocessing.Process(target=draw_f.draw_3D_points, args=(reconstruction_xyz, 'shc reconstruction',))
    p.start()
    draw_f.draw_3D_points(points_membrane_local)



def conclude_cluster_in_percentage(cluster_result, cell_index, cluster_cell_list, cluster_num):
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

# def do_SHc_
# if __name__ == "__main__":
