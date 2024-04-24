#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import dependency library
import os

from sklearn.decomposition import PCA

from static import config

import numpy as np
import pandas as pd
import numpy.linalg as la
from tqdm import tqdm
from matplotlib import pyplot as plt
from pyshtools import SHCoeffs

# import user defined library

import analysis.SH_analyses as sh_analysis
from utils.draw_func import draw_3D_points
import utils.general_func as general_f




def draw_PCA(sh_PCA_path):

    sh_PCA_mean,variance,df_pca=read_PCA_file(sh_PCA_path)
    # sh_PCA_mean = sh_PCA.mean_
    # idx = 0
    for idx in df_pca.index:
        print('components  ', idx)
        # print("inverse log::",inverse_log_expand[:50])

        fig = plt.figure()
        component=df_pca.loc[idx]
        shc_instance_3 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + -5 * component)))
        shc_instance_2 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + -3 * component)))
        shc_instance_1 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + -1 * component)))
        shc_instance_0 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + 0 * component)))
        shc_instance1 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + 1 * component)))
        shc_instance2 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + 3 * component)))
        shc_instance3 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + 5 * component)))

        sh_reconstruction = sh_analysis.do_reconstruction_from_SH(100, shc_instance_3)
        axes_tmp = fig.add_subplot(2, 3, 1, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name='Component '+str(idx) + ' Constituent Weight ' + str(-5),
                       ax=axes_tmp)
        sh_reconstruction = sh_analysis.do_reconstruction_from_SH(100, shc_instance_2)
        axes_tmp = fig.add_subplot(2, 3, 2, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name='Component '+str(idx) + ' Constituent Weight ' + str(-3),
                       ax=axes_tmp)
        sh_reconstruction = sh_analysis.do_reconstruction_from_SH(100, shc_instance_1)
        axes_tmp = fig.add_subplot(2, 3, 3, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name='Component '+str(idx) + ' Constituent Weight ' + str(-1),
                       ax=axes_tmp)

        sh_reconstruction = sh_analysis.do_reconstruction_from_SH(100, shc_instance1)
        axes_tmp = fig.add_subplot(2, 3, 4, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name='Component '+str(idx) + ' Constituent Weight ' + str(1), ax=axes_tmp)
        sh_reconstruction = sh_analysis.do_reconstruction_from_SH(100, shc_instance2)
        axes_tmp = fig.add_subplot(2, 3, 5, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name='Component '+str(idx) + ' Constituent Weight ' + str(3), ax=axes_tmp)
        sh_reconstruction = sh_analysis.do_reconstruction_from_SH(100, shc_instance3)
        axes_tmp = fig.add_subplot(2, 3, 6, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name='Component '+str(idx) + ' Constituent Weight ' + str(5), ax=axes_tmp)

        plt.show()

        # component_index += 1

def save_PCA_file(PCA_file_path,PCA_instance,feature_columns):
    df_PCA_matrices = pd.DataFrame(data=PCA_instance.components_, columns=feature_columns)
    df_PCA_matrices.insert(loc=0, column='explained_variation_ratio', value=list(PCA_instance.explained_variance_ratio_))
    df_PCA_matrices.loc['mean'] = [0] + list(PCA_instance.mean_)
    df_PCA_matrices.to_csv(PCA_file_path)

def read_PCA_file(PCA_file_path):
    PCA_df = general_f.read_csv_to_df(PCA_file_path)
    pca_means = PCA_df.loc['mean'][1:]
    PCA_df.drop(index='mean', inplace=True)
    pca_explained = PCA_df['explained_variation_ratio']
    PCA_df.drop(columns='explained_variation_ratio', inplace=True)

    PCA_instance=PCA(n_components=len(PCA_df.index))
    PCA_instance.mean_=np.array(pca_means).flatten()
    PCA_instance.explained_variance_ratio_=np.array(pca_explained).flatten()
    PCA_instance.components_=PCA_df.values

    return PCA_instance


def calculate_PCA_zk_norm(embryo_path, PCA_matrices_saving_path, k=12):
    """
    # this method is totally wrong OH my god!!!!!!!!!!!! 2021-10-30 by zelin
        # I really don't know why don't you just go to run the tutorial !!!!!
         # this function is implemented by sklearn , just in the PCA class.....
    :param embryo_path:
    :param PCA_matrices_saving_path:
    :param k:
    :return:
    """
    print('PCA norm exist')
    pca_means, _, pca_df = read_PCA_file(PCA_matrices_saving_path)
    # print(means)
    # print(variation)
    # print(n_components)

    embryo_name = os.path.basename(embryo_path)
    path_SHc = os.path.join(config.dir_my_data_SH_time_domain_csv, embryo_name + '_l_25_norm.csv')
    df_SHc = general_f.read_csv_to_df(path_SHc)
    df_SHcPCA = pd.DataFrame(columns=range(k))
    # SHc_u_matrix = df_SHc.values.T - pca_means
    # pca_df.values is k x d , we need inverse of d x k
    # P x zk = x - u
    # inverse_P = np.linalg.inv(pca_df.values.T)
    # Z = inverse_P.dot(df_SHc.values.T)
    # https: // zh.wikipedia.org / wiki / QR % E5 % 88 % 86 % E8 % A7 % A3
    # d > k 本质是最小化{\displaystyle ||A{\hat {x}}-b||}{\displaystyle ||A{\hat {x}}-b||}
    Q, R = la.qr(pca_df.values[:k].T)
    R_ = np.linalg.inv(R)
    # print(R.shape)
    # print(Q.shape)

    for y_idx in tqdm(df_SHc.index, desc='dealing with each cell'):
        y = df_SHc.loc[y_idx]
        # print(pca_means)
        y_u = y - pca_means
        zk = R_.dot(Q.T.dot(y_u))
        # print(zk)
        # print(zk.shape)
        df_SHcPCA.loc[y_idx] = zk

    path_tmp = os.path.join(config.dir_my_data_SH_PCA_csv, embryo_name + '_SHcPCA{}_norm.csv'.format(k))
    df_SHcPCA.to_csv(path_tmp)


# Zk = Z[:k][:]
# print(Zk)
# print(Zk.shape)
# 测一下准不准 画两个出来看看


def calculate_PCA_zk(embryo_path, PCA_matrices_saving_path, k=12):
    print('PCA exist, calculating zk')
    pca_means, _, pca_df = read_PCA_file(PCA_matrices_saving_path)
    # print(means)
    # print(variation)
    # print(n_components)

    embryo_name = os.path.basename(embryo_path)
    path_SHc = os.path.join(config.dir_my_data_SH_time_domain_csv, embryo_name + '_l_25.csv')
    df_SHc = general_f.read_csv_to_df(path_SHc)
    df_SHcPCA = pd.DataFrame(columns=range(k))
    # SHc_u_matrix = df_SHc.values.T - pca_means
    # pca_df.values is k x d , we need inverse of d x k
    # P x zk = x - u
    # inverse_P = np.linalg.inv(pca_df.values.T)
    # Z = inverse_P.dot(df_SHc.values.T)
    # https: // zh.wikipedia.org / wiki / QR % E5 % 88 % 86 % E8 % A7 % A3
    # d > k 本质是最小化{\displaystyle ||A{\hat {x}}-b||}{\displaystyle ||A{\hat {x}}-b||}
    Q, R = la.qr(pca_df.values[:k].T)
    R_ = np.linalg.inv(R)
    # print(R.shape)
    # print(Q.shape)

    for y_idx in tqdm(df_SHc.index, desc='dealing with each cell'):
        y = df_SHc.loc[y_idx]
        # print(pca_means)
        y_u = y - pca_means
        zk = R_.dot(Q.T.dot(y_u))
        # print(zk)
        # print(zk.shape)
        df_SHcPCA.loc[y_idx] = zk

    path_tmp = os.path.join(config.dir_my_data_SH_PCA_csv, embryo_name + '_SHcPCA{}.csv'.format(k))
    df_SHcPCA.to_csv(path_tmp)
