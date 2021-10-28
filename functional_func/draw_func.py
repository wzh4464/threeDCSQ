from matplotlib import pyplot as plt
import numpy as np
import os
import config
import random
import pandas as pd
import pyshtools as pysh

import particular_func.SH_analyses as SH_A_f
import functional_func.general_func as general_f
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def generate_2D_Z_ARRAY(x, y, z):
    x_num = len(x)
    y_num = len(y)
    Z = np.zeros((x_num, y_num), dtype=np.float64)
    for i in range(x_num):
        for j in range(y_num):
            if i == j:
                Z[i][j] = z[i]
            else:
                Z[i][j] = np.float64(0)
    return Z


def draw_3D_curve_with_lines(points_data, fig_name="DEFAULT", fig_size=(6, 6)):
    fig = plt.figure(figsize=fig_size)
    ax = Axes3D(fig)

    # x = points_data[:, 0]  # first column of the 2D matrix
    # y = points_data[:, 1]
    # z = points_data[:, 2]

    num_points_half = int(len(points_data) / 2)
    num_points_quarter = int(len(points_data) / 4)

    points_data = points_data[points_data[:, 0].argsort()]
    # print(points_data.shape)
    # print(points_data)
    # print(points_data[0:num_points_half, :])

    points_data_xnegative = points_data[:num_points_half, :]
    points_data_xnegative = points_data_xnegative[points_data_xnegative[:, 1].argsort()]
    points_data_xnegative_ynegative = points_data_xnegative[:num_points_quarter]
    points_data_xnegative_ynegative = points_data_xnegative_ynegative[points_data_xnegative_ynegative[:, 2].argsort()]
    points_data_xnegative_ypositive = points_data_xnegative[num_points_quarter:]
    points_data_xnegative_ypositive = points_data_xnegative_ypositive[points_data_xnegative_ypositive[:, 2].argsort()]

    points_data_xpositive = points_data[num_points_half:]
    points_data_xpositive = points_data_xpositive[points_data_xpositive[:, 1].argsort()]
    points_data_xpositive_ynegative = points_data_xpositive[:num_points_quarter]
    # points_data_xpositive_ynegative = points_data_xpositive_ynegative[points_data_xpositive_ynegative[:, 2].argsort()]
    points_data_xpositive_ypositive = points_data_xpositive[num_points_quarter:]
    points_data_xpositive_ypositive = points_data_xpositive_ypositive[points_data_xpositive_ypositive[:, 2].argsort()]

    x = points_data_xnegative_ynegative[:, 0]  # first column of the 2D matrix
    y = points_data_xnegative_ynegative[:, 1]
    z = points_data_xnegative_ynegative[:, 2]
    ax.plot3D(x, y, z, 'blue')

    x = points_data_xnegative_ypositive[:, 0]  # first column of the 2D matrix
    y = points_data_xnegative_ypositive[:, 1]
    z = points_data_xnegative_ypositive[:, 2]
    ax.plot3D(x, y, z, 'grey')

    x = points_data_xpositive_ynegative[:, 0]  # first column of the 2D matrix
    y = points_data_xpositive_ynegative[:, 1]
    z = points_data_xpositive_ynegative[:, 2]
    ax.scatter3D(x, y, z, c=z, cmap="Greens")

    x = points_data_xpositive_ypositive[:, 0]  # first column of the 2D matrix
    y = points_data_xpositive_ypositive[:, 1]
    z = points_data_xpositive_ypositive[:, 2]
    ax.plot3D(x, y, z, 'red')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title(fig_name)
    plt.show()


def draw_3D_curve_with_triangle(points_data, fig_name="DEFAULT", fig_size=(10, 10)):
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(fig_name)

    ax = Axes3D(fig)

    x = points_data[:, 0]  # first column of the 2D matrix
    y = points_data[:, 1]
    z = points_data[:, 2]
    ax.plot_trisurf(x, y, z)

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_title(fig_name)
    # plt.title(fig_name)
    plt.show()


def draw_3D_points_in_new_coordinate(points, center=None):
    if center is None:
        center = [0, 0, 0]
    points_new = points - center
    draw_3D_points(points_new)
    return points_new


def draw_3D_points(points_data, fig_name="", fig_size=(10, 10), ax=None, cmap='viridis'):
    x = points_data[:, 0]  # first column of the 2D matrix
    y = points_data[:, 1]
    z = points_data[:, 2]

    if ax == None:
        fig = plt.figure(figsize=fig_size)
        ax = Axes3D(fig)
        # ax.scatter3D(x, y, z, marker='o', c=z, cmap=cmap)
        ax.scatter3D(x, y, z, c=z, cmap=cmap)

        ax.set_zlabel('Z').set_fontstyle('italic')  # 坐标轴
        ax.set_ylabel('Y').set_fontstyle('italic')
        ax.set_xlabel('X').set_fontstyle('italic')
        ax.set_title(fig_name)

        plt.title(fig_name)
        plt.show()
    else:
        # ax.scatter3D(x, y, z, marker='o', c=z, cmap=cmap)
        ax.scatter3D(x, y, z, c=z, cmap=cmap)

        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        ax.set_title(fig_name)


def draw_comparison_SHcPCA_SH(embryo_path, l_degree=25, cell_name='NONE', used_degree=9, used_PCA_num=12):
    """

    :param embryo_path: the path of the specific cell's embryo or the embryo you want to see
    :param cell_name: maybe you want to see specific cell, if not, let it be 'NONE'
    :return:
    """
    embryo_name = os.path.split(embryo_path)[-1]
    PCA_NUM = (used_degree + 1) ** 2

    # ======================it's time to do normalization!=====================================
    path_saving_csv_normalized = os.path.join(config.dir_my_data_SH_time_domain_csv,
                                              embryo_name + '_l_' + str(l_degree) + '_norm.csv')
    if not os.path.exists(path_saving_csv_normalized):
        print("==EEERRRROOOOR==========no embryo normalized sh coefficient df!!!!!!!!!!!!!==================")
        return
    # after build it, we can read it directly
    df_embryo_time_slices = general_f.read_csv_to_df(path_saving_csv_normalized)
    #
    # df_embryo_time_slices = pd.read_csv(path_saving_csv_normalized)
    # df_index_tmp = df_embryo_time_slices.values[:, :1]
    # df_embryo_time_slices.drop(columns=df_embryo_time_slices.columns[0], inplace=True)
    # df_embryo_time_slices.index = list(df_index_tmp.flatten())

    print('finish read ', embryo_name, 'df_sh_norm_coefficients--------------')
    # ========================================================================================

    # ====================RECONSTRUCTION FROM SHcPCA ==================================
    # -------------------read PCA coefficient----------------------
    PCA_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                            embryo_name + '_PCA{}.csv'.format(PCA_NUM))
    if not os.path.exists(PCA_matrices_saving_path):
        print("==EEERRRROOOOR==========no SH PCA coefficient df!!!!!!!!!!!!!==================")
        return

    df_PCA_matrix = general_f.read_csv_to_df(PCA_matrices_saving_path)
    # df_PCA_matrix = pd.read_csv(PCA_matrices_saving_path)
    # df_index_tmp = df_PCA_matrix.values[:, :1]
    # df_PCA_matrix.drop(columns=df_PCA_matrix.columns[0], inplace=True)
    # df_PCA_matrix.index = list(df_index_tmp.flatten())

    mean_PCA = df_PCA_matrix.loc['mean'][1:]  # depend on the first column is explained variance
    df_PCA_matrix.drop(index='mean', inplace=True)

    explained_variance = df_PCA_matrix['explained_variation']
    df_PCA_matrix.drop(columns='explained_variation', inplace=True)

    print(df_PCA_matrix)

    print('finish read ', embryo_name, 'df_PCA MATRIX!--------------')
    # -----------------------------------------------------------------------

    # ---------------read SHcPCA coefficient-------------------
    embryo_time_matrices_saving_path = os.path.join(config.dir_my_data_SH_PCA_csv,
                                                    embryo_name + '_SHcPCA{}.csv'.format(PCA_NUM))
    if not os.path.exists(embryo_time_matrices_saving_path):
        print("==EEERRRROOOOR==========no SHcPCA coefficient df!!!!!!!!!!!!!==================")
        return

    df_SHcPCA_coeffs = general_f.read_csv_to_df(embryo_time_matrices_saving_path)
    # df_SHcPCA_coeffs = pd.read_csv(embryo_time_matrices_saving_path)
    # df_index_tmp = df_SHcPCA_coeffs.values[:, :1]
    # df_SHcPCA_coeffs.drop(columns=df_SHcPCA_coeffs.columns[0], inplace=True)
    # df_SHcPCA_coeffs.index = list(df_index_tmp.flatten())
    print('finish read ', embryo_name, '--SHcPCA coefficient df!--------------')
    # -----------------------------------------------------------------------
    # ================================================================================

    # compare SHc AND SHcPCA image reconstruction
    if cell_name == 'NONE':
        for index_tmp in df_embryo_time_slices.index:
            fig = plt.figure()

            shc_instance = pysh.SHCoeffs.from_array(
                SH_A_f.collapse_flatten_clim(list(df_embryo_time_slices.loc[index_tmp])))
            shc_reconstruction = SH_A_f.do_reconstruction_for_SH(30, shc_instance)
            axes_tmp = fig.add_subplot(1, 2, 1, projection='3d')
            draw_3D_points(shc_reconstruction, fig_name='original sh coefficient', ax=axes_tmp)

            shcPCA_shc_list = list(mean_PCA + np.dot(df_PCA_matrix.values[:12, :].T, df_SHcPCA_coeffs.loc[index_tmp]))
            shcPCA_instance = pysh.SHCoeffs.from_array(SH_A_f.collapse_flatten_clim(shcPCA_shc_list))
            shcPCA_reconstruction = SH_A_f.do_reconstruction_for_SH(30, shcPCA_instance)
            axes_tmp = fig.add_subplot(1, 2, 2, projection='3d')
            draw_3D_points(shcPCA_reconstruction, fig_name=' sh coefficient PCA', ax=axes_tmp)
            plt.show()
    elif cell_name == 'RANDOM':
        while 1:
            index_tmp = random.choice(df_embryo_time_slices.index)

            fig = plt.figure()
            # SHc representation
            shc_instance = pysh.SHCoeffs.from_array(
                SH_A_f.collapse_flatten_clim(list(df_embryo_time_slices.loc[index_tmp][:PCA_NUM])))
            shc_reconstruction = SH_A_f.do_reconstruction_for_SH(30, shc_instance)
            axes_tmp = fig.add_subplot(1, 2, 1, projection='3d')
            draw_3D_points(shc_reconstruction, fig_name=index_tmp + '  original SHc', ax=axes_tmp)

            # SHcPCA representation
            shcPCA_shc_list = list(
                mean_PCA + np.dot(df_PCA_matrix.values[:used_PCA_num, :].T,
                                  df_SHcPCA_coeffs.loc[index_tmp][:used_PCA_num]))
            shcPCA_instance = pysh.SHCoeffs.from_array(SH_A_f.collapse_flatten_clim(shcPCA_shc_list))
            shcPCA_reconstruction = SH_A_f.do_reconstruction_for_SH(30, shcPCA_instance)

            print('SHc--->', list(df_embryo_time_slices.loc[index_tmp][:20]))
            print('SHcPCA--->', shcPCA_shc_list[:20])

            axes_tmp = fig.add_subplot(1, 2, 2, projection='3d')
            draw_3D_points(shcPCA_reconstruction, fig_name=index_tmp + '  SHcPCA coefficient', ax=axes_tmp)
            plt.show()





class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)