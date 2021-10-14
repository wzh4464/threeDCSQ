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

import seaborn as sns
from scipy import spatial

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
Sample06,ABalaapa,078
"""


def show_cell_SPCSMs_info():
    # Sample06,ABalaapa,078
    # Sample06,Dpaap,158
    print('waiting type you input')
    embryo_name, cell_name, tp = str(input()).split(',')
    # cell_name = str(input())
    # tp = str(input())

    print(embryo_name, cell_name, tp)

    embryo_path_csv = os.path.join(r'D:\cell_shape_quantification\DATA\my_data_csv\SH_time_domain_csv',
                                   embryo_name + 'LabelUnified_l_25_norm.csv')
    embryo_csv = general_f.read_csv_to_df(embryo_path_csv)

    # fig_points = plt.figure()
    fig_SPCSMs_info = plt.figure()
    plt.axis('off')

    embryo_path_name = embryo_name + 'LabelUnified'
    embryo_path = os.path.join('D:/cell_shape_quantification/DATA/SegmentCellUnified04-20', embryo_path_name)
    file_name = embryo_name + '_' + tp + '_segCell.nii.gz'
    dict_cell_membrane, dict_center_points = sh_represent.get_nib_embryo_membrane_dict(embryo_path,
                                                                                       file_name)
    _, cell_num = cell_f.get_cell_name_affine_table()
    keys_tmp = cell_num[cell_name]
    local_surface_points = dict_cell_membrane[keys_tmp] - dict_center_points[keys_tmp]
    axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 1, projection='3d')
    draw_f.draw_3D_points(local_surface_points, ax=axes_tmp, fig_name='original ' + cell_name + '::' + tp)

    instance_tmp = pysh.SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(embryo_csv.loc[tp + '::' + cell_name]))
    axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 2, projection='3d')

    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=50, sh_coefficient_instance=instance_tmp),
                             ax=axes_tmp, fig_name=cell_name + '::' + tp, cmap='viridis')

    sn = 20
    x_axis = draw_f.Arrow3D([0, sn + 3], [0, 0],
                            [0, 0], mutation_scale=20,
                            lw=3, arrowstyle="-|>", color="r")
    axes_tmp.add_artist(x_axis)
    y_axis = draw_f.Arrow3D([0, 0], [0, sn + 3],
                            [0, 0], mutation_scale=20,
                            lw=3, arrowstyle="-|>", color="r")
    axes_tmp.add_artist(y_axis)
    z_axis = draw_f.Arrow3D([0, 0], [0, 0],
                            [0, sn + 10], mutation_scale=20,
                            lw=3, arrowstyle="-|>", color="r")
    axes_tmp.add_artist(z_axis)

    axis_points_num = 1000
    lineage_ = np.arange(0, sn, sn / axis_points_num)
    zeros_ = np.zeros(axis_points_num)
    # xz=np.zeros(axis_points_num)
    axes_tmp.scatter3D(lineage_, zeros_, zeros_, s=10, color='r')
    axes_tmp.scatter3D(zeros_, lineage_, zeros_, s=10, color='r')
    sn = sn + 5
    lineage_ = np.arange(0, sn, sn / axis_points_num)
    zeros_ = np.zeros(axis_points_num)
    axes_tmp.scatter3D(zeros_, zeros_, lineage_, s=10, color='r')
    lineage_ = np.arange(-sn, 0, sn / axis_points_num)
    zeros_ = np.zeros(axis_points_num)
    axes_tmp.scatter3D(zeros_, zeros_, lineage_, s=10, color='r')

    axes_tmp.text(sn / 2, 0, -.2 * sn, 'xlabel', 'x', ha='center')
    axes_tmp.text(0, sn / 2, -.2 * sn, 'ylabel', 'y', ha='center')
    axes_tmp.text(-.1 * sn, 0, 0, 'zlabel', 'z', ha='center')

    # Sample06,ABalaapa,078
    # Sample06,P1,001
    axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 3)
    grid_tmp = instance_tmp.expand(lmax=50)
    # axin=inset_axes(axes_tmp, width="50%", height="100%", loc=2)
    grid_tmp.plot(ax=axes_tmp, cmap='RdBu', cmap_reverse=True, title='Heat Map',
                  xlabel='Longitude (from positive half x-axis)',
                  ylabel='Latitude (from horizontal x-y plane)')

    axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 4)
    instance_tmp.plot_spectrum(ax=axes_tmp, fname=cell_name + '::' + tp + ' spectrum curve')
    axes_tmp.set_title(cell_name + '::' + tp + ' spectrum curve')

    axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 5)
    instance_tmp.plot_spectrum2d(ax=axes_tmp, fname=cell_name + '::' + tp + ' 2D spectra')
    axes_tmp.set_title(cell_name + '::' + tp + ' 2D spectra')

    print(instance_tmp.spectrum())
    # axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 4)
    #
    # axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 5)
    #
    # axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 6)
    # axes_tmp.set_title('volume: '+str(instance_tmp.volume()) + '  ' + 'centroid: '+str(instance_tmp.centroid()))
    axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 6, projection='3d')
    grid_tmp.plot3d(cmap='RdBu', cmap_reverse=True,
                    ax=axes_tmp)



    plt.show()


def calculate_spectrum():
    print('waiting input ============>')
    # Sample06,ABalaapa,078,ABalaapa,079
    embryo_name, cell_name1, tp1, cell_name2, tp2 = str(input()).split(',')
    embryo_path = os.path.join(r'D:\cell_shape_quantification\DATA\my_data_csv\SH_time_domain_csv',
                               embryo_name + 'LabelUnified_l_25_norm.csv')
    embryo_csv = general_f.read_csv_to_df(embryo_path)

    # # fig_points = plt.figure()
    # fig_SPCSMs_info = plt.figure()
    # plt.axis('off')

    instance_tmp1 = pysh.SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(embryo_csv.loc[tp1 + '::' + cell_name1]))
    instance_tmp2 = pysh.SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(embryo_csv.loc[tp2 + '::' + cell_name2]))

    # log_spectrum1 = np.log(instance_tmp1.spectrum())
    # log_spectrum2 = np.log(instance_tmp2.spectrum())
    # print('eclidean log',spatial.distance.euclidean(log_spectrum1, log_spectrum2))
    # print('cosine log',spatial.distance.cosine(log_spectrum1, log_spectrum2))

    print('euclidean', spatial.distance.euclidean(instance_tmp1.spectrum(), instance_tmp2.spectrum()))
    # print(instance_tmp1.spectrum(), instance_tmp2.spectrum())

    print('cosine', spatial.distance.cosine(instance_tmp1.spectrum(), instance_tmp2.spectrum()))

    # print('mahalanobis',spatial.distance.mahalanobis(instance_tmp1.spectrum(), instance_tmp2.spectrum()))

    print('correlation', spatial.distance.correlation(instance_tmp1.spectrum(), instance_tmp2.spectrum()))


# transfer 2d spectrum to spectrum (1-d curve adding together)
def transfer_2d_to_spectrum():
    path_original = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc.csv')
    path_norm = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc.csv')

    saving_original_csv= pd.DataFrame(columns=np.arange(start=0,stop=26,step=1))
    df_csv=general_f.read_csv_to_df(path_original)
    for row_idx in df_csv.index:
        saving_original_csv.loc[row_idx]=pysh.SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(df_csv.loc[row_idx])).spectrum()
        # print(num_idx)
        # print(saving_original_csv)
    print(saving_original_csv)
    saving_original_csv.to_csv(os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv','SHc_Spectrum.csv'))

    saving_norm_csv=pd.DataFrame(columns=np.arange(start=0,stop=26,step=1))
    df_csv = general_f.read_csv_to_df(path_original)
    for row_idx in df_csv.index:
        saving_norm_csv.loc[row_idx] = pysh.SHCoeffs.from_array(
            sh_analysis.collapse_flatten_clim(df_csv.loc[row_idx])).spectrum()
        # print(num_idx)
        # print(saving_original_csv)
    saving_original_csv.to_csv(
        os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc_norm_Spectrum.csv'))
    print(saving_norm_csv)



if __name__ == "__main__":
    transfer_2d_to_spectrum()
