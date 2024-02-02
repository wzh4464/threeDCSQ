# import dependency library
import pickle
from pickle import load

import open3d as o3d
import json
from random import uniform

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyshtools import SHGrid
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import LinearSVC

from treelib import Tree

import umap
import umap.plot

import numpy as np
import os
import pandas as pd

from sklearn.cluster import KMeans

from time import time

import pyshtools as pysh
import seaborn as sns

from tqdm import tqdm

from scipy import spatial

from matplotlib import cm
import matplotlib.patches

from datetime import datetime

import igraph as iGraph

# import user defined library

from graph_wavelet.enhanced_GWf import embryo_enhanced_graph_wavelet, get_kth_neighborhood_graph, phi_j_h_wavelet
from lineage_stat.data_structure import get_combined_lineage_tree
from static import dict as dict
from static.dict import cell_fate_map
from transformation.SH_represention import get_nib_embryo_membrane_dict, do_sampling_with_interval
from utils.cell_func import get_cell_name_affine_table, nii_get_cell_surface
from utils.draw_func import draw_3D_points, Arrow3D, set_size
from utils.general_func import read_csv_to_df, load_nitf2_img
from utils.sh_cooperation import collapse_flatten_clim, do_reconstruction_for_SH
from utils.data_io import check_folder

import static.config as config

"""
Sample05,ABalaapa,078
"""


def show_cell_SPCSMs_info():
    # Sample06,ABalaapa,078
    # Sample06,Dpaap,158
    print('waiting type you input')
    embryo_name, cell_name, tp = str(input()).split(',')
    # cell_name = str(input())
    # tp = str(input())

    print(embryo_name, cell_name, tp)

    embryo_path_csv = os.path.join(config.cell_shape_analysis_data_path + r'my_data_csv\SH_time_domain_csv',
                                   embryo_name + 'LabelUnified_l_25_norm.csv')
    embryo_csv = read_csv_to_df(embryo_path_csv)

    # fig_points = plt.figure()
    fig_SPCSMs_info = plt.figure()
    plt.axis('off')

    embryo_path_name = embryo_name + 'LabelUnified'
    embryo_path = os.path.join(r'.\DATA\SegmentCellUnified04-20', embryo_path_name)
    file_name = embryo_name + '_' + tp + '_segCell.nii.gz'
    dict_cell_membrane, dict_center_points = get_nib_embryo_membrane_dict(embryo_path,
                                                                          file_name)
    _, cell_num = get_cell_name_affine_table()
    keys_tmp = cell_num[cell_name]

    local_surface_points = dict_cell_membrane[keys_tmp] - dict_center_points[keys_tmp]
    axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 1, projection='3d')
    draw_3D_points(local_surface_points, ax=axes_tmp, fig_name='original ' + cell_name + '::' + tp)

    instance_tmp = pysh.SHCoeffs.from_array(collapse_flatten_clim(embryo_csv.loc[tp + '::' + cell_name]))
    axes_tmp = fig_SPCSMs_info.add_subplot(2, 3, 2, projection='3d')

    draw_3D_points(do_reconstruction_for_SH(sample_N=50, sh_coefficient_instance=instance_tmp),
                   ax=axes_tmp, fig_name=cell_name + '::' + tp, cmap='viridis')

    sn = 20
    x_axis = Arrow3D([0, sn + 3], [0, 0],
                     [0, 0], mutation_scale=20,
                     lw=3, arrowstyle="-|>", color="r")
    axes_tmp.add_artist(x_axis)
    y_axis = Arrow3D([0, 0], [0, sn + 3],
                     [0, 0], mutation_scale=20,
                     lw=3, arrowstyle="-|>", color="r")
    axes_tmp.add_artist(y_axis)
    z_axis = Arrow3D([0, 0], [0, 0],
                     [0, sn + 23], mutation_scale=20,
                     lw=3, arrowstyle="-|>", color="r")
    axes_tmp.add_artist(z_axis)

    axis_points_num = 1000
    lineage_ = np.arange(0, sn, sn / axis_points_num)
    zeros_ = np.zeros(axis_points_num)
    # xz=np.zeros(axis_points_num)
    axes_tmp.scatter3D(lineage_, zeros_, zeros_, s=10, color='r')
    axes_tmp.scatter3D(zeros_, lineage_, zeros_, s=10, color='r')
    sn = sn + 15
    lineage_ = np.arange(0, sn, sn / axis_points_num)
    zeros_ = np.zeros(axis_points_num)
    axes_tmp.scatter3D(zeros_, zeros_, lineage_, s=10, color='r')
    axes_tmp.axis('off')

    # lineage_ = np.arange(-sn, 0, sn / axis_points_num)
    # zeros_ = np.zeros(axis_points_num)
    # axes_tmp.scatter3D(zeros_, zeros_, lineage_, s=10, color='r')

    axes_tmp.text(sn / 3 * 2, 0, -.2 * sn, 'x', 'x', ha='center')
    axes_tmp.text(0, sn / 3 * 2, -.2 * sn, 'y', 'y', ha='center')
    axes_tmp.text(-0.1 * sn, 0, sn / 3 * 2, 'z', 'z', ha='center')

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
    embryo_csv = read_csv_to_df(embryo_path)

    # # fig_points = plt.figure()
    # fig_SPCSMs_info = plt.figure()
    # plt.axis('off')

    instance_tmp1 = pysh.SHCoeffs.from_array(collapse_flatten_clim(embryo_csv.loc[tp1 + '::' + cell_name1]))
    instance_tmp2 = pysh.SHCoeffs.from_array(collapse_flatten_clim(embryo_csv.loc[tp2 + '::' + cell_name2]))

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
def transfer_2d_to_spectrum_01paer():
    embryo_names = [str(i).zfill(2) for i in range(4, 21)]
    for embryo_name in embryo_names:
        embryo_individual_path = os.path.join(config.cell_shape_analysis_data_path,
                                              'my_data_csv/SH_time_domain_csv/Sample{}LabelUnified_l_25_norm.csv'.format(
                                                  embryo_name))
        df_shc_norm_embryo = read_csv_to_df(embryo_individual_path)

        saving_norm_csv = pd.DataFrame(columns=np.arange(start=0, stop=26, step=1))
        for row_idx in df_shc_norm_embryo.index:
            saving_norm_csv.loc[row_idx] = pysh.SHCoeffs.from_array(
                collapse_flatten_clim(df_shc_norm_embryo.loc[row_idx])).spectrum()
            # print(num_idx)
            # print(saving_original_csv)
        print(saving_norm_csv)
        saving_norm_csv.to_csv(os.path.join(config.cell_shape_analysis_data_path,
                                            'my_data_csv/SH_time_domain_csv/Sample{}_Spectrum_norm.csv'.format(
                                                embryo_name)))

        embryo_individual_path = os.path.join(config.cell_shape_analysis_data_path,
                                              'my_data_csv/SH_time_domain_csv/Sample{}LabelUnified_l_25.csv'.format(
                                                  embryo_name))
        df_shc_embryo = read_csv_to_df(embryo_individual_path)
        saving_sh_csv = pd.DataFrame(columns=np.arange(start=0, stop=26, step=1))
        for row_idx in df_shc_embryo.index:
            saving_sh_csv.loc[row_idx] = pysh.SHCoeffs.from_array(
                collapse_flatten_clim(df_shc_embryo.loc[row_idx])).spectrum()
            # print(num_idx)
            # print(saving_original_csv)
        print(saving_sh_csv)
        saving_sh_csv.to_csv(
            os.path.join(config.cell_shape_analysis_data_path,
                         'my_data_csv/SH_time_domain_csv/Sample{}_Spectrum.csv'.format(embryo_name)))

    # path_original = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc.csv')
    # path_norm = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc_norm.csv')
    #
    # saving_original_csv = pd.DataFrame(columns=np.arange(start=0, stop=26, step=1))
    # df_csv = read_csv_to_df(path_original)
    # for row_idx in df_csv.index:
    #     saving_original_csv.loc[row_idx] = pysh.SHCoeffs.from_array(
    #         collapse_flatten_clim(df_csv.loc[row_idx])).spectrum()
    #     # print(num_idx)
    #     # print(saving_original_csv)
    # print(saving_original_csv)
    # saving_original_csv.to_csv(
    #     os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc_Spectrum.csv'))
    #
    # saving_norm_csv = pd.DataFrame(columns=np.arange(start=0, stop=26, step=1))
    # df_csv = read_csv_to_df(path_norm)
    # for row_idx in df_csv.index:
    #     saving_norm_csv.loc[row_idx] = pysh.SHCoeffs.from_array(
    #         collapse_flatten_clim(df_csv.loc[row_idx])).spectrum()
    #     # print(num_idx)
    #     # print(saving_original_csv)
    # saving_norm_csv.to_csv(
    #     os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc_norm_Spectrum.csv'))
    # print(saving_norm_csv)


def cluster_with_spectrum():
    path_original = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv',
                                 'SHc_norm_Spectrum.csv')
    concat_df_Spectrum = read_csv_to_df(path_original)

    # Neuron, Pharynx, Intestine, Skin, Muscle, Germcell, death, unspecifed
    cluster_num = 9
    estimator = KMeans(n_clusters=cluster_num, max_iter=10000)

    result_origin = estimator.fit_predict(np.power(concat_df_Spectrum.values, 1 / 2))
    print(estimator.cluster_centers_)
    df_kmeans_clustering = pd.DataFrame(index=concat_df_Spectrum.index, columns=['cluster_num'])
    df_kmeans_clustering['cluster_num'] = result_origin
    df_kmeans_clustering.to_csv(
        os.path.join(config.cell_shape_analysis_data_path,
                     'normsqrt_spectrum_cluster_k{}.csv'.format(cluster_num)))


def build_label_supervised_learning():
    cshaper_data_label_df = pd.DataFrame(columns=['Fate'])

    path_original = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv',
                                 'SHc_norm_Spectrum.csv')
    concat_df_Spectrum = read_csv_to_df(path_original)

    dfs = pd.read_excel(config.cell_fate_path, sheet_name=None)['CellFate']
    fate_dict = {}
    for idx in dfs.index:
        # print(row)
        name = dfs.loc[idx]['Name'].split('\'')[0]
        fate = dfs.loc[idx]['Fate'].split('\'')[0]
        fate_dict[name] = fate
    print(fate_dict)
    for idx in tqdm(concat_df_Spectrum.index, desc='Dealing with norm spectrum'):
        cell_name = idx.split('::')[2]
        tmp_cell_name = cell_name
        while tmp_cell_name not in fate_dict.keys():
            tmp_cell_name = tmp_cell_name[:-1]
        cshaper_data_label_df.loc[idx] = dict.cell_fate_map[fate_dict[tmp_cell_name]]
    print(cshaper_data_label_df)
    cshaper_data_label_df.to_csv(os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv',
                                              '17_embryo_fate_label.csv'))
    # print(dfs.loc[dfs['Name'] == 'ABalaaaalpa\'']['Fate'])


def SPCSMs_SVM_by_tp():
    print('reading dsf')
    t0 = time()
    cshaper_X = read_csv_to_df(
        os.path.join('.\DATA\my_data_csv\SH_time_domain_csv', 'SHc_norm.csv'))
    cshaper_Y = read_csv_to_df(
        os.path.join('.\DATA\my_data_csv\SH_time_domain_csv',
                     '17_embryo_fate_label.csv'))
    print('Unspecified', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Unspecified']].index))
    print('Other', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Other']].index))
    print('Death', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Death']].index))
    print('Neuron', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Neuron']].index))
    print('Intestin', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Intestin']].index))
    print('Muscle', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Muscle']].index))
    print('Pharynx', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Pharynx']].index))
    print('Skin', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Skin']].index))
    print('Germ Cell', len(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Germ Cell']].index))

    cshaper_X.drop(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Unspecified']].index, inplace=True)
    cshaper_Y.drop(cshaper_Y[cshaper_Y['Fate'] == dict.cell_fate_map['Unspecified']].index, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        cshaper_X.values[:30000], cshaper_Y.values[:30000].reshape((cshaper_Y.values[:30000].shape[0],)), test_size=0.2,
        random_state=datetime.now().microsecond)
    print("reading done in %0.3fs" % (time() - t0))

    print('train distribution', len(y_train))
    print('train Unspecified', (y_train == dict.cell_fate_map['Unspecified']).sum())
    print('train Other', (y_train == dict.cell_fate_map['Other']).sum())
    print('train Death', (y_train == dict.cell_fate_map['Death']).sum())
    print('train Neuron', (y_train == dict.cell_fate_map['Neuron']).sum())
    print('train Intestin', (y_train == dict.cell_fate_map['Intestin']).sum())
    print('train Muscle', (y_train == dict.cell_fate_map['Muscle']).sum())
    print('train Pharynx', (y_train == dict.cell_fate_map['Pharynx']).sum())
    print('train Skin', (y_train == dict.cell_fate_map['Skin']).sum())
    print('train Germ Cell', (y_train == dict.cell_fate_map['Germ Cell']).sum())

    print('test distribution', len(y_test))
    print('test Unspecified', (y_test == dict.cell_fate_map['Unspecified']).sum())
    print('test Other', (y_test == dict.cell_fate_map['Other']).sum())
    print('test Death', (y_test == dict.cell_fate_map['Death']).sum())
    print('test Neuron', (y_test == dict.cell_fate_map['Neuron']).sum())
    print('test Intestin', (y_test == dict.cell_fate_map['Intestin']).sum())
    print('test Muscle', (y_test == dict.cell_fate_map['Muscle']).sum())
    print('test Pharynx', (y_test == dict.cell_fate_map['Pharynx']).sum())
    print('test Skin', (y_test == dict.cell_fate_map['Skin']).sum())
    print('test Germ Cell', (y_test == dict.cell_fate_map['Germ Cell']).sum())

    print("Total dataset size:")
    print("n_samples: %d" % cshaper_X.values.shape[0])
    print("n_spectrum: %d" % cshaper_X.values.shape[1])
    print("n_classes: %d" % len(dict.cell_fate_dict))

    # ==================================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 24
    print("Extracting the top %d eigenshape from %d cells"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    print("Projecting the input data on the eigenshape orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    # Train a SVM classification model

    print("-----Fitting the classifier to the training set------")
    print('going through pipeline searching best classifier')
    t0 = time()

    # ===================================================================================
    nystroem_transformer = Nystroem(random_state=datetime.now().microsecond)
    linearsvc_classifier = LinearSVC(random_state=datetime.now().microsecond)
    pipe = Pipeline(
        [("scale", StandardScaler()), ("transformer", nystroem_transformer), ("classifier", linearsvc_classifier)])
    param_grid = {
        # "pca__n_components": [12,48, 96],
        # {'classifier__C': [1e3, 1e4] 1000.0, 'classifier__tol': [1e-2, 1e-3] 0.01, 'transformer__gamma': [0.0001, 0.001] 0.001}

        "transformer__gamma": [0.01, 0.001],
        "classifier__tol": [1e-2, 1e-1],
        "classifier__C": [1e3, 1e2]
    }
    # =====================================================================================

    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    clf = search.fit(X_train_pca, y_train)
    print(search.cv_results_)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    print("done in %0.3fs" % (time() - t0))

    # Quantitative evaluation of the model quality on the test set
    print("Predicting cell fate on the test set")
    t0 = time()
    y_pred = search.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))
    print(classification_report(y_test, y_pred, target_names=dict.cell_fate_dict[1:]))
    print(confusion_matrix(y_test, y_pred, labels=dict.cell_fate_num[1:]))
    # ==================================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# really stupid, figures for paper you just plot one by one then combine via vision or ppt any way
def figure_for_science_01paper():
    # Sample06,Capp,079
    # Sample06,ABalaapa,078
    print('waiting type you input1')
    embryo_name1, cell_name1, tp1 = str(input()).split(',')

    print('waiting type you input2')
    embryo_name, cell_name, tp = str(input()).split(',')
    # cell_name = str(input())
    # tp = str(input())

    embryo_path_csv = os.path.join(r'D:\cell_shape_quantification\DATA', r'my_data_csv\SH_time_domain_csv',
                                   embryo_name + 'LabelUnified_l_25_norm.csv')
    print(embryo_path_csv)
    embryo_csv = read_csv_to_df(embryo_path_csv)

    plt.rcParams['text.usetex'] = True

    fig_SPCSMs_info = plt.figure()

    axes_tmp1 = fig_SPCSMs_info.add_subplot(2, 2, 1, projection='3d')
    instance_tmp1 = pysh.SHCoeffs.from_array(
        collapse_flatten_clim(embryo_csv.loc[tp1 + '::' + cell_name1]))
    instance_tmp1_expanded = instance_tmp1.expand(lmax=100).data

    Y2d = np.arange(-90, 90, 180 / 203)
    X2d = np.arange(0, 360, 360 / 405)
    X2d, Y2d = np.meshgrid(X2d, Y2d)

    axes_tmp1.plot_surface(X2d, Y2d, instance_tmp1_expanded, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, rstride=60, cstride=10)
    axes_tmp1.set_zlabel(r'\textit{z}', fontsize=18)  # 坐标轴
    axes_tmp1.set_ylabel(r'\textit{y}', fontsize=18)
    axes_tmp1.set_xlabel(r'\textit{x}', fontsize=18)
    # axes_tmp1.colorbar()

    axes_tmp2 = fig_SPCSMs_info.add_subplot(2, 2, 2)
    grid_2 = instance_tmp1.expand(lmax=100)
    grid_2.plot(ax=axes_tmp2, cmap='RdBu', cmap_reverse=True, title='Heat Map',
                xlabel=r'\textit{x}', tick_labelsize=14, titlesize=18,
                ylabel=r'\textit{y}', axes_labelsize=18, tick_interval=[60, 60], colorbar='right',
                cb_label=r'\textit{z} value / 0.015625 $\mu M$')
    set_size(5, 5, ax=axes_tmp2)

    # embryo_path_name = embryo_name + 'LabelUnified'
    # embryo_path = os.path.join('D:/cell_shape_quantification/DATA/SegmentCellUnified04-20', embryo_path_name)
    # file_name = embryo_name + '_' + tp + '_segCell.nii.gz'
    # dict_cell_membrane, dict_center_points =  get_nib_embryo_membrane_dict(embryo_path,
    #                                                                                    file_name)
    instance_tmp = pysh.SHCoeffs.from_array(collapse_flatten_clim(embryo_csv.loc[tp + '::' + cell_name]))
    axes_tmp3 = fig_SPCSMs_info.add_subplot(2, 2, 3, projection='3d')
    draw_3D_points(do_reconstruction_for_SH(sample_N=50, sh_coefficient_instance=instance_tmp),
                   ax=axes_tmp3, cmap=cm.coolwarm)
    sn = 20
    x_axis = Arrow3D([0, sn + 3], [0, 0],
                     [0, 0], mutation_scale=20,
                     lw=3, arrowstyle="-|>", color="r")
    axes_tmp3.add_artist(x_axis)
    y_axis = Arrow3D([0, 0], [0, sn + 3],
                     [0, 0], mutation_scale=20,
                     lw=3, arrowstyle="-|>", color="r")
    axes_tmp3.add_artist(y_axis)
    z_axis = Arrow3D([0, 0], [0, 0],
                     [0, sn + 23], mutation_scale=20,
                     lw=3, arrowstyle="-|>", color="r")
    axes_tmp3.add_artist(z_axis)

    axis_points_num = 1000
    lineage_ = np.arange(0, sn, sn / axis_points_num)
    zeros_ = np.zeros(axis_points_num)
    # xz=np.zeros(axis_points_num)
    axes_tmp3.scatter3D(lineage_, zeros_, zeros_, s=10, color='r')
    axes_tmp3.scatter3D(zeros_, lineage_, zeros_, s=10, color='r')
    sn = sn + 15
    lineage_ = np.arange(0, sn, sn / axis_points_num)
    zeros_ = np.zeros(axis_points_num)
    axes_tmp3.scatter3D(zeros_, zeros_, lineage_, s=10, color='r')
    # axes_tmp.axis('off')

    # longitude circle
    x_lon = np.arange(0, 15, 15 / 1000)
    # print(x_lon)
    y_lon = np.sqrt(225 - np.power(x_lon, 2))
    # print(y_lon)
    axes_tmp3.scatter3D(x_lon, y_lon, np.zeros(1000), s=3, color='blue')
    axes_tmp3.text(19, 19, 0, r'\textit{longitude}', (-1, 1, 0), ha='center', fontsize=16)

    # latitude circle
    y_lat = np.arange(0, 15, 15 / 1000)
    z_lat = np.sqrt(225 - np.power(y_lat, 2))
    axes_tmp3.scatter3D(np.zeros(1000), y_lat, z_lat, s=3, color='black')
    axes_tmp3.text(0, 18, 18, r'\textit{latitude}', (-1, 1, 0), ha='center', fontsize=16)

    axes_tmp3.text(sn / 3 * 2, 0, -.2 * sn, r'\textit{x}', (-1, 1, 0), ha='center', fontsize=16).set_fontstyle('italic')
    axes_tmp3.text(0, sn / 3 * 2, -.2 * sn, r'\textit{y}', (-1, 1, 0), ha='center', fontsize=16).set_fontstyle('italic')
    axes_tmp3.text(-0.1 * sn, 0, sn + 10, r'\textit{z}', (-1, 1, 0), ha='center', fontsize=16).set_fontstyle('italic')
    axes_tmp3.set_zlabel(r'\textit{z}', fontsize=18)  # 坐标轴
    axes_tmp3.set_ylabel(r'\textit{y}', fontsize=18)
    axes_tmp3.set_xlabel(r'\textit{x}', fontsize=18)

    # axes_tmp3.annotate('XXXXXXXX', xy=(0.93, -0.01), ha='left', va='top', xycoords='axes fraction', weight='bold', style='italic')

    axes_tmp4 = fig_SPCSMs_info.add_subplot(2, 2, 4)
    grid_tmp = instance_tmp.expand(lmax=100)
    # axin=inset_axes(axes_tmp, width="50%", height="100%", loc=2)
    grid_tmp.plot(ax=axes_tmp4, cmap='RdBu', cmap_reverse=True, title='Spherical Grid', titlesize=18,
                  xlabel=r'Longitude \textit{x}-\textit{y} plane (\textit{degree} \textdegree)',
                  ylabel=r'Latitude \textit{y}-\textit{z} plane (\textit{degree} \textdegree)', axes_labelsize=18,
                  tick_labelsize=14,
                  tick_interval=[60, 60], colorbar='right', cb_label=r'distance from centroid / 0.015625 $\mu M$')

    fig_SPCSMs_info.text(0.05, 0.7, '3D Surface ', fontsize=24)
    fig_SPCSMs_info.text(0.05, 0.25, '3D Closed Surface', fontsize=24)
    # Sample06,Dpaap,158   Sample06,ABalaapa,078

    arrow = matplotlib.patches.FancyArrowPatch(
        (0.4, 0.7), (0.5, 0.7), transform=fig_SPCSMs_info.transFigure,  # Place arrow in figure coord system
        fc="g", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha=0.3,
        mutation_scale=40.
    )
    # 5. Add patch to list of objects to draw onto the figure
    fig_SPCSMs_info.patches.append(arrow)

    arrow = matplotlib.patches.FancyArrowPatch(
        (0.4, 0.3), (0.5, 0.3), transform=fig_SPCSMs_info.transFigure,  # Place arrow in figure coord system
        fc="g", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha=0.3,
        mutation_scale=40.
    )
    # 5. Add patch to list of objects to draw onto the figure
    fig_SPCSMs_info.patches.append(arrow)

    plt.show()
    # saving_path = os.path.join(
    #     r'C:\\Users\zelinli6\OneDrive - City University of Hong Kong\Documents\01paper\Reconstruction preseant',
    #     embryo_name + cell_name + tp + 'spherical_matrix.svg')
    # plt.savefig(saving_path, format='svg')


def calculate_SH_PCA_coordinate():
    PCA_matrices_saving_path = os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv\SH_PCA_coordinate',
                                            'SHc_norm_PCA.csv')

    path_saving_csv_normalized = os.path.join(config.dir_my_data_SH_time_domain_csv, 'SHc_norm.csv')
    df_SHc_norm = read_csv_to_df(path_saving_csv_normalized)
    print('finish read all embryo cell df_sh_norm_coefficients--------------')

    sh_PCA = PCA(n_components=24)
    pd.DataFrame(data=sh_PCA.fit_transform(df_SHc_norm.values), index=df_SHc_norm.index).to_csv(
        PCA_matrices_saving_path)


def plot_voxel_and_reconstructed_surface_01paper():
    # Sample05,ABpl,014
    # Sample04,ABpl,012
    """
    plot original surface and reconstructed surface through SPHARM
    """

    print('waiting type you input: sample name and time points for embryogenesis')
    embryo_name, cell_name, tp = str(input()).split(',')

    num_cell_name, cell_num = get_cell_name_affine_table()
    this_cell_keys = cell_num[cell_name]

    """
    the following plotted figure need configure then capture to .png
    
    camera position and rotation: paste action in figure window
    # Sample04,ABpl,012
    {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 58.949796990685456, 73.062096966802002, 17.53928827322666 ],
			"boundingbox_min" : [ -58.050203009314544, -64.937903033197998, -32.46071172677334 ],
			"field_of_view" : 60.0,
			"front" : [ -0.022170625825107981, 0.031107593735803508, -0.9992701241218469 ],
			"lookat" : [ 0.44979699068545642, 4.0620969668020024, -7.46071172677334 ],
			"up" : [ 0.039903629142300223, 0.99874686470810126, 0.030205969890268435 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
    }
    
    choose: shift+2
    press 3-5 time + to make points grow bigger
    remember: L to open shining and shadow
    
    then press P , the capture figure .png would save to where the python command you run 
    
    """
    # ===========plot original dialation surface for particular shape====================================
    # embryo_img = load_nitf2_img(
    #     os.path.join(config.data_path, 'Segmentation Results/SegmentedCell/' + embryo_name + 'LabelUnified',
    #                  embryo_name + '_' + tp + '_segCell.nii.gz')).get_fdata().astype(float)
    # cell_surface_points, center = nii_get_cell_surface(embryo_img, this_cell_keys)
    # # print((cell_surface_points - center).shape)
    # print(np.max((cell_surface_points-center),axis=0))
    # m_pcd = o3d.geometry.PointCloud()
    # m_pcd.points = o3d.utility.Vector3dVector(cell_surface_points - center)
    # m_pcd.estimate_normals()
    # o3d.visualization.draw_geometries([m_pcd])
    # =========================================================================================

    # ==============plot reconstruction surface============================================
    # the path need to change to non-norm path
    SHc_path = os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv/SH_time_domain_csv',
                            embryo_name + 'LabelUnified_l_25.csv')
    df_SHcPCA = read_csv_to_df(SHc_path)
    sh_instance = pysh.SHCoeffs.from_array(collapse_flatten_clim(df_SHcPCA.loc[tp + '::' + cell_name]))
    m_pcd = o3d.geometry.PointCloud()
    resctruct_xyz = do_reconstruction_for_SH(100, sh_instance)
    print(resctruct_xyz)
    pd.DataFrame(resctruct_xyz).to_csv("{}.csv".format(embryo_name + ' ' + cell_name + ' ' + tp))
    m_pcd.points = o3d.utility.Vector3dVector(resctruct_xyz)
    m_pcd.estimate_normals()
    # o3d.visualization.draw_geometries([m_pcd],{
    o3d.visualization.draw_geometries(geometry_list=[m_pcd],
                                      window_name=embryo_name + ' ' + cell_name + ' ' + tp,
                                      # boundingbox_max=[58.949796990685456, 73.062096966802002, 17.53928827322666],
                                      # boundingbox_min=[-58.050203009314544, -64.937903033197998, -32.46071172677334],
                                      # field_of_view=60.0,
                                      front=[-0.022170625825107981, 0.031107593735803508, -0.9992701241218469],
                                      lookat=[0.44979699068545642, 4.0620969668020024, -7.46071172677334],
                                      up=[0.039903629142300223, 0.99874686470810126, 0.030205969890268435],
                                      zoom=0.69999999999999996)

    # ==================================================================================


def plot_and_save_5_type_figures_01paper():
    # Sample05,ABpl,014

    print('waiting type you input: sample name and time points for embryogenesis')
    embryo_name, cell_name, tp = str(input()).split(',')

    num_cell_name, cell_num = get_cell_name_affine_table()
    this_cell_keys = cell_num[cell_name]

    embryo_img = load_nitf2_img(
        os.path.join(config.cell_shape_analysis_data_path,
                     'Segmentation Results/SegmentedCell/' + embryo_name + 'LabelUnified',
                     embryo_name + '_' + tp + '_segCell.nii.gz')).get_fdata().astype(float)
    cell_surface_points, center = nii_get_cell_surface(embryo_img, this_cell_keys)

    # ====================plot 2. 2D mapping parameterization============================
    plt.rcParams['text.usetex'] = True

    fig_SPCSMs_info = plt.figure()

    grid_data, _ = do_sampling_with_interval(24, cell_surface_points - center, average_num=3)

    axes_tmp4 = fig_SPCSMs_info.add_subplot(111)
    grid_tmp = SHGrid.from_array(grid_data)
    # axin=inset_axes(axes_tmp, width="50%", height="100%", loc=2)
    grid_tmp.plot(ax=axes_tmp4, cmap='RdBu', cmap_reverse=True, title='3D Surface Spherical Grid', tick_labelsize=14,
                  titlesize=18, axes_labelsize=18,
                  xlabel=r'Longitude - \textit{x}-\textit{y} plane (degree \textdegree)',
                  ylabel=r'Latitude \textit{y}-\textit{z} plane (degree \textdegree)',
                  tick_interval=[60, 60], colorbar='right', cb_label=r'Distance from centroid / 0.015625 $\mu M$')
    # plt.show()

    saving_path = os.path.join(
        r'C:\\Users\zelinli6\OneDrive - City University of Hong Kong\Documents\01paper\Reconstruction preseant',
        embryo_name + cell_name + tp + '2DMap.svg')
    plt.savefig(saving_path, format='svg')
    # =================================================================================================

    # ============== plot 3.  SPHARM surface ARRAY============================================
    plt.rcParams['text.usetex'] = True

    fig_SPCSMs_info = plt.figure()
    axes_tmp = fig_SPCSMs_info.add_subplot(111)

    # the path need to change to non-norm path
    SHc_path = os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv/SH_time_domain_csv',
                            embryo_name + 'LabelUnified_l_25.csv')
    df_SHcPCA = read_csv_to_df(SHc_path)
    sh_instance = pysh.SHCoeffs.from_array(collapse_flatten_clim(df_SHcPCA.loc[tp + '::' + cell_name]))
    sh_instance.plot_spectrum2d(title='SPHARM Coefficient Array', ax=axes_tmp, degree_label=r'SPAHRM degree \textit{l}',
                                tick_labelsize=14, titlesize=18, axes_labelsize=18,
                                order_label=r'SPAHRM order \textit{m}', cmap='RdBu',
                                cb_label='Component value Colorbar', cb_triangles='both')
    saving_path = os.path.join(
        r'C:\\Users\zelinli6\OneDrive - City University of Hong Kong\Documents\01paper\Reconstruction preseant',
        embryo_name + cell_name + tp + 'SPHARM.svg')
    # plt.show()
    plt.savefig(saving_path, format='svg')
    # ==================================================================================

    # ==============4. plot SPHARM spectrum vector============================================

    plt.rcParams['text.usetex'] = True

    # the path need to change to non-norm path
    SHc_path = os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv/SH_time_domain_csv',
                            embryo_name + 'LabelUnified_l_25.csv')
    df_SHcPCA = read_csv_to_df(SHc_path)
    sh_instance = pysh.SHCoeffs.from_array(collapse_flatten_clim(df_SHcPCA.loc[tp + '::' + cell_name]))
    print(sh_instance.spectrum())
    fig, ax = plt.subplots()
    # s_list = [7870, 81937, 17529598, 6225227]
    l_list = np.arange(1, 26)
    ax.bar(l_list, np.log(sh_instance.spectrum()[1:]))

    plt.title('Numbers of Four eventtypes')
    plt.xlabel('Eventtype')
    plt.ylabel('Number')
    plt.show()
    # saving_path = os.path.join(
    #     r'C:\\Users\zelinli6\OneDrive - City University of Hong Kong\Documents\01paper\Reconstruction preseant',
    #     embryo_name + cell_name + tp + 'SPHARM.svg')
    # plt.savefig(saving_path, format='svg')
    # ==================================================================================
    # Sample05,ABpl,014




def enhanced_graph_wavelet_feature():
    embryo_names = [str(i).zfill(2) for i in range(20, 21)]
    name_cellname, cellname_number = get_cell_name_affine_table()

    hop = 1
    for embryo_name in embryo_names:
        print('embryo  ', embryo_name)
        df_embryo_cell_fea = read_csv_to_df(
            os.path.join(config.cell_shape_analysis_data_path,
                         'my_data_csv/SH_time_domain_csv/Sample{}_Spectrum_norm.csv'.format(embryo_name)))

        df_enhanced_cell_fea = pd.DataFrame(index=df_embryo_cell_fea.index, columns=df_embryo_cell_fea.columns)
        # start frame!
        current_frame = '001'
        last_fame = '001'
        this_graph = iGraph.Graph.Read_GraphML(
            os.path.join(config.cell_shape_analysis_data_path, 'Graph_embryo/Sample' + embryo_name,
                         'Sample' + embryo_name + '_' + current_frame + '.graphml'))
        for idx in df_embryo_cell_fea.index:  # segmented successfully
            print(idx)
            cell_name, current_frame = idx.split('::')[1], idx.split('::')[0]
            if current_frame != last_fame:  # frame by frame to read the graph(cell graph base on contact) file
                # is time to read a new graph, move to next frame

                this_graph = iGraph.Graph.Read_GraphML(
                    os.path.join(config.cell_shape_analysis_data_path, 'Graph_embryo/Sample' + embryo_name,
                                 'Sample' + embryo_name + '_' + current_frame + '.graphml'))

            last_fame = current_frame
            star_node = this_graph.vs.find(name=cell_name)
            cellname_node_list = []
            for ith in range(hop + 1):
                # print(this_graph.neighborhood_size(star_node, order=ith))
                if ith == 0:
                    cellname_node_list.append(this_graph.neighborhood(star_node, order=ith))
                else:
                    previous_hop_node_num = len(this_graph.neighborhood(star_node, order=ith - 1))
                    cell_contact_kth_list = this_graph.neighborhood(star_node, order=ith)[previous_hop_node_num:]
                    if len(cell_contact_kth_list) > 0:
                        cellname_node_list.append(cell_contact_kth_list)
                    else:
                        break
            # get the node's features
            node_fea_list = []
            # for ith,ith_value in enumerate(cellname_node_list):
            for ith_value in cellname_node_list:
                for node_id in ith_value:
                    idx_tmp = current_frame + '::' + this_graph.vs[node_id]['name']
                    node_fea_list.append(list(df_embryo_cell_fea.loc[idx_tmp]))
            # enhance the feature by cell graph and graph wavelet
            phi_j_h_list = []
            tmp_C_j_vk_list = []
            for h_ in range(hop + 1):
                phi_j_h_ = phi_j_h_wavelet(hop, h_, wavelet='MexicanHat')
                phi_j_h_list.append(phi_j_h_)
                if h_ < len(cellname_node_list):
                    # print(h_,len(cellname_node_list),len(cellname_node_list[h_]),cellname_node_list[h_])
                    tmp_C_j_vk_list.append(phi_j_h_ ** 2 / len(cellname_node_list[h_]))
            C_j_vk = (sum(tmp_C_j_vk_list)) ** (-1 / 2)
            df_enhanced_cell_fea.loc[idx] = embryo_enhanced_graph_wavelet(node_fea_list, cellname_node_list,
                                                                          phi_j_h_list, C_j_vk)
            # df_enhanced_cell_fea.to_csv(os.path.join(config.data_path, 'my_data_csv/norm_Spectrum_graph_enhanced_csv',
            #                                          'Sample' + embryo_name + '_h{}_M.csv'.format(hop)))
        print(df_enhanced_cell_fea)
        df_enhanced_cell_fea.to_csv(
            os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv/norm_Spectrum_graph_enhanced_csv',
                         'Sample' + embryo_name + '_h{}_M.csv'.format(hop)))


def calculate_cell_norm_shape_asymmetric_between_sisters():
    cell_combine_tree, begin_frame = get_combined_lineage_tree()

    # EIGENGRID
    pca_num = 96
    path_saving_dynamic_eigengrid = config.cell_shape_analysis_data_path + r'my_data_csv/norm_2DMATRIX_PCA_csv'
    df_asymmetric_eigengrid = pd.DataFrame(columns=range(pca_num))
    df_dynamic_f = read_csv_to_df(
        os.path.join(path_saving_dynamic_eigengrid, 'Mean_cellLineageTree_dynamic_eigengrid.csv'))
    for cell_name in df_dynamic_f.index:
        # cell_combine_tree.expand_tree(sorting=False):

        # cell_name = idx.split('::')[1]
        # time_int, node_id=
        if not cell_combine_tree.contains(cell_name):
            continue

        children_list = cell_combine_tree.children(cell_name)

        if not len(children_list) == 2:
            continue
        else:
            child1 = children_list[0].identifier
            child2 = children_list[1].identifier
            print(cell_name, child1, child2)
            # print(np.power(list(df_dynamic_f.loc[child1]) - list(df_dynamic_f.loc[child2]), 2))
            df_asymmetric_eigengrid.loc[cell_name] = list(
                np.absolute(df_dynamic_f.loc[child1] - df_dynamic_f.loc[child2]))
    df_asymmetric_eigengrid.to_csv(
        os.path.join(path_saving_dynamic_eigengrid, 'asymmetric_Mean_cellLineageTree_dynamic_eigengrid.csv'))

    # EIGENHARMONIC
    pca_num = 12
    path_saving_dynamic_eigenharmonic = config.cell_shape_analysis_data_path + r'my_data_csv/norm_SH_PCA_csv'
    df_asymmetric_eigenharmonic = pd.DataFrame(columns=range(pca_num))
    df_dynamic_f = read_csv_to_df(
        os.path.join(path_saving_dynamic_eigenharmonic, 'Mean_cellLineageTree_dynamic_eigenharmonic.csv'))
    for cell_name in df_dynamic_f.index:
        # cell_combine_tree.expand_tree(sorting=False):

        # cell_name = idx.split('::')[1]
        # time_int, node_id=
        if not cell_combine_tree.contains(cell_name):
            continue

        children_list = cell_combine_tree.children(cell_name)

        if not len(children_list) == 2:
            continue
        else:
            child1 = children_list[0].identifier
            child2 = children_list[1].identifier
            print(cell_name, child1, child2)
            # print(np.power(list(df_dynamic_f.loc[child1]) - list(df_dynamic_f.loc[child2]), 2))
            df_asymmetric_eigenharmonic.loc[cell_name] = list(
                np.absolute(df_dynamic_f.loc[child1] - df_dynamic_f.loc[child2]))
    df_asymmetric_eigenharmonic.to_csv(
        os.path.join(path_saving_dynamic_eigenharmonic, 'asymmetric_Mean_cellLineageTree_dynamic_eigenharmonic.csv'))

    # energy spectrum
    f_num = 26
    path_saving_dynamic_spectrum = config.cell_shape_analysis_data_path + r'my_data_csv/norm_Spectrum_csv'
    df_asymmetric_spectrum = pd.DataFrame(columns=range(f_num))
    df_dynamic_f = read_csv_to_df(
        os.path.join(path_saving_dynamic_spectrum, 'Mean_cellLineageTree_dynamic_spectrum.csv'))
    for cell_name in df_dynamic_f.index:
        # cell_combine_tree.expand_tree(sorting=False):

        # cell_name = idx.split('::')[1]
        # time_int, node_id=
        if not cell_combine_tree.contains(cell_name):
            continue

        children_list = cell_combine_tree.children(cell_name)

        if not len(children_list) == 2:
            continue
        else:
            child1 = children_list[0].identifier
            child2 = children_list[1].identifier
            print(cell_name, child1, child2)
            # print(np.power(list(df_dynamic_f.loc[child1]) - list(df_dynamic_f.loc[child2]), 2))
            df_asymmetric_spectrum.loc[cell_name] = list(
                np.absolute(df_dynamic_f.loc[child1] - df_dynamic_f.loc[child2]))
    df_asymmetric_spectrum.to_csv(
        os.path.join(path_saving_dynamic_spectrum, 'asymmetric_Mean_cellLineageTree_dynamic_spectrum.csv'))

    # noC00spectrum
    f_num = 25
    path_saving_dynamic_noC00spectrum = config.cell_shape_analysis_data_path + r'my_data_csv/Spectrum_no_C00_csv'
    df_asymmetric_noC00spectrum = pd.DataFrame(columns=range(f_num))
    df_dynamic_f = read_csv_to_df(
        os.path.join(path_saving_dynamic_noC00spectrum, 'Mean_cellLineageTree_dynamic_spectrum_no_C00.csv'))
    for cell_name in df_dynamic_f.index:
        # cell_combine_tree.expand_tree(sorting=False):

        # cell_name = idx.split('::')[1]
        # time_int, node_id=
        if not cell_combine_tree.contains(cell_name):
            continue

        children_list = cell_combine_tree.children(cell_name)

        if not len(children_list) == 2:
            continue
        else:
            child1 = children_list[0].identifier
            child2 = children_list[1].identifier
            print(cell_name, child1, child2)
            # print(np.power(list(df_dynamic_f.loc[child1]) - list(df_dynamic_f.loc[child2]), 2))
            df_asymmetric_noC00spectrum.loc[cell_name] = list(
                np.absolute(df_dynamic_f.loc[child1] - df_dynamic_f.loc[child2]))
    df_asymmetric_noC00spectrum.to_csv(
        os.path.join(path_saving_dynamic_noC00spectrum, 'asymmetric_Mean_cellLineageTree_dynamic_noC00spectrum.csv'))


def cell_shape_asymmetric_boxplot():
    detected_cell_lineage = {
        'P0': ['P0'],
        'AB1': ['AB'],
        'P1': ['P1'],

        'AB2': ['ABa', 'ABp'],
        'EMS1': ['EMS'],
        'P2': ['P2'],

        'AB4': ['ABal', 'ABar', 'ABpl', 'ABpr'],
        'EMS2': ['E', 'MS'],
        'C1': ['C'],
        'P3': ['P3'],

        # 'AB8': ['ABpra', 'ABprp', 'ABala', 'ABalp', 'ABpla', 'ABplp', 'ABara', 'ABarp'],

    }

    embryo_names = [str(i).zfill(2) for i in range(4, 21)]
    name_cellname, cellname_number = get_cell_name_affine_table()

    cells_asymmetric1_1 = {}
    cells_asymmetric1_2 = {}

    cells_asymmetric2_1 = {}
    cells_asymmetric2_2 = {}

    for idx in detected_cell_lineage.keys():
        for cell_name in detected_cell_lineage[idx]:
            # print(cell_name)
            cells_asymmetric1_1[cell_name] = []
            cells_asymmetric1_2[cell_name] = []

            cells_asymmetric2_1[cell_name] = []
            cells_asymmetric2_2[cell_name] = []

    for embryo_name in embryo_names:
        # --------------------the tree of this embryo-------------------------------------
        cell_tree_file_path = os.path.join(config.win_cell_shape_analysis_data_path + r'lineage_tree/LifeSpan',
                                           'Sample{}_cell_life_tree'.format(embryo_name))
        with open(cell_tree_file_path, 'rb') as f:
            # print(f)
            embryo_tree = Tree(load(f))
        # -------------------------------------------------------------------------------

        # --------------------------the feature file path-- you can modify------------------
        eigengrid_file_path = os.path.join(config.win_cell_shape_analysis_data_path, 'my_data_csv/Spectrum_no_C00_csv',
                                           'Sample{}_dynamic_spectrum_no_C00.csv'.format(embryo_name))
        df_eigengrid = read_csv_to_df(eigengrid_file_path)
        for cell_name in cells_asymmetric1_1.keys():
            children_list = embryo_tree.children(cell_name)
            child1 = children_list[0].identifier
            child2 = children_list[1].identifier

            if child1 in df_eigengrid.index and child2 in df_eigengrid.index:
                # ------------------the dissimilarity between cells------------------

                cells_asymmetric1_1[cell_name].append(np.sqrt(
                    # np.sum(np.power(df_eigengrid.loc[child1][1:] - df_eigengrid.loc[child2][1:], 2))))
                    np.sum(np.power(df_eigengrid.loc[child1] - df_eigengrid.loc[child2], 2))))

                # tmp_cell_asymmetric=np.sqrt(np.sum(np.power(np.abs(df_eigengrid.loc[child1][:6] / df_eigengrid.loc[child2][:6]), 2)))
                # cells_asymmetric1_2[cell_name].append(tmp_cell_asymmetric if tmp_cell_asymmetric>1 else 1/tmp_cell_asymmetric)

                cells_asymmetric2_1[cell_name].append(np.abs(df_eigengrid.loc[child1][0] - df_eigengrid.loc[child2][0]))
                # tmp_cell_asymmetric=np.abs(df_eigengrid.loc[child1][0] / df_eigengrid.loc[child2][0])
                # cells_asymmetric2_2[cell_name].append(tmp_cell_asymmetric if tmp_cell_asymmetric>1 else 1/tmp_cell_asymmetric)

    df_symmetry_cellname = pd.DataFrame(
        # columns=['cell_name', 'asymmetry1-1', 'asymmetry1-2', 'asymmetry2-1', 'asymmetry2-2'])
        columns=['cell_name', 'asymmetry1-1', 'asymmetry2-1'])
    for cell_name in cells_asymmetric1_1.keys():
        for ith, _ in enumerate(cells_asymmetric1_1[cell_name]):
            df_symmetry_cellname.loc[str(ith) + '-' + cell_name] = [cell_name, cells_asymmetric1_1[cell_name][ith],
                                                                    # cells_asymmetric1_2[cell_name][ith],
                                                                    cells_asymmetric2_1[cell_name][ith],
                                                                    # cells_asymmetric2_2[cell_name][ith]]
                                                                    ]

    print(df_symmetry_cellname)
    sns.set_theme(style="whitegrid")
    # print(df_symmetry_cellname['asymmetry1-1'].quantile(.25))

    ax = sns.boxplot(x="asymmetry1-1", y="cell_name", data=df_symmetry_cellname, orient="h", showfliers=False)
    # ax = sns.boxplot(data=iris, orient="h", palette="Set2")
    plt.show()


def division_time_asymmetry(frame_time_ratio=1.39):
    # time segregation ratio
    embryo_names = [str(i).zfill(2) for i in range(4, 21)]
    for embryo_name in embryo_names:
        # --------------------the tree of this embryo-------------------------------------
        cell_tree_file_path = os.path.join(config.cell_shape_analysis_data_path + r'lineage_tree/LifeSpan',
                                           'Sample{}_cell_life_tree'.format(embryo_name))
        df_time_segregation_ratio = pd.DataFrame(
            columns=['cell_cycle_length', 'cell_cycle_asymmetric_ratio', 'division_time_asymmetry',
                     'cell_cycle_segregation_ratio', 'division_time_segregation_asymmetry', ])
        with open(cell_tree_file_path, 'rb') as f:
            # print(f)
            embryo_tree = Tree(load(f))
        # for cell_name in tqdm(embryo_tree.expand_tree(), desc="dealing with Sample{}".format(embryo_name)):
        for cell_name in embryo_tree.expand_tree(sorting=False):

            cell_time = embryo_tree.get_node(cell_name).data.get_time()
            # print(embryo_name,cell_name,cell_time)

            if len(cell_time) == 0:
                print('no this cell', 'Sample{}  '.format(embryo_name), cell_name)
            elif cell_time[0] < 150:
                # 1.cell_cycle_length
                cell_cycle_length = len(cell_time) * frame_time_ratio

                # parent=embryo_tree.parent(cell_name).identifier
                twins = embryo_tree.children(embryo_tree.parent(cell_name).identifier)
                sister = twins[1] if twins[0].identifier == cell_name else twins[0]
                sister_time = sister.data.get_time()
                # 2.cell_cycle_asymmetric_ratio
                cell_cycle_asymmetric_ratio = 0
                division_time_asymmetry = 0
                if len(sister_time) != 0:
                    cell_cycle_asymmetry_tmp = len(cell_time) / len(sister_time)
                    cell_cycle_asymmetric_ratio = cell_cycle_asymmetry_tmp if cell_cycle_asymmetry_tmp > 1 else 1 / cell_cycle_asymmetry_tmp
                    # 3.division_time_asymmetry
                    division_time_asymmetry = abs(cell_time[-1] - sister_time[-1]) * frame_time_ratio

                cell_cycle_segregation_ratio = 0
                division_time_segregation_asymmetry = 0
                children_list = embryo_tree.children(cell_name)
                if len(children_list) == 2:

                    child1_time = children_list[0].data.get_time()
                    child2_time = children_list[1].data.get_time()

                    if len(child1_time) != 0 and len(child2_time) != 0:
                        # 4.cell_cycle_segregation_ratio
                        cell_cycle_segregation_ratio_tmp = len(child1_time) / len(child2_time)
                        cell_cycle_segregation_ratio = cell_cycle_segregation_ratio_tmp if cell_cycle_segregation_ratio_tmp > 1 else 1 / cell_cycle_segregation_ratio_tmp
                        # 5. division_time_segregation_asymmetry
                        division_time_segregation_asymmetry = abs(child1_time[-1] - child2_time[-1]) * frame_time_ratio
                # add these 5 time feature (characteristics) in to dataframe
                df_time_segregation_ratio.loc[cell_name] = [cell_cycle_length, cell_cycle_asymmetric_ratio,
                                                            division_time_asymmetry, cell_cycle_segregation_ratio,
                                                            division_time_segregation_asymmetry]
        df_time_segregation_ratio.to_csv(
            os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv/time_segregation_ratio',
                         'Sample{}.csv'.format(embryo_name)))
    # we need to calculate ficision time asymmetric and cell cycle length


def sns_pairplot_visualization():
    # get cell fate dictionary
    df_cell_fate = pd.read_csv(os.path.join(config.cell_shape_analysis_data_path, 'CellFate.csv'))
    cell_fate_dict = {}
    for idx in df_cell_fate.index:
        cell_fate_dict[df_cell_fate.at[idx, 'Name'].strip('\'')] = df_cell_fate.at[idx, 'Fate'].strip('\'')

    # eigengrid or what, just change the directory path and the file name
    path_saving_dynamic_eigengrid = config.cell_shape_analysis_data_path + r'my_data_csv/norm_Spectrum_csv'
    df_dynamic_f = read_csv_to_df(
        os.path.join(path_saving_dynamic_eigengrid, 'Mean_cellLineageTree_static_Spectrum.csv'))

    y_fate = []
    for cell_name in df_dynamic_f.index:
        cell_name = cell_name.split('::')[1]
        if cell_name in cell_fate_dict.keys():
            y_fate.append(cell_fate_dict[cell_name])
        else:
            y_fate.append('Unspecified')

    df_used = pd.DataFrame(df_dynamic_f.values[:, 1:11])
    df_used['cell_fate'] = pd.Series(np.array(y_fate))
    # df_dynamic_f['cell_fate'] = pd.Series(digits.target).map(lambda x: ' {}'.format(cell_fate_map[x]))
    # https://seaborn.pydata.org/generated/seaborn.pairplot.html !!!!!!!!!!!!!!!!!!!!
    sns.pairplot(df_used, hue='cell_fate')
    plt.show()


def t_sne_visualization():
    df_cell_fate = pd.read_csv(os.path.join(config.cell_shape_analysis_data_path, 'CellFate.csv'))
    cell_fate_dict = {}
    for idx in df_cell_fate.index:
        cell_fate_dict[df_cell_fate.at[idx, 'Name'].strip('\'')] = df_cell_fate.at[idx, 'Fate'].strip('\'')

    time_start_threshold = 150
    # eigengrid or what, just change the directory path and the file name
    path_saving_dynamic_eigengrid = config.cell_shape_analysis_data_path + r'my_data_csv/norm_Spectrum_graph_enhanced_csv'
    df_dynamic_f = read_csv_to_df(
        os.path.join(path_saving_dynamic_eigengrid, 'Mean_cellLineageTree_static_enhanced_h3_M.csv'))
    # ------------different color means different cell fate ----------------
    y_label = []
    X_fea = []
    for idx in df_dynamic_f.index:
        cell_frame = idx.split('::')[0]
        if int(cell_frame) > time_start_threshold:
            X_fea.append(df_dynamic_f.loc[idx])
            cell_name = idx.split('::')[1]
            if cell_name in cell_fate_dict.keys():
                y_label.append(cell_fate_dict[cell_name])
            else:
                y_label.append('Unspecified')

    fea_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(np.array(X_fea))
    df_dynamic_f['cell_fate'] = pd.Series(np.array(y_label))
    # ---------------plot with plt.scatter---------------
    colors = [sns.color_palette()[x] for x in range(9)]
    cmap = ListedColormap(colors)
    plt.scatter(
        fea_embedded[:, 0],
        fea_embedded[:, 1],
        c=[x for x in pd.Series(np.array(y_label)).map(cell_fate_map)], cmap=cmap)
    # c=y_label)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('TSNE projection of the cell feature dataset', fontsize=24)

    plt.colorbar()
    plt.show()


def umap_visualization_1():
    df_cell_fate = pd.read_csv(os.path.join(config.cell_shape_analysis_data_path, 'CellFate.csv'))
    cell_fate_dict = {}
    for idx in df_cell_fate.index:
        cell_fate_dict[df_cell_fate.at[idx, 'Name'].strip('\'')] = df_cell_fate.at[idx, 'Fate'].strip('\'')

    time_start_threshold = 150
    # eigengrid or what, just change the directory path and the file name
    path_saving_dynamic_eigengrid = config.cell_shape_analysis_data_path + r'my_data_csv/norm_Spectrum_graph_enhanced_csv'
    df_dynamic_f = read_csv_to_df(
        os.path.join(path_saving_dynamic_eigengrid, 'Mean_cellLineageTree_static_enhanced_h3_M.csv'))

    # ------------different color means different cell fate ----------------
    y_label = []
    X_fea = []
    for idx in df_dynamic_f.index:
        cell_frame = idx.split('::')[0]
        if int(cell_frame) > time_start_threshold:
            X_fea.append(df_dynamic_f.loc[idx])
            cell_name = idx.split('::')[1]
            if cell_name in cell_fate_dict.keys():
                y_label.append(cell_fate_dict[cell_name])
            else:
                y_label.append('Unspecified')

    # # -----------different color means different time -----------
    # y_label = []
    # for cell_name in df_dynamic_f.index:
    #     cell_frame=cell_name.split('::')[1]
    #     y_label.append(int(int(cell_frame)/20))

    reducer = umap.UMAP()
    scaled_cell_fea_data = StandardScaler().fit_transform(X_fea)
    embedding = reducer.fit_transform(scaled_cell_fea_data)
    df_dynamic_f['cell_fate'] = pd.Series(np.array(y_label))
    # ---------------plot with plt.scatter---------------
    colors = [sns.color_palette()[x] for x in range(9)]
    cmap = ListedColormap(colors)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[x for x in pd.Series(np.array(y_label)).map(cell_fate_map)], cmap=cmap)
    # c=y_label)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the cell feature dataset', fontsize=24)

    plt.colorbar()
    plt.show()

    # mapper=umap.UMAP().fit(df_dynamic_f.values[:,1:])
    # umap.plot.points(mapper, labels=np.array(y_label), theme='fire')
    # plt.show()


if __name__ == "__main__":
    print('test2 run')
    construct_cell_graph_each_embryo()
    # correlation_matrix_of_original_cell_shape_fea_and_time_fea()
    # enhanced_graph_wavelet_feature()
    # division_time_asymmetry()

    # plot_voxel_and_reconstructed_surface_01paper()
    # figure_for_science_01paper()
    # while (True):
    #     plot_voxel_and_reconstructed_surface_01paper()
