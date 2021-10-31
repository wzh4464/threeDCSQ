import functional_func.draw_func as draw_pack
import functional_func.general_func as general_f
import functional_func.cell_func as cell_f
import config
import numpy as np
import os
import pandas as pd

from sklearn.cluster import KMeans

from time import time

import functional_func.draw_func as draw_f
import pyshtools as pysh
import particular_func.SH_represention as sh_represent
import particular_func.SH_analyses as sh_analysis
import matplotlib.pyplot as plt
import dict as dict

from tqdm import tqdm

from scipy import spatial

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from matplotlib import cm
import matplotlib.patches

from datetime import datetime

from matplotlib.font_manager import FontProperties

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
    path_norm = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc_norm.csv')

    saving_original_csv = pd.DataFrame(columns=np.arange(start=0, stop=26, step=1))
    df_csv = general_f.read_csv_to_df(path_original)
    for row_idx in df_csv.index:
        saving_original_csv.loc[row_idx] = pysh.SHCoeffs.from_array(
            sh_analysis.collapse_flatten_clim(df_csv.loc[row_idx])).spectrum()
        # print(num_idx)
        # print(saving_original_csv)
    print(saving_original_csv)
    saving_original_csv.to_csv(
        os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc_Spectrum.csv'))

    saving_norm_csv = pd.DataFrame(columns=np.arange(start=0, stop=26, step=1))
    df_csv = general_f.read_csv_to_df(path_norm)
    for row_idx in df_csv.index:
        saving_norm_csv.loc[row_idx] = pysh.SHCoeffs.from_array(
            sh_analysis.collapse_flatten_clim(df_csv.loc[row_idx])).spectrum()
        # print(num_idx)
        # print(saving_original_csv)
    saving_norm_csv.to_csv(
        os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc_norm_Spectrum.csv'))
    print(saving_norm_csv)


def cluster_with_spectrum():
    # Neuron, Pharynx, Intestine, Skin, Muscle, Germcell, death, unspecifed
    cluster_num = 6
    estimator = KMeans(n_clusters=cluster_num, max_iter=10000)
    path_original = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv',
                                 'SHc_norm_Spectrum.csv')
    concat_df_Spectrum = general_f.read_csv_to_df(path_original)
    result_origin = estimator.fit_predict(np.power(concat_df_Spectrum.values, 1 / 2))
    print(estimator.cluster_centers_)
    df_kmeans_clustering = pd.DataFrame(index=concat_df_Spectrum.index, columns=['cluster_num'])
    df_kmeans_clustering['cluster_num'] = result_origin
    df_kmeans_clustering.to_csv(
        os.path.join(config.dir_my_data_SH_clustering_csv,
                     'normsqrt_spectrum_cluster_k{}.csv'.format(cluster_num)))


def build_label_supervised_learning():
    cshaper_data_label_df = pd.DataFrame(columns=['Fate'])

    path_original = os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv',
                                 'SHc_norm_Spectrum.csv')
    concat_df_Spectrum = general_f.read_csv_to_df(path_original)

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
        cshaper_data_label_df.loc[idx] = fate_dict[tmp_cell_name]
    print(cshaper_data_label_df)
    cshaper_data_label_df.to_csv(os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv',
                                              '17_embryo_fate_label.csv'))
    # print(dfs.loc[dfs['Name'] == 'ABalaaaalpa\'']['Fate'])


def SPCSMs_SVM():
    print('reading dsf')
    t0 = time()
    cshaper_X = general_f.read_csv_to_df(
        os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv', 'SHc_norm.csv'))
    cshaper_Y = general_f.read_csv_to_df(
        os.path.join('D:/cell_shape_quantification/DATA/my_data_csv/SH_time_domain_csv',
                     '17_embryo_fate_label.csv'))
    X_train, X_test, y_train, y_test = train_test_split(
        cshaper_X.values, cshaper_Y.values.reshape((cshaper_Y.values.shape[0],)), test_size=0.2,
        random_state=datetime.now().microsecond)
    print("reading done in %0.3fs" % (time() - t0))

    print(X_train.shape)
    print(y_train.shape)

    print("Total dataset size:")
    print("n_samples: %d" % cshaper_X.values.shape[0])
    print("n_spectrum: %d" % cshaper_X.values.shape[1])
    print("n_classes: %d" % len(dict.cell_fate_dict))

    # #############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 48

    print("Extracting the top %d eigenfaces from %d cells"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',S
              whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    print("Projecting the input data on the eigenshape orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    #
    # print(X_train.shape)
    # print(X_train_pca.shape)
    # print(X_test_pca.shape)
    # print(pca.inverse_transform(X_train_pca))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    #
    # param_grid = {'C': [1e4],
    #               'gamma': [0.001], }
    # clf = GridSearchCV(
    #     SVC(kernel='rbf', class_weight='balanced'), param_grid
    # )
    # clf = clf.fit(X_train_pca, y_train)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    #
    # print(y_train.shape)
    #
    # random_indices=np.random.choice(X_train_pca.shape[0],size=100000,replace=False)
    # print(random_indices)
    # X_train_cut=X_train_pca[random_indices,:]
    #
    # print(X_train_cut.shape)
    # print(X_train_pca.shape)
    #
    # y_train_cut=y_train[random_indices]
    #
    #
    # print(y_train.shape)

    clf.fit(X_train_pca, y_train)
    print('test score', clf.score(X_test_pca, y_test))

    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting cell fate on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    print(clf.classes_)
    print(classification_report(y_test, y_pred, target_names=dict.cell_fate_dict))
    print(confusion_matrix(y_test, y_pred, labels=dict.cell_fate_dict))

    print("Predicting probability of cell fate on the test set")
    t0 = time()
    clf.predict_proba(X_test_pca).tofile('test_proba.csv', sep=',', format='%10.5f')
    print("done in %0.3fs" % (time() - t0))


def draw_figure_for_science():
    # Sample06,Dpaap,158
    # Sample06,ABalaapa,078
    print('waiting type you input1')
    embryo_name1, cell_name1, tp1 = str(input()).split(',')

    print('waiting type you input2')
    embryo_name, cell_name, tp = str(input()).split(',')
    # cell_name = str(input())
    # tp = str(input())

    print(embryo_name, cell_name, tp)

    embryo_path_csv = os.path.join(r'D:\cell_shape_quantification\DATA\my_data_csv\SH_time_domain_csv',
                                   embryo_name + 'LabelUnified_l_25_norm.csv')
    embryo_csv = general_f.read_csv_to_df(embryo_path_csv)

    # plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {'text.usetex': True,
              }
    plt.rcParams.update(params)
    fig_SPCSMs_info = plt.figure()

    axes_tmp1 = fig_SPCSMs_info.add_subplot(2, 2, 1, projection='3d')
    instance_tmp1 = pysh.SHCoeffs.from_array(
        sh_analysis.collapse_flatten_clim(embryo_csv.loc[tp1 + '::' + cell_name1])).expand(lmax=100)
    instance_tmp1_expanded = instance_tmp1.data

    Y2d = np.arange(-90, 90, 180 / 203)
    X2d = np.arange(0, 360, 360 / 405)
    X2d, Y2d = np.meshgrid(X2d, Y2d)

    axes_tmp1.plot_surface(X2d, Y2d, instance_tmp1_expanded, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, rstride=60, cstride=10)

    axes_tmp2 = fig_SPCSMs_info.add_subplot(2, 2, 2)
    instance_tmp1.plot(ax=axes_tmp2, cmap='RdBu', cmap_reverse=True, title='Heat Map',
                       xlabel='x of X-Y plane',
                       ylabel='y of X-Y plane', axes_labelsize=12, tick_interval=[60, 60])
    draw_f.set_size(5, 5, ax=axes_tmp2)

    # embryo_path_name = embryo_name + 'LabelUnified'
    # embryo_path = os.path.join('D:/cell_shape_quantification/DATA/SegmentCellUnified04-20', embryo_path_name)
    # file_name = embryo_name + '_' + tp + '_segCell.nii.gz'
    # dict_cell_membrane, dict_center_points = sh_represent.get_nib_embryo_membrane_dict(embryo_path,
    #                                                                                    file_name)
    instance_tmp = pysh.SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(embryo_csv.loc[tp + '::' + cell_name]))
    axes_tmp3 = fig_SPCSMs_info.add_subplot(2, 2, 3, projection='3d')
    draw_pack.draw_3D_points(sh_analysis.do_reconstruction_for_SH(sample_N=50, sh_coefficient_instance=instance_tmp),
                             ax=axes_tmp3, cmap=cm.coolwarm)
    sn = 20
    x_axis = draw_f.Arrow3D([0, sn + 3], [0, 0],
                            [0, 0], mutation_scale=20,
                            lw=3, arrowstyle="-|>", color="r")
    axes_tmp3.add_artist(x_axis)
    y_axis = draw_f.Arrow3D([0, 0], [0, sn + 3],
                            [0, 0], mutation_scale=20,
                            lw=3, arrowstyle="-|>", color="r")
    axes_tmp3.add_artist(y_axis)
    z_axis = draw_f.Arrow3D([0, 0], [0, 0],
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
    axes_tmp3.text(23, 23, 0, 'longitude', (-1, 1, 0), ha='center')

    # latitude circle
    y_lat = np.arange(0, 15, 15 / 1000)
    z_lat = np.sqrt(225 - np.power(y_lat, 2))
    axes_tmp3.scatter3D(np.zeros(1000), y_lat, z_lat, s=3, color='black')
    axes_tmp3.text(0, 18, 18, 'latitude', (-1, 1, 0), ha='center')

    axes_tmp3.text(sn / 3 * 2, 0, -.2 * sn, 'x', (-1, 1, 0), ha='center').set_fontstyle('italic')
    axes_tmp3.text(0, sn / 3 * 2, -.2 * sn, 'y', (-1, 1, 0), ha='center').set_fontstyle('italic')
    axes_tmp3.text(-0.1 * sn, 0, sn + 10, 'z', (-1, 1, 0), ha='center').set_fontstyle('italic')

    axes_tmp3.text('XXXXXXXX', xy=(0.93, -0.01), ha='left', va='top', xycoords='axes fraction', weight='bold',
                   style='italic')
    # axes_tmp3.annotate('XXXXXXXX', xy=(0.93, -0.01), ha='left', va='top', xycoords='axes fraction', weight='bold', style='italic')

    axes_tmp4 = fig_SPCSMs_info.add_subplot(2, 2, 4)
    grid_tmp = instance_tmp.expand(lmax=100)
    # axin=inset_axes(axes_tmp, width="50%", height="100%", loc=2)
    grid_tmp.plot(ax=axes_tmp4, cmap='RdBu', cmap_reverse=True, title='Heat Map',
                  xlabel='Longitude (X-Y plane)',
                  ylabel='Latitude (Y-Z plane)', axes_labelsize=12, tick_interval=[60, 60])

    fig_SPCSMs_info.text(0, 0.7, '3D Surface Mapping', fontsize=12)
    fig_SPCSMs_info.text(0, 0.25, '3D Object Mapping', fontsize=12)
    # Sample06,Dpaap,158
    # Sample06,ABalaapa,078

    # axes_tmp1.add_patch(
    #     matplotlib.patches.Rectangle((200., -4.), 50., 6., transform=axes_tmp1.transData, alpha=0.3, color="g"))

    arrow = matplotlib.patches.FancyArrowPatch(
        (0.4, 0.7), (0.6, 0.7), transform=fig_SPCSMs_info.transFigure,  # Place arrow in figure coord system
        fc="g", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha=0.3,
        mutation_scale=40.
    )
    # 5. Add patch to list of objects to draw onto the figure
    fig_SPCSMs_info.patches.append(arrow)

    arrow = matplotlib.patches.FancyArrowPatch(
        (0.4, 0.3), (0.6, 0.3), transform=fig_SPCSMs_info.transFigure,  # Place arrow in figure coord system
        fc="g", connectionstyle="arc3,rad=0.2", arrowstyle='simple', alpha=0.3,
        mutation_scale=40.
    )
    # 5. Add patch to list of objects to draw onto the figure
    fig_SPCSMs_info.patches.append(arrow)

    plt.show()


if __name__ == "__main__":
    print('test2 run')

    SPCSMs_SVM()
