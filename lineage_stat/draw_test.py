# import dependency package
import pickle as pkl
import pandas as pd

from treelib import Tree

import os
import numpy as np
import sys
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# import user package

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import lineage_stat.data_structure as data_struct
from lineage_stat.lineage_tree import draw_cell_lineage_tree
from utils.general_func import read_csv_to_df
from static.config import win_cell_shape_analysis_data_path


# ---------------------------------------------------------------------------------
# ##for all, how to combine 20 embryos to one lineage tree pic? a big problem!#####
# ---------------------------------------------------------------------------------


def draw_static_each_embryo_cell_lineage_tree(max_frame=100,showing=False):
    for embryo_index in np.arange(start=4, stop=21, step=1):
        # if embryo_index == 7:
        # path_tmp = r'./DATA/SegmentCellUnified04-20/Sample' + f'{cell_index:02}' + 'LabelUnified'
        # print(path_tmp)
        # ===========================draw lineage for one embryo=======================================================

        embryo_num = f'{embryo_index:02}'
        embryo_name = 'Sample{}LabelUnified'.format(embryo_num)
        print(embryo_name)

        # --------------read the tree with node and time list in data -----------------
        tmp_path = cell_shape_analysis_data_path + r'lineage_tree/LifeSpan/Sample{}_cell_life_tree'.format(embryo_num)
        with open(tmp_path, 'rb') as f:
            # print(f)
            cell_life_tree = Tree(pkl.load(f))

        # cell_life_tree.show(key=False)

        # ----------------------------------------------------------------------------
        draw_weight_of_eigengrid(embryo_name, cell_life_tree, max_frame=max_frame,showing=showing)
        # draw_SHCPCA_KMEANS(embryo_name, embryo_time_tree, id_root_tmp)
        #
        # draw_euclidean_tree(embryo_name, embryo_time_tree, id_root_tmp)
        # draw_norm_spectrum_Kmeans(embryo_name, embryo_time_tree, id_root_tmp)


def draw_weight_of_eigengrid(embryo_name, embryo_time_tree, print_num=1, max_frame=100,showing=False):
    """
    draw the eigengrid weight vector coefficient
    :param max_frame:
    :param embryo_name:
    :param embryo_time_tree:
    :param print_num:
    :return:
    """
    # ==================drawing for PCA; multiple lineage tree pictures for one embryo========================
    pca_num = 12

    # ----------------read SHcPCA result first--------------------------------
    path_SHcPCA_csv = os.path.join(win_cell_shape_analysis_data_path + r'my_data_csv/norm_2DMATRIX_PCA_csv',
                                   embryo_name + '_2Dmatrix_PCA.csv'.format(pca_num))
    df_SHcPCA_target = read_csv_to_df(path_SHcPCA_csv)

    # print('finished read the SHcPCA----->>', path_SHcPCA_csv)
    # ----------------------------------------------------------------------------

    # https: // www.webucator.com / article / python - color - constants - module /
    # colors = ['red4', 'red3', 'red2', 'red1', 'orangered1', 'orange', 'yellow2','yellow1','yellow2', 'lightblue1',lightblue', 'dodgerblue1',
    #           'dodgerblue2', 'dodgerblue3', 'dodgerblue4']

    # colors = np.array(
    #     [(139, 0, 0), (205, 0, 0), (238, 0, 0), (255, 0, 0), (255, 69, 0), (255, 128, 0),
    #      (238, 238, 0), (255, 255, 0), (238, 238, 0),
    #      (89, 210, 255), (63, 180, 255), (30, 144, 255), (28, 134, 238), (24, 116, 205), (16, 78, 139)]) / 255
    # colors2 = ['red4', 'red3', 'red2', 'red1', 'orangered1', 'orange', 'darkolivegreen3','darkolivegreen2','darkolivegreen3', 'lightblue1',lightblue', 'dodgerblue1',
    #           'dodgerblue2', 'dodgerblue3', 'dodgerblue4']
    colors2 = np.array(
        [(139, 0, 0), (205, 0, 0), (238, 0, 0), (255, 0, 0), (255, 69, 0), (255, 128, 0),
         (162, 205, 90), (188, 238, 104), (162, 205, 90),
         (89, 210, 255), (63, 180, 255), (30, 144, 255), (28, 134, 238), (24, 116, 205), (16, 78, 139)]) / 255
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors2)

    for i_PCA in range(print_num):
        # print(pd.Series(index=df_SHcPCA_target.index, data=df_SHcPCA_target[str(i_PCA)]).to_dict().keys())
        draw_cell_lineage_tree(embryo_time_tree, values_dict=pd.Series(index=df_SHcPCA_target.index,
                                                                       data=df_SHcPCA_target[str(i_PCA)]).to_dict(),
                               plot_title=embryo_name.split('L')[0] + '\'s eigengrid ' + str(i_PCA), is_abs=True,
                               color_map=cmap1, is_frame=True, time_resolution=1.39, end_time_point=max_frame, path_saving=os.path.join(cell_shape_analysis_data_path, r'lineage_tree\tree_plot\eigengrid'))
    # =================================================================================================


def draw_spharm_pca_lifespan(embryo_name, embryo_time_tree, print_num=4):
    # ==================drawing for PCA; multiple lineage tree pictures for one embryo========================
    pca_num = 12


def draw_static_eigengrid_meantree(print_num=2):
    # ==================drawing for PCA; multiple lineage tree pictures for one embryo========================

    embryo_names = [str(i).zfill(2) for i in range(4, 21)]
    norm_2dmatrix_pca_csv_path = cell_shape_analysis_data_path + r'my_data_csv/norm_2DMATRIX_PCA_csv'
    df_pd_values_dict = {}
    # ----------------read SHcPCA result first--------------------------------
    for embryo_name in embryo_names:
        path_SHcPCA_csv = os.path.join(norm_2dmatrix_pca_csv_path,
                                       'Sample' + embryo_name + 'LabelUnified_2Dmatrix_PCA.csv')
        df_pd_values_dict[embryo_name] = read_csv_to_df(path_SHcPCA_csv)
    # ----------------------------------------------------------------------------

    cell_combine_tree, begin_frame = data_struct.get_combined_lineage_tree()

    # frame = time / 1.39 +begin_frame
    for i in range(print_num):
        column = str(i+1)
        values_dict = {}
        for node_id in cell_combine_tree.expand_tree(sorting=False):
            for time_int in cell_combine_tree.get_node(node_id).data.get_time():
                # if time_int > 0:
                tp_value_list = []
                for embryo_name in df_pd_values_dict.keys():
                    frame_int = int(time_int / 1.39 + begin_frame[embryo_name])
                    frame_and_cell_index = f'{frame_int:03}' + '::' + node_id
                    if frame_and_cell_index in df_pd_values_dict[embryo_name].index:
                        # print(frame_and_cell_index,df_pd_values_dict[embryo_name].loc[frame_and_cell_index][column])
                        tp_value_list.append(df_pd_values_dict[embryo_name].at[frame_and_cell_index, column])

                # we have already got all values at this time from all(17) embryos, we just need to draw its average
                tp_and_cell_index = f'{time_int:03}' + '::' + node_id
                if len(tp_value_list) == 0:  # need to do interpolation
                    print(tp_and_cell_index, tp_value_list)
                else:
                    values_dict[tp_and_cell_index] = np.average(tp_value_list)

        # do interpolation for the lost cell value!
        for node_id in cell_combine_tree.expand_tree(sorting=False):
            for time_int in cell_combine_tree.get_node(node_id).data.get_time():
                tp_value_list = []
                tp_and_cell_index = f'{time_int:03}' + '::' + node_id

                if tp_and_cell_index not in values_dict.keys():  # need to do interpolation
                    print(tp_and_cell_index, tp_value_list)
                    if cell_combine_tree.get_node(node_id).is_leaf() and time_int == \
                            cell_combine_tree.get_node(node_id).data.get_time()[-1]:
                        pass
                    else:
                        for i in range(10):
                            complement_pre_tp_and_cell_index = f'{(time_int - i):03}' + '::' + node_id
                            complement_post_tp_and_cell_index = f'{(time_int + i):03}' + '::' + node_id
                            if complement_pre_tp_and_cell_index in values_dict.keys():
                                values_dict[tp_and_cell_index] = values_dict[complement_pre_tp_and_cell_index]
                                break
                            elif complement_post_tp_and_cell_index in values_dict.keys():
                                values_dict[tp_and_cell_index] = values_dict[complement_post_tp_and_cell_index]
                                break

        # https: // www.webucator.com / article / python - color - constants - module /
        # colors2 = ['red4', 'red3', 'red2', 'red1', 'orangered1', 'orange', 'darkolivegreen2','darkolivegreen1','darkolivegreen2', 'lightblue1',lightblue', 'dodgerblue1',
        #           'dodgerblue2', 'dodgerblue3', 'dodgerblue4']
        colors2 = np.array(
            [(139, 0, 0), (205, 0, 0), (238, 0, 0), (255, 0, 0), (255, 69, 0), (255, 128, 0),
             (188, 238, 104), (202, 255, 112), (188, 238, 104),
             (89, 210, 255), (63, 180, 255), (30, 144, 255), (28, 134, 238), (24, 116, 205), (16, 78, 139)]) / 255

        # cmap_list = ListedColormap(colors)
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors2)
        draw_cell_lineage_tree(cell_combine_tree, values_dict=values_dict,
                               plot_title='(Static) Eigengrid ' + column + ' on average cell lineage tree',
                               color_map=cmap1)


def draw_static_eigenharmonic_meantree(print_num=2):
    # ==================drawing for PCA; multiple lineage tree pictures for one embryo========================
    pca_num = 12

    embryo_names = [str(i).zfill(2) for i in range(4, 21)]
    norm_shcpca_csv_path = cell_shape_analysis_data_path + r'my_data_csv/norm_SH_PCA_csv'
    df_pd_values_dict = {}
    # ----------------read SHcPCA result first--------------------------------
    for embryo_name in embryo_names:
        path_SHcPCA_csv = os.path.join(norm_shcpca_csv_path,
                                       'Sample' + embryo_name + 'LabelUnified_SHcPCA' + str(pca_num) + '_norm.csv')
        df_pd_values_dict[embryo_name] = read_csv_to_df(path_SHcPCA_csv)
    # ----------------------------------------------------------------------------

    cell_combine_tree, begin_frame = data_struct.get_combined_lineage_tree()

    # frame = time / 1.39 +begin_frame
    for i in range(print_num):
        column = str(i)
        values_dict = {}
        for node_id in cell_combine_tree.expand_tree(sorting=False):
            for time_int in cell_combine_tree.get_node(node_id).data.get_time():
                # if time_int > 0:
                tp_value_list = []
                for embryo_name in df_pd_values_dict.keys():
                    frame_int = int(time_int / 1.39 + begin_frame[embryo_name])
                    frame_and_cell_index = f'{frame_int:03}' + '::' + node_id
                    if frame_and_cell_index in df_pd_values_dict[embryo_name].index:
                        # print(frame_and_cell_index,df_pd_values_dict[embryo_name].loc[frame_and_cell_index][column])
                        tp_value_list.append(df_pd_values_dict[embryo_name].at[frame_and_cell_index, column])

                # we have already got all values at this time from all(17) embryos, we just need to draw its average
                tp_and_cell_index = f'{time_int:03}' + '::' + node_id
                if len(tp_value_list) == 0:  # need to do interpolation
                    print(tp_and_cell_index, tp_value_list)
                else:
                    values_dict[tp_and_cell_index] = np.average(tp_value_list)

        # do interpolation for the lost cell value!
        for node_id in cell_combine_tree.expand_tree(sorting=False):
            for time_int in cell_combine_tree.get_node(node_id).data.get_time():
                tp_value_list = []
                tp_and_cell_index = f'{time_int:03}' + '::' + node_id

                if tp_and_cell_index not in values_dict.keys():  # need to do interpolation
                    print(tp_and_cell_index, tp_value_list)
                    if cell_combine_tree.get_node(node_id).is_leaf() and time_int == \
                            cell_combine_tree.get_node(node_id).data.get_time()[-1]:
                        pass
                    else:
                        for i in range(10):
                            complement_pre_tp_and_cell_index = f'{(time_int - i):03}' + '::' + node_id
                            complement_post_tp_and_cell_index = f'{(time_int + i):03}' + '::' + node_id
                            if complement_pre_tp_and_cell_index in values_dict.keys():
                                values_dict[tp_and_cell_index] = values_dict[complement_pre_tp_and_cell_index]
                                break
                            elif complement_post_tp_and_cell_index in values_dict.keys():
                                values_dict[tp_and_cell_index] = values_dict[complement_post_tp_and_cell_index]
                                break

        # https: // www.webucator.com / article / python - color - constants - module /
        # colors2 = ['red4', 'red3', 'red2', 'red1', 'orangered1', 'orange', 'darkolivegreen2','darkolivegreen1','darkolivegreen2', 'lightblue1',lightblue', 'dodgerblue1',
        #           'dodgerblue2', 'dodgerblue3', 'dodgerblue4']
        colors2 = np.array(
            [(139, 0, 0), (205, 0, 0), (238, 0, 0), (255, 0, 0), (255, 69, 0), (255, 128, 0),
             (188, 238, 104), (202, 255, 112), (188, 238, 104),
             (89, 210, 255), (63, 180, 255), (30, 144, 255), (28, 134, 238), (24, 116, 205), (16, 78, 139)]) / 255

        # cmap_list = ListedColormap(colors)
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors2)
        draw_cell_lineage_tree(cell_combine_tree, values_dict=values_dict,
                               embryo_name='combine_tree',
                               plot_title='average SPHARM\'s ' + column + 'th principle component', color_map=cmap1)


def draw_eigengrid_dynamic_mean_tree(print_num=2):
    # ==================drawing for PCA; multiple lineage tree pictures for one embryo========================
    pca_num = 12
    norm_shcpca_csv_path = cell_shape_analysis_data_path + r'my_data_csv/norm_2DMATRIX_PCA_csv'
    # ----------------read SHcPCA result first--------------------------------
    path_SHcPCA_lifespan_csv = os.path.join(norm_shcpca_csv_path,
                                            'Mean_cellLineageTree_dynamic_eigengrid.csv')
    df_pd_spharmpca_lifespan = read_csv_to_df(path_SHcPCA_lifespan_csv)

    # ----------------------------------------------------------------------------

    cell_combine_tree, begin_frame = data_struct.get_combined_lineage_tree()

    # frame = time / 1.39 +begin_frame
    for i in range(print_num):
        column = str(i+1)
        values_dict = {}
        for node_id in cell_combine_tree.expand_tree(sorting=False):
            for time_int in cell_combine_tree.get_node(node_id).data.get_time():
                tp_and_cell_index = f'{time_int:03}' + '::' + node_id
                values_dict[tp_and_cell_index] = df_pd_spharmpca_lifespan.at[node_id, column]

        colors2 = np.array(
            [(139, 0, 0), (205, 0, 0), (238, 0, 0), (255, 0, 0), (255, 69, 0), (255, 128, 0),
             (188, 238, 104), (202, 255, 112), (188, 238, 104),
             (89, 210, 255), (63, 180, 255), (30, 144, 255), (28, 134, 238), (24, 116, 205), (16, 78, 139)]) / 255

        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors2)
        draw_cell_lineage_tree(cell_combine_tree, values_dict=values_dict,
                               plot_title='Dynamic feature - eigengrid ' + column + ' average cell lineage tree',
                               color_map=cmap1, end_time_point=220)


def draw_shc_PCA_dynamic_mean_tree(print_num=2):
    # ==================drawing for PCA; multiple lineage tree pictures for one embryo========================
    pca_num = 12
    norm_shcpca_csv_path = cell_shape_analysis_data_path + r'my_data_csv/norm_SH_PCA_csv'
    # ----------------read SHcPCA result first--------------------------------
    path_SHcPCA_lifespan_csv = os.path.join(norm_shcpca_csv_path,
                                            'lifespan_avg_SHcPCA' + str(pca_num) + '_norm.csv')
    df_pd_spharmpca_lifespan = read_csv_to_df(path_SHcPCA_lifespan_csv)

    # ----------------------------------------------------------------------------

    cell_combine_tree, _ = data_struct.get_combined_lineage_tree()

    # frame = time / 1.39 +begin_frame
    for i in range(print_num):
        column = str(i)
        values_dict = {}
        for node_id in cell_combine_tree.expand_tree(sorting=False):
            for time_int in cell_combine_tree.get_node(node_id).data.get_time():
                tp_and_cell_index = f'{time_int:03}' + '::' + node_id
                values_dict[tp_and_cell_index] = df_pd_spharmpca_lifespan.at[node_id, column]

        colors2 = np.array(
            [(139, 0, 0), (205, 0, 0), (238, 0, 0), (255, 0, 0), (255, 69, 0), (255, 128, 0),
             (188, 238, 104), (202, 255, 112), (188, 238, 104),
             (89, 210, 255), (63, 180, 255), (30, 144, 255), (28, 134, 238), (24, 116, 205), (16, 78, 139)]) / 255

        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors2)
        draw_cell_lineage_tree(cell_combine_tree, values_dict=values_dict,
                               embryo_name='combine_tree',
                               plot_title='lifespan average SPHARM\'s ' + column + 'th principle component',
                               color_map=cmap1)


def draw_cell_fate_lineage_tree_01paper():
    cell_fate_map = {'Unspecified': 0, 'Other': 1, 'Death': 2, 'Neuron': 3, 'Intestin': 4, 'Muscle': 5, 'Pharynx': 6,
                     'Skin': 7, 'Germ Cell': 8}
    cell_combine_tree, begin_frame = data_struct.get_combined_lineage_tree()
    df_cell_fate = pd.read_excel(os.path.join(win_cell_shape_analysis_data_path, 'CellFate.xls'))
    cell_fate_dict = {}
    for idx in df_cell_fate.index:
        cell_fate_dict[df_cell_fate.at[idx, 'Name'].strip('\'')] = df_cell_fate.at[idx, 'Fate'].strip('\'')
    values_dict = {}
    for node_id in cell_combine_tree.expand_tree(sorting=False):
        for time_int in cell_combine_tree.get_node(node_id).data.get_time():
            tp_and_cell_index = f'{time_int:03}' + '::' + node_id
            values_dict[tp_and_cell_index] = cell_fate_map[cell_fate_dict[node_id]]

    colors = np.array(
        [(240, 230, 140), (0, 255, 255), (255, 0, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
         (30, 144, 255), (188, 143, 143)]) / 255

    cmap = ListedColormap(colors)
    draw_cell_lineage_tree(cell_combine_tree, values_dict=values_dict,
                           plot_title='Cell Fate Lineage Tree',end_time_point=67,
                           color_map=cmap, is_abs=False)


if __name__ == "__main__":
    draw_cell_fate_lineage_tree_01paper()
    # draw_static_each_embryo_cell_lineage_tree(max_frame=150)
    # draw_2Dmatrix_PCA_combined()
    # draw_PCA_combined(print_num=12)
    # draw_eigengrid_dynamic_mean_tree(print_num=1)
    # draw_tree_test()
    # data_struct.get_combined_lineage_tree()
