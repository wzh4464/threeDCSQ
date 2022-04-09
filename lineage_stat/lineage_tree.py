#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
This py file defines all structures and generate lineage tree that will be used in the shape analysis
'''

# import dependency library
import os

import matplotlib.pyplot as plt

from treelib import Tree
import numpy as np

# import user defined library

from static.config import data_path

def draw_life_span_tree(cell_tree: Tree, values_dict: dict, embryo_name='', plot_title='', color_map='seismic',
                        is_frame=False, time_resolution=1,is_abs=True):
    """

    :param cell_tree:
    :param values_dict:
    :param embryo_name:
    :return:
    """
    drawing_points_array = []
    # ABpl may appear 1 min later than ABal,we would set time 0 as ABa begin to split!
    # draw ABa, ABp, EMS, P1 only
    begin_frame = max(cell_tree.get_node('ABa').data.get_time()[-1], cell_tree.get_node('ABp').data.get_time()[-1])
    for node_id in cell_tree.expand_tree(sorting=False):
        this_cell_node = cell_tree.get_node(node_id)
        for queue_index, time_int in enumerate(this_cell_node.data.get_time()):
            tp_and_cell_index = f'{time_int:03}' + '::' + node_id
            # print(values_dict)
            # print(this_cell_node.data.get_position_x(), time_int, values_dict[tp_and_cell_index])
            if tp_and_cell_index in values_dict.keys():
                if is_frame:
                    drawing_points_array.append(
                        [this_cell_node.data.get_position_x(), -(time_int - begin_frame) * time_resolution,
                         values_dict[tp_and_cell_index]])
                else:
                    drawing_points_array.append(
                        [this_cell_node.data.get_position_x(), -time_int, values_dict[tp_and_cell_index]])

                if queue_index == 0:
                    mother_position_x = cell_tree.parent(node_id).data.get_position_x()
                    for x in np.arange(min(mother_position_x, this_cell_node.data.get_position_x()),
                                       max(mother_position_x, this_cell_node.data.get_position_x())):
                        if is_frame:
                            drawing_points_array.append(
                                [x, -(time_int - begin_frame) * time_resolution, values_dict[tp_and_cell_index]])
                        else:
                            drawing_points_array.append([x, -time_int, values_dict[tp_and_cell_index]])

    np_drawing_points_array = np.array(drawing_points_array)

    # make yellow to becom the colorbar center
    if is_abs:
        edge_value = np.nanmax(np.abs(np_drawing_points_array[:, 2]))
        # print(np.average(np.abs(np_drawing_points_array[:, 2])))
        # print(np.nanmax(np.abs(np_drawing_points_array[:, 2])))
        print('edge value', edge_value)
        drawing_points_array.append([0, 0, edge_value])
        drawing_points_array.append([0, 0, -edge_value])
    np_drawing_points_array = np.array(drawing_points_array)

    my_dpi = 100
    fig = plt.figure(figsize=(10000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    plt.scatter(x=np_drawing_points_array[:, 0], y=np_drawing_points_array[:, 1], marker='s', s=6,
                c=np_drawing_points_array[:, 2], cmap=color_map)
    # fig.colorbar(sc,ax=ax,)
    plt.axis('off')
    # plt.colorbar(sc,aspect=100,orientation="horizontal")
    plt.title(plot_title, fontsize=80)

    saving_path = os.path.join(data_path + r'lineage_tree/tree_plot', embryo_name)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    saving_path = os.path.join(saving_path, plot_title + '.pdf')
    print(saving_path)

    cbar=plt.colorbar(location='bottom')
    # ticklabs = cbar.ax.get_yticklabels()
    # cbar.ax.set_yticklabels(ticklabs,fontsize=40)

    # time axis !
    # plt.arrow(-8500,220, 0, -100, shape='full', lw=0, length_includes_head=True, head_width=5)
    plt.arrow(-8500, 20, 0, -230, width=20, shape='full', head_length=15)
    plt.text(-8600, -250, 'min', fontsize=60)
    plt.text(-8900, 0, '0', fontsize=60)
    plt.text(-8900, -50, '50', fontsize=60)
    plt.text(-8900, -100, '100', fontsize=60)
    plt.text(-8900, -150, '150', fontsize=60)
    plt.text(-8900, -200, '200', fontsize=60)

    # plt.show()
    plt.savefig(saving_path, format='pdf')

#
#
# # ===========================draw a for one embryo==no average, just one sample============================
# def draw_the_drawable_tree(root_node_id, embryo_time_tree, embryo_name, df_SHcPCA_target, is_drawing_cluster=False,
#                            column_index=-1, outlier_rate=2, embryo_num=None,
#                            color_map_selection='seismic', is_color_anchoring=False,
#                            plot_title=''):
#     """
#
#     :param root_node_id: the root node id of the lineage tree while we are drawing
#     :param embryo_time_tree: the complete embryo timing tree
#     :param embryo_name: as named, using to construct csv name
#     :param df_SHcPCA_target: the df from reading SHcPCA df
#     :param is_drawing_cluster: if we draw from cluster num, that's true; if from SHcPCA, we would draw multiple tree pic
#     :param column_index: if we draw from PCA, the PCA num we are drawing(sorted by explained variance)
#     :param color_map_selection: colormap for lineage tree pictures
#     :param plot_title: plot tittle(saving file name)
#     :return:
#     """
#     x_y_record_dict = {}
#     cell_division_left_or_right = {}
#
#     width = 12
#
#     drawing_points_array = []
#
#     cell_array_tag_tmp = {}
#
#     for tree_id_index, tree_node_id in enumerate(embryo_time_tree.expand_tree(nid=root_node_id, sorting=False)):
#
#         time_int = int(tree_node_id.split('_')[1])
#         if not embryo_num:
#             tp_and_cell_index = f'{time_int:03}' + '::' + tree_node_id.split('_')[0]
#         else:
#             tp_and_cell_index = f'{embryo_num:02}' + '::' + f'{time_int:03}' + '::' + tree_node_id.split('_')[0]
#         # print(tp_and_cell_index)
#         if tp_and_cell_index in df_SHcPCA_target.index:
#             # max_cluster_num = df_SHcPCA_max_cluster.at[tp_and_cell_index, 'cluster_num']
#             # print(tree_id_index, tree_node_id, tp_and_cell_index)
#             if is_drawing_cluster:
#                 if column_index == -1:
#                     target_numerical = df_SHcPCA_target.at[tp_and_cell_index, 'cluster_num']
#                 else:
#                     target_numerical = df_SHcPCA_target.at[tp_and_cell_index, str(column_index)]
#             else:
#                 target_numerical = df_SHcPCA_target.at[tp_and_cell_index, str(column_index)]
#
#         else:
#             # would delete these missing at lat drawing part
#             if is_drawing_cluster:
#                 target_numerical = -1
#             else:
#                 target_numerical = 0
#
#         if tree_node_id == root_node_id:
#             # the root of tree
#             # print(tree_node_id)
#             x_y_record_dict[tree_node_id] = (0, 0)
#
#             drawing_points_array.append([0, 0, target_numerical])
#             # print(tree_node_id, 0, 0)
#
#         else:
#             # cell generation number to help get x axis coordinate
#             cell_generation_num = embryo_time_tree.get_node(tree_node_id).data.get_generation()
#             parent_id = embryo_time_tree.get_node(tree_node_id).predecessor(embryo_time_tree.identifier)
#             parent_node_xy = x_y_record_dict[parent_id]
#             if len(embryo_time_tree.children(parent_id)) > 1:
#                 # if parent's children number is two, the cell divide
#
#                 if parent_id in cell_division_left_or_right.keys():
#                     # if paren_id in the judgement dictionary exits, mean that one of its child has added into the plot
#                     # list, so draw it in the right, so x add a offset
#                     # parent_node_xy[0] is parent's x
#                     x = parent_node_xy[0] + (2 ** (width - cell_generation_num))
#                     # parent_node_xy[1] is parent's y
#                     y = parent_node_xy[1] - 1
#                     x_y_record_dict[tree_node_id] = (x, y)
#                     # draw horizontal dividing line
#                     for x_division_tmp in np.arange(parent_node_xy[0] + 1, x + 1):
#                         drawing_points_array.append([x_division_tmp, y, target_numerical])
#                         # print(tree_node_id, x_division_tmp, y)
#
#                 else:
#                     cell_division_left_or_right[parent_id] = True
#                     x = parent_node_xy[0] - (2 ** (width - cell_generation_num))
#                     y = parent_node_xy[1] - 1
#                     x_y_record_dict[tree_node_id] = (x, y)
#                     # draw horizontal dividing line
#                     for x_division_tmp in np.arange(x, parent_node_xy[0]):
#                         drawing_points_array.append([x_division_tmp, y, target_numerical])
#                         # print(tree_node_id, x_division_tmp, y)
#             elif len(embryo_time_tree.children(parent_id)) == 1:
#                 x = parent_node_xy[0]
#                 y = parent_node_xy[1] - 1
#                 x_y_record_dict[tree_node_id] = (x, y)
#                 drawing_points_array.append([x, y, target_numerical])
#                 # print(tree_node_id, x, y)
#
#             # if tree_node_id.split('_')[0] not in cell_array_tag_tmp.keys() and len(tree_node_id.split('_')[0]) <= 6:
#             #     cell_array_tag_tmp[tree_node_id.split('_')[0]] = 1
#             #     print(tree_node_id, x)
#
#     # deal with the loss cell points, delete it
#     max_, min_ = drawing_points_array[0][2], drawing_points_array[0][2]
#
#     if not is_drawing_cluster and is_color_anchoring:
#         # in order to equalize the negative number and positive number, need to set max and min number while drawing
#         abs_maximum = 0
#         for item_tmp in drawing_points_array:
#             if item_tmp[2] == 0:
#                 drawing_points_array.remove(item_tmp)
#             elif item_tmp[2] > max_:
#                 max_ = item_tmp[2]
#             elif item_tmp[2] < min_:
#                 min_ = item_tmp[2]
#
#             if abs(item_tmp[2]) > abs_maximum:
#                 abs_maximum = abs(item_tmp[2])
#         drawing_points_array.append([-20, 0, -abs_maximum])
#         drawing_points_array.append([20, 0, abs_maximum])
#     elif not is_drawing_cluster and not is_color_anchoring:
#         for item_tmp in drawing_points_array:
#             if item_tmp[2] == 0:
#                 drawing_points_array.remove(item_tmp)
#             elif item_tmp[2] > max_:
#                 max_ = item_tmp[2]
#             elif item_tmp[2] < min_:
#                 min_ = item_tmp[2]
#     elif is_drawing_cluster:
#         for item_tmp in drawing_points_array:
#             if item_tmp[2] == -1:
#                 drawing_points_array.remove(item_tmp)
#             elif item_tmp[2] > max_:
#                 max_ = item_tmp[2]
#             elif item_tmp[2] < min_:
#                 min_ = item_tmp[2]
#
#     # print(embryo_time_tree.all_nodes()[:10])
#     np_drawing_points_array = np.array(drawing_points_array)
#     fig, ax = plt.subplots(figsize=(20, 10))
#
#     # delete the irregular points
#     print('the plot name:', plot_title)
#     print('its maximum and minimum is ', max_, min_)
#     percentile = np.percentile(np_drawing_points_array[:, 2], (25, 50, 75), interpolation='linear')
#     Q1 = percentile[0]  # 上四分位数
#     Q2 = percentile[1]
#     Q3 = percentile[2]  # 下四分位数
#     IQR = Q3 - Q1  # 四分位距
#     ulim = Q3 + outlier_rate * IQR  # 上限 非异常范围内的最大值
#     llim = Q1 - outlier_rate * IQR  # 下限 非异常范围内的最小值
#     print('its average, 25% 50% 75% is', np.average(np_drawing_points_array[:, 2]), Q1, Q2, Q3)
#     for item_tmp in drawing_points_array:
#         if item_tmp[2] > ulim or item_tmp[2] < llim:
#             drawing_points_array.remove(item_tmp)
#             # print(item_tmp)
#
#     np_drawing_points_array = np.array(drawing_points_array)
#
#     sc = ax.scatter(x=np_drawing_points_array[:, 0], y=np_drawing_points_array[:, 1], marker='s', s=4,
#                     c=np_drawing_points_array[:, 2],
#                     cmap=color_map_selection)
#     # fig.colorbar(sc,ax=ax,)
#     plt.axis('off')
#     # plt.colorbar(sc,aspect=100,orientation="horizontal")
#     plt.title(plot_title)
#
#     saving_path = os.path.join(r'../DATA/lineage_tree/tree_plot', embryo_name)
#     if not os.path.exists(saving_path):
#         os.mkdir(saving_path)
#     saving_path = os.path.join(saving_path, plot_title)
#
#     print(saving_path)
#     plt.savefig(saving_path)
#     # plt.show()

# =================================================================================================
