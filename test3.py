# import dependency library

import json
import os
from random import uniform

import igl
import pandas as pd
import scipy as sp
import numpy as np
import meshplot as mp

# import user defined library
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

import analysis.curvature as analysis_curvature
from lineage_stat.data_structure import get_combined_lineage_tree
from utils.cell_func import get_cell_name_affine_table
from utils.general_func import read_csv_to_df
from utils.sh_cooperation import get_flatten_ldegree_morder
from utils.shape_model import generate_alpha_shape

import static.config as config


def calculate_cell_curvature():
    # Sample20,ABplpapapa,150
    # Sample20,ABalaapa,078
    # Sample20,ABa,005
    # Sample20,MSp,035
    print('waiting type you input: sample name and time points for embryogenesis')
    embryo_name, cell_name, tp = str(input()).split(',')

    num_cell_name, cell_num = get_cell_name_affine_table()
    this_cell_keys = cell_num[cell_name]

    # -------getting all data points including dilation  points -- for generate alpha shape--------
    with open(os.path.join(config.data_path + r'cell_dia_points', embryo_name + '_' + tp + '_segCell.json')) as fp:
        cell_data = json.load(fp)
    cell_points_building_as = []
    # print(cell_data.keys())
    for item_str in cell_data[str(this_cell_keys)]:
        x, y, z = item_str.split('_')
        x, y, z = float(x) + uniform(0, 0.001), float(y) + uniform(0, 0.001), float(
            z) + uniform(0, 0.001)
        cell_points_building_as.append([x, y, z])
    cell_points_building_as = np.array(cell_points_building_as)
    # print(cell_points_building_as)
    m_mesh = generate_alpha_shape(cell_points_building_as, alpha_value=0.88)
    f = np.asarray(m_mesh.triangles)
    v = np.asarray(m_mesh.vertices)
    print('vertices number', v.shape, 'facet number', f.shape)
    k = igl.gaussian_curvature(v, f)

    mp.offline()
    import matplotlib as mpl

    cmap = mpl.cm.coolwarm

    # https: // stackoverflow.com / questions / 61585101 / create - corresponding - rgb - list - colormap - based - on - values - in -another - list

    edge_val = max(abs(k))
    modified_k = np.concatenate((k, np.array([-edge_val, edge_val])), axis=0)
    color_value_RGB = cmap(modified_k)
    [mean_x, mean_y, mean_z] = np.average(v, axis=0)
    v_plot = np.concatenate((v, np.array([[mean_x, mean_y, mean_z], [mean_x, mean_y, mean_z]])), axis=0)
    mp.plot(v_plot, f, color_value_RGB[:, :3])

    min_val, max_val = min(modified_k), max(modified_k)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    fig, ax = plt.subplots()
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Distance (least to greatest)')
    ax.tick_params(axis='x', rotation=90)

    plt.show()


def mean_embryo_2DGrid_SPHARM():
    # COMBINE TREE FEATURES
    embryo_names = [str(i).zfill(2) for i in range(4, 21)]

    # # -----------------------2d grid reconstruction of mean cells in 17 embryos------------------------
    # df_2DGrid_dict = {}
    # for embryo_name in embryo_names:
    #     path_2DGRID_csv = os.path.join(config.data_path, 'my_data_csv', 'SH_time_domain_csv',
    #                                    'Sample' + embryo_name + 'LabelUnified_l_25.csv')
    #     df_2DGrid_dict[embryo_name] = read_csv_to_df(path_2DGRID_csv)

    # -----------------------spharm reconstruction of mean cells in 17 embryos
    df_SPAHRM_dict = {}

    for embryo_name in embryo_names:
        path_SPHARM_csv = os.path.join(config.data_path, 'my_data_csv', 'SH_time_domain_csv',
                                       'Sample' + embryo_name + 'LabelUnified_l_25.csv')
        df_SPAHRM_dict[embryo_name] = read_csv_to_df(path_SPHARM_csv)

    cell_combine_tree, begin_frame = get_combined_lineage_tree()
    # ------static eigenharmonic weight in mean cell lineage tree-----------------------
    df_static_mean_embryo = pd.DataFrame(columns=get_flatten_ldegree_morder(25))
    for node_id in cell_combine_tree.expand_tree():
        for time_int in cell_combine_tree.get_node(node_id).data.get_time():
            # if time_int > 0:
            tp_value_list = []
            for embryo_name in df_SPAHRM_dict.keys():
                frame_int = int(time_int / 1.39 + begin_frame[embryo_name])
                frame_and_cell_index = f'{frame_int:03}' + '::' + node_id
                if frame_and_cell_index in df_SPAHRM_dict[embryo_name].index:
                    tp_value_list.append(df_SPAHRM_dict[embryo_name].loc[frame_and_cell_index])

            # we have already got all values at this time from all(17) embryos, we just need to draw its average
            tp_and_cell_index = f'{time_int:03}' + '::' + node_id
            if len(tp_value_list) == 0:  # need to do interpolation
                print('lost cell even in mean cell lineage tree', tp_and_cell_index, tp_value_list)
            else:
                df_static_mean_embryo.loc[tp_and_cell_index] = np.mean(np.array(tp_value_list), axis=0)
    print(df_static_mean_embryo)
    df_static_mean_embryo.to_csv(
        os.path.join(config.data_path, 'my_data_csv', 'SH_time_domain_csv', 'MeanSample_l_25.csv'))


if __name__ == "__main__":
    mean_embryo_2DGrid_SPHARM()
