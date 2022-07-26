# import dependency library

import json
import os
from random import uniform

import igl
import pandas as pd
import scipy as sp
import numpy as np
import meshplot as mp
import open3d as o3d

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


def calculate_cell_curvature(showCellMesh=True):
    # Sample04,ABpl,10
    # Sample20,ABplpapapa,150
    # Sample20,ABalaapa,078
    # Sample20,ABa,005
    # Sample20,MSp,035
    print('waiting type you input: sample, cell name and time points for embryogenesis')
    embryo_name, cell_name, tp = str(input()).split(',')

    num_cell_name, cell_num = get_cell_name_affine_table()
    this_cell_key = cell_num[cell_name]

    print('reading or showing', this_cell_key)
    cellMesh_file_saving_path = os.path.join(config.data_cell_mesh_and_contact, '3DMesh', embryo_name, tp, str(this_cell_key) + '.ply')
    m_mesh = o3d.io.read_triangle_mesh(cellMesh_file_saving_path).filter_smooth_taubin(number_of_iterations=100)
    if showCellMesh:
        o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True,
                                          mesh_show_wireframe=True)

    f = np.asarray(m_mesh.triangles)
    v = np.asarray(m_mesh.vertices)
    print('vertices number', v.shape, 'facet number', f.shape)
    k = igl.gaussian_curvature(v, f)
    print(k)
    print(np.unique(np.asarray(k*100).astype(int),return_counts=True))

    mp.offline()
    import matplotlib as mpl

    cmap = mpl.cm.coolwarm

    # https://stackoverflow.com/questions/61585101/create-corresponding-rgb-list-colormap-based-on-values-in-another-list

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
        path_SPHARM_csv = os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv', 'SH_time_domain_csv',
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
        os.path.join(config.cell_shape_analysis_data_path, 'my_data_csv', 'SH_time_domain_csv', 'MeanSample_l_25.csv'))


if __name__ == "__main__":
    calculate_cell_curvature()
