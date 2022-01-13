# import dependency library

import json
import os
from random import uniform

import igl
import scipy as sp
import numpy as np
import meshplot as mp

# import user defined library
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

import analysis.curvature as analysis_curvature
from utils.cell_func import get_cell_name_affine_table
from utils.shape_model import generate_alpha_shape

data_path = r'D:/cell_shape_quantification/DATA/'


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
    with open(os.path.join(data_path + r'cell_dia_points', embryo_name + '_' + tp + '_segCell.json')) as fp:
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
    m_mesh = generate_alpha_shape(cell_points_building_as,alpha_value=0.88)
    f = np.asarray(m_mesh.triangles)
    v = np.asarray(m_mesh.vertices)
    print('vertices number', v.shape, 'facet number', f.shape)
    k = igl.gaussian_curvature(v, f)

    mp.offline()
    import matplotlib as mpl

    cmap = mpl.cm.coolwarm

    #https: // stackoverflow.com / questions / 61585101 / create - corresponding - rgb - list - colormap - based - on - values - in -another - list

    edge_val=max(abs(k))
    modified_k=np.concatenate((k, np.array([-edge_val,edge_val])), axis=0)
    color_value_RGB = cmap(modified_k)
    [mean_x,mean_y,mean_z]=np.average(v,axis=0)
    v_plot=np.concatenate((v, np.array([[mean_x,mean_y,mean_z],[mean_x,mean_y,mean_z]])), axis=0)
    mp.plot(v_plot, f, color_value_RGB[:, :3])

    min_val, max_val = min(modified_k), max(modified_k)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    fig, ax = plt.subplots()
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Distance (least to greatest)')
    ax.tick_params(axis='x', rotation=90)

    plt.show()


if __name__ == "__main__":
    calculate_cell_curvature()
