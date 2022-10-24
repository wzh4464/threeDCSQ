# ------system or others' pack--------------
import json
import os
import pickle
from copy import deepcopy
import nibabel as nib
import numpy as np
import pandas as pd
import open3d as o3d
from time import time
import multiprocessing as mp
import matplotlib.pyplot as plt

from scipy import ndimage
from tqdm import tqdm

# -----user's package----------
import static.config as my_config
from utils.cell_func import get_cell_name_affine_table
from utils.general_func import load_nitf2_img
from utils.shape_preprocess import export_dia_cell_points_json
from utils.shape_model import generate_alpha_shape, get_contact_surface_mesh


def detect_outer_cells():

    gui_data_path=r'D:\MembraneProjectData\GUIData\WebData_CMap_cell_label_v2'
    max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
                    '200117plc1pop1ip3']

    label_name_dict = pd.read_csv(os.path.join(gui_data_path,'name_dictionary.csv'), header=0, index_col=0).to_dict()[
        '0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}

    for idx, embryo_name in enumerate(embryo_names):
        df_volume_data = pd.read_csv(os.path.join(gui_data_path, embryo_name, embryo_name + '_volume.csv'),
                                     header=0, index_col=0)
        # df_volume_data.loc[:, :] = 0
        print(df_volume_data)

        for tp in range(1, max_times[idx] + 1):

            summary_tmp = {}
            path_tmp = os.path.join(r'D:\MembraneProjectData\GUIData\WebData_CMap_cell_label_v2', embryo_name,'SegCell')
            frame_this_embryo = str(tp).zfill(3)
            file_name = embryo_name + '_' + frame_this_embryo + '_segCell.nii.gz'
            volume = nib.load(os.path.join(path_tmp, file_name)).get_fdata().astype(int)
            for label_tmp in np.unique(volume)[1:]:
                summary_tmp[label_tmp] = 1
            volume_closing = ndimage.binary_closing((volume != 0), iterations=2)
            volume_outer = np.logical_xor(volume_closing, ndimage.binary_erosion(volume_closing))
            # print(np.unique(volume_outer,return_counts=True))
            outer_arr_tmp = np.where(volume_outer)
            # print()
            for index in range(outer_arr_tmp[0].shape[0]):
                x = outer_arr_tmp[0][index]
                y = outer_arr_tmp[1][index]
                z = outer_arr_tmp[2][index]
                label = volume[x, y, z]
                # print(x,y,z,label)
                if label != 0:
                    # if label in summary_tmp.keys():
                    summary_tmp[label] += 1
                    # else:
                    #     summary_tmp[label]=1
                else:
                    continue
            print(embryo_name, tp, summary_tmp)
            for tmp_key, tmp_value in summary_tmp.items():
                if tmp_value > 3:
                    # print(tp,label_name_dict[int(tmp_key)],tmp_value)
                    df_volume_data.loc[tp][label_name_dict[int(tmp_key)]] = 0  # outer is 0
                else:
                    df_volume_data.loc[tp][label_name_dict[int(tmp_key)]] = 1  # inner is 1

        print(df_volume_data)
        df_volume_data.to_csv(os.path.join(my_config.data_stat_tem_windows, embryo_name + '_outerCell.csv'))


def calculate_cell_surface_and_contact_points(config_arg, is_debug=False):
    embryo_name = config_arg['embryo_name']
    is_calculate_cell_mesh = config_arg.get('is_calculate_cell_mesh',None)
    is_calculate_contact_file = config_arg.get('is_calculate_contact_file',None)
    showCellMesh = config_arg['showCellMesh']
    showCellContact = config_arg.get('showCellContact',None)
    time_point = config_arg['time_point']
    path_embryo = config_arg.get('path_embryo', None)

    if not path_embryo:
        path_embryo = os.path.join(my_config.data_linux_CMAP_seg, embryo_name, 'SegCellTimeCombinedLabelUnified')
    else:
        print('calculating ', path_embryo, embryo_name, time_point, ' embryo stat')

    # ------------------------calculate surface points using dialation for each cell --------------------
    # for file_name in os.listdir(path_tmp):
    # if os.path.isfile(os.path.join(path_tmp, file_name)):
    frame_this_embryo = str(time_point).zfill(3)
    file_name = embryo_name + '_' + frame_this_embryo + '_segCell.nii.gz'

    # ------------if contact file exists, finish this embryo---------------
    # contact_saving_path = os.path.join(my_config.data_cell_mesh_and_contact, 'stat', embryo_name,
    #                                    embryo_name+'_'+file_name.split('.')[0] + '_contact.txt')
    # # ===============very important line===================
    # if os.path.exists(contact_saving_path):
    #     print(contact_saving_path,' existed')
    #     return 0
    # # =====================================================

    volume = nib.load(os.path.join(path_embryo, file_name)).get_fdata().astype(int).transpose([2, 1, 0])
    if is_debug:
        print(np.unique(volume, return_counts=True))
    # this_img = load_nitf2_img()

    # volume = this_img.get_fdata().astype(int)
    # -------------------
    cell_mask = volume != 0
    boundary_mask = (cell_mask == 0) & ndimage.binary_dilation(cell_mask)
    [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
    boundary_elements = []

    # find boundary between cells
    for (x, y, z) in zip(x_bound, y_bound, z_bound):
        neighbors = volume[np.ix_(range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2))]
        neighbor_labels = list(np.unique(neighbors))
        neighbor_labels.remove(0)
        if len(neighbor_labels) == 2:  # contact between two cells
            boundary_elements.append(neighbor_labels)
    # cell contact pairs
    cell_contact_pairs = list(np.unique(np.array(boundary_elements), axis=0))
    cell_conatact_pair_renew = []
    contact_points_dict = {}
    contact_area_dict = {}

    for (label1, label2) in cell_contact_pairs:
        contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1),
                                      ndimage.binary_dilation(volume == label2))
        contact_mask = np.logical_and(contact_mask, boundary_mask)
        if contact_mask.sum() > 2:

            cell_conatact_pair_renew.append((label1, label2))
            str_key = str(label1) + '_' + str(label2)
            contact_area_dict[str_key] = 0

            point_position_x, point_position_y, point_position_z = np.where(contact_mask == True)

            contact_points_list = []
            for i in range(len(point_position_x)):
                contact_points_list.append([point_position_x[i], point_position_y[i], point_position_z[i]])
            # print(str_key)
            contact_points_dict[str_key] = contact_points_list
            if is_debug:
                print('contact', str_key, len(contact_points_list))

    if is_debug:
        print('volume info', np.unique(volume, return_counts=True))
    cell_list = np.unique(volume)
    contact_mesh_dict = {}
    showing_record = []

    volume_dict = {}
    surface_dict = {}
    contact_dict = {}

    if is_debug:
        print('configuration', config_arg)
        print('cell list ', cell_list)

    # if not is_calculate_contact_file:
    #     print('loading ', contact_saving_path)
    #     with open(contact_saving_path,'rb') as handle:
    #         contact_mesh_dict = pickle.load(handle)

    # print(cell_list)
    weight_surface = 1.2031
    count_ratio_tmp = []
    for cell_key in cell_list:
        if cell_key != 0:
            cell_mask = np.logical_xor(ndimage.binary_dilation(volume == cell_key), (volume == cell_key))
            # if is_debug:
            #     print('-------',cell_key,'---------')
            #     print('surface num',(cell_mask==1).sum())
            #     print('inside sum',(volume == cell_key).sum())
            if (cell_mask == 1).sum() > 15000:
                volume_dict[cell_key] = (volume == cell_key).sum()
                surface_dict[cell_key] = cell_mask.sum() * weight_surface  # 1.2031... is derived by other papers
                irregularity = surface_dict[cell_key] ** (1 / 2) / volume_dict[cell_key] ** (1 / 3)
                if is_debug:
                    print('irregularity   ', irregularity)
                if irregularity < 2.199085:
                    print('impossible small surface', time_point, cell_key)
                for (cell1, cell2) in cell_conatact_pair_renew:
                    idx = str(cell1) + '_' + str(cell2)
                    # idx_test=
                    if cell_key not in (cell1, cell2) or idx in contact_dict.keys():
                        continue
                    # --------------------contact-----------------------------------------
                    contact_dict[idx] = len(contact_points_dict[idx]) * weight_surface
            else:
                if is_debug:
                    print(cell_key, 'surface point num', (cell_mask == 1).sum(), ' inside point num(dia)',
                          (ndimage.binary_dilation(volume == cell_key) == 1).sum())
                # ------------saving cell mesh---------------------

                tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_key) == 1)
                # print(len(tuple_tmp))
                sphere_list = np.concatenate(
                    (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
                adjusted_rate = 0.01
                sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, adjusted_rate,
                                                                                     (len(tuple_tmp[0]), 3))
                m_mesh = generate_alpha_shape(sphere_list_adjusted, displaying=showCellMesh)

                alpha_v = 1

                if not m_mesh.is_watertight():
                    for i in range(10):
                        alpha_v = alpha_v + i * 0.1
                        sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, adjusted_rate * (i + 1),
                                                                                             (len(tuple_tmp[0]), 3))
                        m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=alpha_v,
                                                      displaying=showCellMesh)
                        if is_debug:
                            print('watertight', m_mesh.is_watertight())
                            print(f"alpha={alpha_v:.3f}")
                            print('edge manifold boundary', m_mesh.is_edge_manifold(allow_boundary_edges=False))

                        if m_mesh.is_watertight():
                            break
                if is_debug:
                    print(cell_key, '=======mesh info=========', m_mesh)
                    print('edge manifold', m_mesh.is_edge_manifold(allow_boundary_edges=True))
                    print('edge manifold boundary', m_mesh.is_edge_manifold(allow_boundary_edges=False))
                    print('vertex manifold', m_mesh.is_vertex_manifold())
                    print('watertight', m_mesh.is_watertight())
                    print(f"alpha={alpha_v:.3f}")
                    print('volume====>', m_mesh.get_volume(), 'using  ', (volume == cell_key).sum())
                    print('surface area=======>', m_mesh.get_surface_area(), 'weighted ',
                          cell_mask.sum() * weight_surface)
                    # o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                    #                                   window_name=str(cell_key))
                # -----------------can not get watertight cell anyway-----------------------------------
                if not m_mesh.is_watertight():
                    print('no watertight mesh even after 10 times generation!!!', file_name, cell_key, 'use point sum')
                    volume_dict[cell_key] = (volume == cell_key).sum()
                    surface_dict[cell_key] = cell_mask.sum() * weight_surface  # 1.154... is derived by 2/sqrt(3)
                    irregularity = (surface_dict[cell_key] ** (1 / 2) / volume_dict[cell_key] ** (1 / 3))
                    if is_debug:
                        print(irregularity)
                    if irregularity < 2.199085:
                        print('impossible small surface!!!!!!!!!!!', time_point, cell_key)
                    for (cell1, cell2) in cell_conatact_pair_renew:
                        idx = str(cell1) + '_' + str(cell2)
                        # idx_test=
                        if cell_key not in (cell1, cell2) or idx in contact_dict.keys():
                            continue
                        # --------------------contact-----------------------------------------
                        contact_dict[idx] = len(contact_points_dict[idx]) * weight_surface
                    continue
                else:  # watertight mesh
                    m_mesh = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(np.asarray(m_mesh.vertices).astype(int)), m_mesh.triangles)
                    volume_dict[cell_key] = (volume == cell_key).sum()
                    surface_dict[cell_key] = m_mesh.get_surface_area()

                    if (surface_dict[cell_key] ** (1 / 2) / volume_dict[cell_key] ** (1 / 3)) < 2.199085:
                        print('impossible small surface', time_point, cell_key)

                    # if is_debug:
                    #     o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True,mesh_show_wireframe=True)
                    #     print()
                    # ============contact surface detection========================
                    # cell_vertices = np.asarray(m_mesh.vertices).astype(int)
                    cell_vertices = np.asarray(m_mesh.vertices)
                    # ====================contact file ============================

                    for (cell1, cell2) in cell_conatact_pair_renew:
                        idx = str(cell1) + '_' + str(cell2)
                        # idx_test=
                        if cell_key not in (cell1, cell2) or idx in contact_dict.keys():
                            continue
                        # --------------------contact-----------------------------------------
                        # print('calculating, saving or showing',path_tmp, file_name, idx, ' contact surface')

                        # build a mask to erase not contact points
                        # enumerate each points in contact surface
                        contact_mask_not = [True for i in range(len(cell_vertices))]
                        contact_vertices_loc_list = []
                        for [x, y, z] in contact_points_dict[idx]:
                            # print(x,y,z)
                            contact_vertices_loc = np.where(np.prod(cell_vertices == [x, y, z], axis=-1))
                            if len(contact_vertices_loc[0]) != 0:
                                contact_vertices_loc_list.append(contact_vertices_loc[0][0])
                                contact_mask_not[contact_vertices_loc[0][0]] = False
                        contact_mesh_dict[idx] = contact_vertices_loc_list
                        contact_mesh = deepcopy(m_mesh)
                        contact_mesh.remove_vertices_by_mask(contact_mask_not)
                        # eodo: finished: 1. check 179 embryo,2. contact sum compare with cell surface area
                        alpha_surface_area = contact_mesh.get_surface_area()
                        points_surface_area = len(contact_points_dict[idx]) * weight_surface
                        contact_dict[
                            idx] = alpha_surface_area if alpha_surface_area > points_surface_area else points_surface_area

                        if is_debug:
                            # contact_mesh = deepcopy(m_mesh)
                            # contact_mesh.remove_vertices_by_mask(contact_mask_not)
                            print(idx, '=======mesh info=========', contact_mesh)
                            print('edge manifold', contact_mesh.is_edge_manifold(allow_boundary_edges=True))
                            print('edge manifold boundary', contact_mesh.is_edge_manifold(allow_boundary_edges=False))
                            print('vertex manifold', contact_mesh.is_vertex_manifold())
                            # print('watertight', m_mesh.is_watertight())
                            print('contact surface area=======>', contact_mesh.get_surface_area(), 'while points num',
                                  points_surface_area)
                            # o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
                            #                                   mesh_show_wireframe=True)
                # -----------------CHECK IF THE SUM OF CONTACT SURFACE AREA-------------------------

                if is_debug:
                    this_cell_surface = surface_dict[cell_key]
                    contact_list_tmp = []

                    for (cell1, cell2) in cell_conatact_pair_renew:
                        idx_tmp = str(cell1) + '_' + str(cell2)
                        # idx_test=
                        if cell_key not in (cell1, cell2):
                            continue
                        contact_list_tmp.append(contact_dict[idx_tmp])
                    print('contact sum', sum(contact_list_tmp), '  surface area ', this_cell_surface)
                    ratio_this_cell = sum(contact_list_tmp) / this_cell_surface
                    if ratio_this_cell > 1:
                        count_ratio_tmp.append(ratio_this_cell)
                        # print('impossible things!!!!')

    if is_debug:
        abnormal_contact_cell_ratio = len(count_ratio_tmp) / (len(cell_list) - 1)
        print('volume dict ', volume_dict)
        print('surface dict', surface_dict)
        print('contact dict', contact_dict)
        print('-------------abnormal contact cell ratio!!!!====>  ', abnormal_contact_cell_ratio)
    # ------------saving volume surface and contact file for an embryo------------
    else:
        path_tmp = os.path.join(my_config.data_linux_cell_mesh_and_contact, 'stat', embryo_name)
        if not os.path.exists(path_tmp):
            os.mkdir(path_tmp)
        with open(os.path.join(path_tmp, file_name.split('.')[0] + '_volume.txt'), 'wb+') as handle:
            pickle.dump(volume_dict, handle, protocol=4)
        with open(os.path.join(path_tmp, file_name.split('.')[0] + '_surface.txt'), 'wb+') as handle:
            pickle.dump(surface_dict, handle, protocol=4)
        with open(os.path.join(path_tmp, file_name.split('.')[0] + '_contact.txt'), 'wb+') as handle:
            pickle.dump(contact_dict, handle, protocol=4)
    # else:

    #

    # path_tmp=os.path.join(my_config.data_cell_mesh_and_contact,'tem', embryo_name)
    # if not os.path.exists(path_tmp):
    #     os.mkdir(path_tmp)
    # with open(os.path.join(path_tmp,file_name.split('.')[0] + '_volume.json'),'wb+') as handle:
    #     json.dump(volume_dict, handle)
    # with open(os.path.join(path_tmp,file_name.split('.')[0] + '_surface.json'),'wb+') as handle:
    #     json.dump(surface_dict, handle)
    # with open(os.path.join(path_tmp,file_name.split('.')[0] + '_contact.json'),'wb+') as handle:
    #     json.dump(contact_dict, handle)

    # already get the contact pair and the contact points x y z
    # return cell_conatact_pair_renew, contact_points_dict
    # start calculate contact surface area

    # -------------------------------------------------------------------------------------------------------


# ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1', '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2', '200117plc1pop1ip3']

def calculate_cell_surface_and_contact_points_CMap(is_calculate_cell_mesh=True, is_calculate_contact_file=True,
                                                   showCellMesh=False,
                                                   showCellContact=False):
    """
    I believe is is_calculate_cell_mesh and is_calculate_contact_file should be the same ,
     i don't know what happen if they are not ( need to justify maybe)
    just saving cell mesh and contact surface record
    :param is_calculate_contact_file:
    :param showCellMesh:
    :param showCellContact:
    :return:
    """
    # max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    max_times = [195, 140, 155]

    # embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
    #                 '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
    #                 '200117plc1pop1ip3']
    embryo_names = ['200122plc1lag1ip2', '200117plc1pop1ip2', '200117plc1pop1ip3']

    # # --------TEST ONE EMBRYO-----------
    # config_tmp = {}
    # config_tmp["embryo_name"] = '191108plc1p1'
    # config_tmp['is_calculate_cell_mesh'] = is_calculate_cell_mesh
    # config_tmp['is_calculate_contact_file'] = is_calculate_contact_file
    # config_tmp['showCellMesh'] = showCellMesh
    # config_tmp['showCellContact'] = showCellContact
    # config_tmp['time_point'] = 179
    #
    # calculate_cell_surface_and_contact_points(config_tmp,is_debug=True)
    # # --------------------------------
    # input()

    for idx, embryo_name in enumerate(embryo_names):
        configs = []
        config_tmp = {}
        config_tmp["embryo_name"] = embryo_name
        config_tmp['is_calculate_cell_mesh'] = is_calculate_cell_mesh
        config_tmp['is_calculate_contact_file'] = is_calculate_contact_file
        config_tmp['showCellMesh'] = showCellMesh
        config_tmp['showCellContact'] = showCellContact
        for tp in tqdm(range(1, max_times[idx] + 1), desc="Compose configs"):
            config_tmp['time_point'] = tp
            configs.append(config_tmp.copy())

        mpPool = mp.Pool(30)
        # mpPool = mp.Pool(9)

        for idx_, _ in enumerate(
                tqdm(mpPool.imap_unordered(calculate_cell_surface_and_contact_points, configs), total=max_times[idx],
                     desc="calculating {} segmentations (contact graph)".format(embryo_name))):
            #
            pass


def calculate_cell_surface_and_contact_points_CShaper(is_calculate_cell_mesh=True, is_calculate_contact_file=True,
                                                      showCellMesh=False,
                                                      showCellContact=False):
    """
    I believe is is_calculate_cell_mesh and is_calculate_contact_file should be the same ,
     i don't know what happen if they are not ( need to justify maybe)
    just saving cell mesh and contact surface record
    :param is_calculate_contact_file:
    :param showCellMesh:
    :param showCellContact:
    :return:
    """
    embryo_names = ['Sample' + str(i).zfill(2) for i in range(4, 21)]
    max_times = [150, 170, 210, 165, 160, 160, 160, 170, 165, 150, 155, 170, 160, 160, 160, 160, 170]

    # # --------TEST ONE EMBRYO-----------
    # config_tmp = {}
    # config_tmp["embryo_name"] = 'Sample04'
    # config_tmp['is_calculate_cell_mesh'] = is_calculate_cell_mesh
    # config_tmp['is_calculate_contact_file'] = is_calculate_contact_file
    # config_tmp['showCellMesh'] = showCellMesh
    # config_tmp['showCellContact'] = showCellContact
    # config_tmp['time_point'] = 100
    # config_tmp['path_embryo'] = os.path.join(my_config.cell_shape_analysis_data_path,
    #                                              'Segmentation Results','SegmentedCell', 'Sample04LabelUnified')
    #
    # calculate_cell_surface_and_contact_points(config_tmp, is_debug=True)
    # # --------------------------------
    # input()

    for idx, embryo_name in enumerate(embryo_names):
        configs = []
        config_tmp = {}
        config_tmp["embryo_name"] = embryo_name
        config_tmp['is_calculate_cell_mesh'] = is_calculate_cell_mesh
        config_tmp['is_calculate_contact_file'] = is_calculate_contact_file
        config_tmp['showCellMesh'] = showCellMesh
        config_tmp['showCellContact'] = showCellContact
        config_tmp['path_embryo'] = os.path.join(my_config.cell_shape_analysis_data_path,
                                                 'Segmentation Results', 'UpdatedSegmentedCell', embryo_name)
        for tp in tqdm(range(1, max_times[idx] + 1), desc="Compose configs"):
            config_tmp['time_point'] = tp
            configs.append(config_tmp.copy())

        mpPool = mp.Pool(30)
        # mpPool = mp.Pool(4)

        for idx_, _ in enumerate(
                tqdm(mpPool.imap_unordered(calculate_cell_surface_and_contact_points, configs),
                     total=max_times[idx],
                     desc="calculating {} segmentations (contact graph)".format(embryo_name))):
            #
            pass
        # -------------------------------------------------------------------------------------------------------


def display_cell_mesh_contact_CMap(is_showing_cell_mesh=False, is_showing_cell_contact=True):
    # 200109plc1p1,ABalpppppap,181
    embryo_name, cell_name, tp = str(input()).split(',')
    path_tmp = os.path.join(my_config.data_linux_CMAP_seg, embryo_name, 'SegCell',
                            '{}_{}_segCell.nii.gz'.format(embryo_name, tp))
    this_img = load_nitf2_img(path_tmp)
    volume = this_img.get_fdata().astype(int)
    # print(np.unique(volume))

    _, name_label_dict = get_cell_name_affine_table(path=os.path.join(my_config.data_linux_CMAP_seg, 'name_dictionary.csv'))
    cell_idx = name_label_dict[cell_name]

    # -------------------
    cell_mask = volume != 0
    boundary_mask = (cell_mask == 0) & ndimage.binary_dilation(cell_mask)
    [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
    boundary_elements = []

    # find boundary between cells
    for (x, y, z) in zip(x_bound, y_bound, z_bound):
        neighbors = volume[np.ix_(range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2))]
        neighbor_labels = list(np.unique(neighbors))
        neighbor_labels.remove(0)
        if len(neighbor_labels) == 2:  # contact between two cells
            boundary_elements.append(neighbor_labels)
    # cell contact pairs
    cell_contact_pairs = list(np.unique(np.array(boundary_elements), axis=0))
    cell_conatact_pair_renew = []
    contact_points_dict = {}
    contact_area_dict = {}

    # print(cell_contact_pairs)

    for (label1, label2) in cell_contact_pairs:
        # print(cell_idx,label1,label2)
        if cell_idx in (label1, label2):
            print('calculating ', (label1, label2), ' contact ')
            contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1),
                                          ndimage.binary_dilation(volume == label2))
            contact_mask = np.logical_and(contact_mask, boundary_mask)
            if contact_mask.sum() > 2:

                cell_conatact_pair_renew.append((label1, label2))
                str_key = str(label1) + '_' + str(label2)
                contact_area_dict[str_key] = 0

                point_position_x, point_position_y, point_position_z = np.where(contact_mask == True)

                contact_points_list = []
                for i in range(len(point_position_x)):
                    contact_points_list.append([point_position_x[i], point_position_y[i], point_position_z[i]])
                # print(str_key)
                contact_points_dict[str_key] = contact_points_list

    contact_mesh_dict = {}
    contact_sur_area = []
    print((volume == cell_idx).sum())

    print('calculating and saving', cell_idx, ' surface')
    tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_idx) == 1)

    print(tuple_tmp)
    sphere_list = np.concatenate(
        (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
    sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, 0.001,
                                                                         (len(tuple_tmp[0]), 3))
    # print(np.unique(volume == cell_idx),tuple_tmp,sphere_list_adjusted)
    m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=1, displaying=is_showing_cell_mesh)
    print(len(m_mesh.cluster_connected_triangles()[2]))

    # print('saving mesh')
    # ============contact surface detection========================
    cell_vertices = np.asarray(m_mesh.vertices).astype(int)
    for (cell1, cell2) in cell_conatact_pair_renew:
        idx = str(cell1) + '_' + str(cell2)
        if cell_idx not in (cell1, cell2):
            continue

        # --------------------contact-----------------------------------------

        # build a mask to erase not contact points
        # enumerate each points in contact surface
        contact_mask_not = [True for i in range(len(cell_vertices))]
        contact_vertices_loc_list = []
        for [x, y, z] in contact_points_dict[idx]:
            # print(x,y,z)
            contact_vertices_loc = np.where(np.prod(cell_vertices == [x, y, z], axis=-1))
            if len(contact_vertices_loc[0]) != 0:
                contact_vertices_loc_list.append(contact_vertices_loc[0][0])
                contact_mask_not[contact_vertices_loc[0][0]] = False
        contact_mesh_dict[idx] = contact_vertices_loc_list
        contact_mesh = deepcopy(m_mesh)
        contact_mesh.remove_vertices_by_mask(contact_mask_not)
        contact_sur_area.append(contact_mesh.get_surface_area())

        if is_showing_cell_contact:
            o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
                                              mesh_show_wireframe=True)
    print('cell volume', (volume == cell_idx).sum() * 0.25 ** 3, ' mesh caculation===>', m_mesh.get_volume())
    print('cell surface area', m_mesh.get_surface_area() * 0.25 ** 2)
    print('sum of cell contact area', sum(contact_sur_area) * 0.25 ** 2)
    print('list of cell contact area', contact_sur_area)

    # small ratio----
    # 200109plc1p1,ABalpppppap,181   (normal part of it no contact)
    # 200117plc1pop1ip2,Dap,122     (segmenation as two cell)

    # wrong dividing cells
    # 200322plc1p2,Caaaa,181


def display_cell_mesh_contact_CShaper(is_show_original_points=False,is_showing_cell_mesh=False, is_showing_cell_contact=True):
    # Sample05,ABpl,014
    embryo_name, cell_name, tp = str(input()).split(',')
    path_tmp = os.path.join(my_config.data_win_CShaper_seg, 'SegmentedCell', embryo_name + 'LabelUnified',
                            '{}_{}_segCell.nii.gz'.format(embryo_name, tp))
    this_img = load_nitf2_img(path_tmp)
    volume = this_img.get_fdata().astype(int)
    # print(np.unique(volume))

    label_name_dict, name_label_dict = get_cell_name_affine_table(path=os.path.join(my_config.data_win_CShaper_seg, 'name_dictionary.csv'))
    cell_idx = name_label_dict[cell_name]

    # -------------------
    cell_mask = volume != 0
    boundary_mask = (cell_mask == 0) & ndimage.binary_dilation(cell_mask)
    [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
    boundary_elements = []

    # find boundary between cells
    for (x, y, z) in zip(x_bound, y_bound, z_bound):
        neighbors = volume[np.ix_(range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2))]
        neighbor_labels = list(np.unique(neighbors))
        neighbor_labels.remove(0)
        if len(neighbor_labels) == 2:  # contact between two cells
            boundary_elements.append(neighbor_labels)
    # cell contact pairs
    cell_contact_pairs = list(np.unique(np.array(boundary_elements), axis=0))
    cell_conatact_pair_renew = []
    contact_points_dict = {}
    contact_area_dict = {}

    # print(cell_contact_pairs)

    for (label1, label2) in cell_contact_pairs:
        # print(cell_idx,label1,label2)
        if cell_idx in (label1, label2):
            print('calculating ', (label1, label2), ' contact ')
            contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1),
                                          ndimage.binary_dilation(volume == label2))
            contact_mask = np.logical_and(contact_mask, boundary_mask)
            if contact_mask.sum() > 2:

                cell_conatact_pair_renew.append((label1, label2))
                str_key = str(label1) + '_' + str(label2)
                contact_area_dict[str_key] = 0

                point_position_x, point_position_y, point_position_z = np.where(contact_mask == True)

                contact_points_list = []
                for i in range(len(point_position_x)):
                    contact_points_list.append([point_position_x[i], point_position_y[i], point_position_z[i]])
                # print(str_key)
                contact_points_dict[str_key] = contact_points_list

    contact_mesh_dict = {}
    contact_sur_area = []
    print((volume == cell_idx).sum())

    print('calculating and saving', cell_idx, ' surface')
    tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_idx) == 1)

    if is_show_original_points:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None],s=1)
        plt.show()

    print(tuple_tmp)
    sphere_list = np.concatenate(
        (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
    sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, 0.001,
                                                                         (len(tuple_tmp[0]), 3))
    # print(np.unique(volume == cell_idx),tuple_tmp,sphere_list_adjusted)
    m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=1, displaying=is_showing_cell_mesh)
    print(len(m_mesh.cluster_connected_triangles()[2]))

    # print('saving mesh')
    # ============contact surface detection========================
    cell_vertices = np.asarray(m_mesh.vertices).astype(int)
    for (cell1, cell2) in cell_conatact_pair_renew:
        idx = str(cell1) + '_' + str(cell2)
        if cell_idx not in (cell1, cell2):
            continue

        # --------------------contact-----------------------------------------

        # build a mask to erase not contact points
        # enumerate each points in contact surface
        contact_mask_not = [True for i in range(len(cell_vertices))]
        contact_vertices_loc_list = []
        for [x, y, z] in contact_points_dict[idx]:
            # print(x,y,z)
            contact_vertices_loc = np.where(np.prod(cell_vertices == [x, y, z], axis=-1))
            if len(contact_vertices_loc[0]) != 0:
                contact_vertices_loc_list.append(contact_vertices_loc[0][0])
                contact_mask_not[contact_vertices_loc[0][0]] = False
        contact_mesh_dict[idx] = contact_vertices_loc_list
        contact_mesh = deepcopy(m_mesh)
        contact_mesh.remove_vertices_by_mask(contact_mask_not)
        contact_sur_area.append(contact_mesh.get_surface_area())

        if is_showing_cell_contact:
            print(label_name_dict[cell1], label_name_dict[cell2], ' contact pairt')
            o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
                                              mesh_show_wireframe=True)
    print('cell volume', (volume == cell_idx).sum() * 0.25 ** 3, ' mesh caculation===>', m_mesh.get_volume())
    print('cell surface area', m_mesh.get_surface_area() * 0.25 ** 2)
    print('sum of cell contact area', sum(contact_sur_area) * 0.25 ** 2)
    print('list of cell contact area', contact_sur_area)



if __name__ == "__main__":
    display_cell_mesh_contact_CMap(is_showing_cell_contact=True, is_showing_cell_mesh=True)
    # calculate_cell_surface_and_contact_points_CShaper()
    # detect_outer_cells()
    # calculate_cell_surface_and_contact_points(is_calculate_cell_mesh=False, is_calculate_contact_file=False,
    #                                           showCellMesh=True, showCellContact=True)
    # display_cell_mesh_contact_CMap(is_showing_cell_mesh=False,is_showing_cell_contact=False)
    # calculate_cell_surface_and_contact_points_CMap()
    # display_cell_mesh_contact_CMap()

    # calculate_cell_surface_and_contact_points_CMap()
