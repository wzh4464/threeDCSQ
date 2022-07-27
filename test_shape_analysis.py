import json
import os
import pickle
from copy import deepcopy
from random import uniform

import nibabel as nib
import numpy as np
import pandas as pd
import open3d as o3d
from time import time
import multiprocessing as mp
from tqdm import tqdm


from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area
from tqdm import tqdm

import static.config as my_config
from utils.cell_func import get_cell_name_affine_table
from utils.general_func import load_nitf2_img
from utils.shape_preprocess import get_contact_area, export_dia_cell_points_json, export_dia_cell_surface_points_json
from utils.shape_model import generate_alpha_shape, get_contact_surface_mesh


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
    # [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    max_times = [205, 205, 255, 195, 195, 185]
    # ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
    #                     '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
    #                     '200117plc1pop1ip3']
    embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1']

    for idx, embryo_name in enumerate(embryo_names):
        configs = []
        config_tmp={}
        config_tmp["embryo_name"] = embryo_name
        config_tmp['is_calculate_cell_mesh']=is_calculate_cell_mesh
        config_tmp['is_calculate_contact_file']=is_calculate_contact_file
        config_tmp['showCellMesh']=showCellMesh
        config_tmp['showCellContact']=showCellContact
        for tp in tqdm(range(1, max_times[idx] + 1), desc="Compose {} configs".format(embryo_name)):
            config_tmp['time_point'] = tp
            configs.append(config_tmp.copy())

        mpPool = mp.Pool(4)
        for idx_, _ in enumerate(
            tqdm(mpPool.imap_unordered(calculate_cell_surface_and_contact_points,configs), total=max_times[idx],
                     desc="calculating {} segmentations (contact graph)".format(embryo_name))):
            #
            pass


def calculate_cell_surface_and_contact_points(config_arg):
    embryo_name=config_arg['embryo_name']
    is_calculate_cell_mesh=config_arg['is_calculate_cell_mesh']
    is_calculate_contact_file=config_arg['is_calculate_contact_file']
    showCellMesh=config_arg['showCellMesh']
    showCellContact=config_arg['showCellContact']
    time_point=config_arg['time_point']

    path_tmp = os.path.join(my_config.data_CMAP_seg, embryo_name, 'SegCell')

    # ------------------------calculate surface points using dialation for each cell --------------------
    # for file_name in os.listdir(path_tmp):
        # if os.path.isfile(os.path.join(path_tmp, file_name)):
    frame_this_embryo = str(time_point).zfill(3)
    file_name=embryo_name+'_'+frame_this_embryo+'_segCell.nii.gz'

    volume = nib.load(os.path.join(path_tmp, file_name)).get_fdata().astype(int).transpose([2, 1, 0])
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

    cell_list = np.unique(volume)
    contact_mesh_dict = {}
    showing_record = []
    if not is_calculate_contact_file:
        print('loading ', my_config.data_cell_mesh_and_contact, 'contactSurface', embryo_name,
              file_name.split('.')[0] + '.pickle')
        with open(os.path.join(my_config.data_cell_mesh_and_contact, 'contactSurface',
                               embryo_name, file_name.split('.')[0] + '.pickle'),
                  'rb') as handle:
            contact_mesh_dict = pickle.load(handle)
    # print(cell_list)
    for cell_key in cell_list:
        if cell_key != 0:

            # ------------saving cell mesh---------------------
            cellMesh_saving_path = os.path.join(my_config.data_cell_mesh_and_contact, '3DMesh',
                                                embryo_name, frame_this_embryo)
            if not os.path.exists(cellMesh_saving_path):
                os.makedirs(cellMesh_saving_path)
            cellMesh_file_saving_path = os.path.join(cellMesh_saving_path, str(cell_key) + '.ply')

            # print(os.path.exists(cellMesh_file_saving_path))
            if not os.path.exists(cellMesh_file_saving_path) or is_calculate_cell_mesh:
                # print('calculating and saving', path_tmp, file_name,cell_key, ' surface')
                tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_key) == 1)
                sphere_list = np.concatenate(
                    (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
                sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, 0.001,
                                                                                     (len(tuple_tmp[0]), 3))
                m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=1, displaying=showCellMesh)
                # print('saving mesh')
                o3d.io.write_triangle_mesh(cellMesh_file_saving_path, m_mesh)
                # is_contact_file = True
            else:
                print('reading or showing', cell_key)
                m_mesh = o3d.io.read_triangle_mesh(cellMesh_file_saving_path)
                if showCellMesh:
                    o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True,
                                                      mesh_show_wireframe=True)
            # ============contact surface detection========================
            cell_vertices = np.asarray(m_mesh.vertices).astype(int)
            # ====================contact file ============================
            if is_calculate_contact_file:
                # ---------------saving contact file-----------------------------------

                for (cell1, cell2) in cell_conatact_pair_renew:
                    idx = str(cell1) + '_' + str(cell2)
                    if cell_key not in (cell1, cell2) or idx in contact_mesh_dict.keys():
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

                    if showCellContact:
                        contact_mesh = deepcopy(m_mesh)
                        contact_mesh.remove_vertices_by_mask(contact_mask_not)
                        o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
                                                          mesh_show_wireframe=True)


            else:

                for (cell1, cell2) in cell_conatact_pair_renew:
                    contact_mask_not = [True for i in range(len(cell_vertices))]
                    idx = str(cell1) + '_' + str(cell2)
                    if cell_key not in (cell1, cell2) or idx in showing_record:
                        continue
                    print('reading or showing', idx, ' contact surface')
                    print(showing_record)
                    showing_record.append(idx)
                    for value_ in contact_mesh_dict[idx]:
                        contact_mask_not[value_] = False

                    if showCellContact:
                        contact_mesh = deepcopy(m_mesh)
                        contact_mesh.remove_vertices_by_mask(contact_mask_not)
                        o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
                                                          mesh_show_wireframe=True)

    # ------------saving contact file for an embryo------------
    if is_calculate_contact_file:
        contact_saving_path = os.path.join(my_config.data_cell_mesh_and_contact, 'contactSurface', embryo_name)
        if not os.path.exists(contact_saving_path):
            os.mkdir(contact_saving_path)
        with open(os.path.join(contact_saving_path, file_name.split('.')[0] + '.pickle'),
                  'wb+') as handle:
            pickle.dump(contact_mesh_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # already get the contact pair and the contact points x y z
    # return cell_conatact_pair_renew, contact_points_dict
    # start calculate contact surface area

    # -------------------------------------------------------------------------------------------------------


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
    embryo_names = [str(i).zfill(2) for i in range(4, 21)]

    for embryo_name in embryo_names:
        # ------------------------calculate surface points using dialation for each cell --------------------
        path_tmp = my_config.cell_shape_analysis_data_path + r'Segmentation Results\SegmentedCell/Sample' + embryo_name + 'LabelUnified'
        for file_name in os.listdir(path_tmp):
            if os.path.isfile(os.path.join(path_tmp, file_name)):
                print(path_tmp, file_name)
                frame_this_embryo = file_name.split('_')[1]

                this_img = load_nitf2_img(os.path.join(path_tmp, file_name))

                volume = this_img.get_fdata().astype(int)
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

                cell_list = np.unique(volume)
                contact_mesh_dict = {}
                showing_record = []
                if not is_calculate_contact_file:
                    print('loading ', my_config.data_cell_mesh_and_contact, 'contactSurface', 'Sample' + embryo_name,
                          file_name.split('.')[0] + '.pickle')
                    with open(os.path.join(my_config.data_cell_mesh_and_contact, 'contactSurface',
                                           'Sample' + embryo_name, file_name.split('.')[0] + '.pickle'),
                              'rb') as handle:
                        contact_mesh_dict = pickle.load(handle)
                # print(cell_list)
                for cell_key in tqdm(cell_list,desc="Collecting cell mesh and contact of cshaper embryo{}".format(embryo_name)):
                    if cell_key != 0:

                        # ------------saving cell mesh---------------------
                        cellMesh_saving_path = os.path.join(my_config.data_cell_mesh_and_contact, '3DMesh',
                                                            'Sample' + embryo_name, frame_this_embryo)
                        if not os.path.exists(cellMesh_saving_path):
                            os.makedirs(cellMesh_saving_path)
                        cellMesh_file_saving_path = os.path.join(cellMesh_saving_path, str(cell_key) + '.ply')

                        # print(os.path.exists(cellMesh_file_saving_path))
                        if not os.path.exists(cellMesh_file_saving_path) or is_calculate_cell_mesh:
                            print('calculating and saving', cell_key, ' surface')
                            tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_key) == 1)
                            sphere_list = np.concatenate(
                                (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
                            sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, 0.001,
                                                                                                 (len(tuple_tmp[0]), 3))
                            m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=1, displaying=showCellMesh)
                            # print('saving mesh')
                            o3d.io.write_triangle_mesh(cellMesh_file_saving_path, m_mesh)
                            # is_contact_file = True
                        else:
                            print('reading or showing',cell_key)
                            m_mesh = o3d.io.read_triangle_mesh(cellMesh_file_saving_path)
                            if showCellMesh:
                                o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True,
                                                                  mesh_show_wireframe=True)
                        # ============contact surface detection========================
                        cell_vertices = np.asarray(m_mesh.vertices).astype(int)
                        # ====================contact file ============================
                        if is_calculate_contact_file:
                            # ---------------saving contact file-----------------------------------

                            for (cell1, cell2) in cell_conatact_pair_renew:
                                idx = str(cell1) + '_' + str(cell2)
                                if cell_key not in (cell1, cell2) or idx in contact_mesh_dict.keys():
                                    continue

                                # --------------------contact-----------------------------------------
                                print('calculating, saving or showing', idx, ' contact surface')

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

                                if showCellContact:
                                    contact_mesh = deepcopy(m_mesh)
                                    contact_mesh.remove_vertices_by_mask(contact_mask_not)
                                    o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
                                                                      mesh_show_wireframe=True)
                        else:

                            for (cell1, cell2) in cell_conatact_pair_renew:
                                contact_mask_not = [True for i in range(len(cell_vertices))]
                                idx = str(cell1) + '_' + str(cell2)
                                if cell_key not in (cell1, cell2) or idx in showing_record:
                                    continue
                                print('reading or showing', idx, ' contact surface')
                                print(showing_record)
                                showing_record.append(idx)
                                for value_ in contact_mesh_dict[idx]:
                                    contact_mask_not[value_] = False

                                if showCellContact:
                                    contact_mesh = deepcopy(m_mesh)
                                    contact_mesh.remove_vertices_by_mask(contact_mask_not)
                                    o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
                                                                      mesh_show_wireframe=True)

                # ------------saving contact file for an embryo------------
                if is_calculate_contact_file:
                    contact_saving_path = os.path.join(my_config.data_cell_mesh_and_contact, 'contactSurface',
                                                       'Sample' + embryo_name)
                    if not os.path.exists(contact_saving_path):
                        os.mkdir(contact_saving_path)
                    with open(os.path.join(contact_saving_path, file_name.split('.')[0] + '.pickle'),
                              'wb+') as handle:
                        pickle.dump(contact_mesh_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # already get the contact pair and the contact points x y z
                # return cell_conatact_pair_renew, contact_points_dict
                # start calculate contact surface area

        # -------------------------------------------------------------------------------------------------------


def display_cell_mesh_contact_CMap(is_showing_cell_mesh=False, is_showing_cell_contact=True):
    # 200109plc1p1,ABalpppppap,181
    embryo_name, cell_name, tp = str(input()).split(',')
    path_tmp = os.path.join(my_config.data_CMAP_seg, embryo_name, 'SegCell',
                            '{}_{}_segCell.nii.gz'.format(embryo_name, tp))
    this_img = load_nitf2_img(path_tmp)
    volume = this_img.get_fdata().astype(int)
    # print(np.unique(volume))

    _, name_label_dict = get_cell_name_affine_table(path=my_config.data_CMAP_seg + r'name_dictionary.csv')
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
            print('calculating ',(label1, label2),' contact ')
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

    print('calculating and saving', cell_idx, ' surface')
    tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_idx) == 1)
    print(tuple_tmp)
    sphere_list = np.concatenate(
        (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
    sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, 0.001,
                                                                         (len(tuple_tmp[0]), 3))
    # print(np.unique(volume == cell_idx),tuple_tmp,sphere_list_adjusted)
    m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=1, displaying=is_showing_cell_mesh)
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
    print('cell volume', (volume == cell_idx).sum()*0.25**3,' mesh caculation===>',m_mesh.get_volume())
    print('cell surface area', m_mesh.get_surface_area()*0.25**2)
    print('sum of cell contact area', sum(contact_sur_area)*0.25**2)
    print('list of cell contact area', contact_sur_area)

    # small ratio----
    # 200109plc1p1,ABalpppppap,181   (normal part of it no contact)
    # 200117plc1pop1ip2,Dap,122     (segmenation as two cell)


if __name__ == "__main__":
    # calculate_cell_surface_and_contact_points(is_calculate_cell_mesh=False, is_calculate_contact_file=False,
    #                                           showCellMesh=True, showCellContact=True)
    display_cell_mesh_contact_CMap(is_showing_cell_mesh=False,is_showing_cell_contact=False)
    # calculate_cell_surface_and_contact_points_CMap()
