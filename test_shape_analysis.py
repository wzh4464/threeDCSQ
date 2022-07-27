import json
import os
import pickle
from copy import deepcopy
import pandas as pd
import nibabel as nib
import numpy as np
import open3d as o3d
from time import time
import multiprocessing as mp

from scipy import ndimage
# from skimage.measure import marching_cubes, mesh_surface_area
from tqdm import tqdm

import static.config as my_config
from utils.cell_func import get_cell_name_affine_table
from utils.general_func import load_nitf2_img
from utils.shape_preprocess import export_dia_cell_points_json
from utils.shape_model import generate_alpha_shape, get_contact_surface_mesh

def detect_outer_cells():

    max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
                    '200117plc1pop1ip3']


    label_name_dict=pd.read_csv(os.path.join(my_config.data_label_name_dictionary), header=0,index_col=0).to_dict()['0']
    name_label_dict={label:name for name, label in label_name_dict.items()}

    for idx,embryo_name in enumerate(embryo_names):
        df_volume_data=pd.read_csv(os.path.join(my_config.data_stat,embryo_name,embryo_name+'_volume.csv'),header=0,index_col=0)
        # df_volume_data.loc[:, :] = 0
        print(df_volume_data)


        for tp in range(1,max_times[idx]+1):


            summary_tmp={}
            path_tmp = os.path.join(my_config.data_CMAP_seg, embryo_name, 'SegCellTimeCombinedLabelUnified')
            frame_this_embryo = str(tp).zfill(3)
            file_name = embryo_name + '_' + frame_this_embryo + '_segCell.nii.gz'
            volume = nib.load(os.path.join(path_tmp, file_name)).get_fdata().astype(int).transpose([2, 1, 0])
            for label_tmp in np.unique(volume)[1:]:
                summary_tmp[label_tmp]=1
            volume_closing=ndimage.binary_closing((volume!=0),iterations=5)
            volume_outer=np.logical_xor(volume_closing,ndimage.binary_erosion(volume_closing))
            # print(np.unique(volume_outer,return_counts=True))
            outer_arr_tmp=np.where(volume_outer)
            # print()
            for index in range(outer_arr_tmp[0].shape[0]):
                x=outer_arr_tmp[0][index]
                y = outer_arr_tmp[1][index]
                z = outer_arr_tmp[2][index]
                label=volume[x,y,z]
                # print(x,y,z,label)
                if label!=0:
                    # if label in summary_tmp.keys():
                    summary_tmp[label]+=1
                    # else:
                    #     summary_tmp[label]=1
                else:
                    continue
            print(embryo_name,tp,summary_tmp)
            for tmp_key,tmp_value in summary_tmp.items():
                if tmp_value>3:
                    # print(tp,label_name_dict[int(tmp_key)],tmp_value)
                    df_volume_data.loc[tp][label_name_dict[int(tmp_key)]]=0 # outer is 0
                else:
                    df_volume_data.loc[tp][label_name_dict[int(tmp_key)]]=1 # inner is 1


        print(df_volume_data)
        df_volume_data.to_csv(os.path.join(my_config.data_stat_tem,embryo_name+'_outerCell.csv'))



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
    embryo_names = ['200122plc1lag1ip2', '200117plc1pop1ip2','200117plc1pop1ip3']

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
        config_tmp={}
        config_tmp["embryo_name"] = embryo_name
        config_tmp['is_calculate_cell_mesh']=is_calculate_cell_mesh
        config_tmp['is_calculate_contact_file']=is_calculate_contact_file
        config_tmp['showCellMesh']=showCellMesh
        config_tmp['showCellContact']=showCellContact
        for tp in tqdm(range(1, max_times[idx] + 1), desc="Compose configs"):
            config_tmp['time_point'] = tp
            configs.append(config_tmp.copy())

        mpPool = mp.Pool(30)
        # mpPool = mp.Pool(9)

        for idx_, _ in enumerate(
            tqdm(mpPool.imap_unordered(calculate_cell_surface_and_contact_points,configs), total=max_times[idx],
                     desc="Naming {} segmentations (contact graph)".format(embryo_name))):
            #
            pass


def calculate_cell_surface_and_contact_points(config_arg, is_debug=False):
    embryo_name=config_arg['embryo_name']
    is_calculate_cell_mesh=config_arg['is_calculate_cell_mesh']
    is_calculate_contact_file=config_arg['is_calculate_contact_file']
    showCellMesh=config_arg['showCellMesh']
    showCellContact=config_arg['showCellContact']
    time_point=config_arg['time_point']

    path_tmp = os.path.join(my_config.data_CMAP_seg, embryo_name, 'SegCellTimeCombinedLabelUnified')

    # ------------------------calculate surface points using dialation for each cell --------------------
    # for file_name in os.listdir(path_tmp):
        # if os.path.isfile(os.path.join(path_tmp, file_name)):
    frame_this_embryo = str(time_point).zfill(3)
    file_name=embryo_name+'_'+frame_this_embryo+'_segCell.nii.gz'

    # ------------if contact file exists, finish this embryo---------------
    # contact_saving_path = os.path.join(my_config.data_cell_mesh_and_contact, 'stat', embryo_name,
    #                                    embryo_name+'_'+file_name.split('.')[0] + '_contact.txt')
    # # ===============very important line===================
    # if os.path.exists(contact_saving_path):
    #     print(contact_saving_path,' existed')
    #     return 0
    # # =====================================================

    volume = nib.load(os.path.join(path_tmp, file_name)).get_fdata().astype(int).transpose([2, 1, 0])
    if is_debug:
        print(np.unique(volume,return_counts=True))
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
                print('contact',str_key,len(contact_points_list))

    if is_debug:
        print('volume info',np.unique(volume,return_counts=True))
    cell_list = np.unique(volume)
    contact_mesh_dict = {}
    showing_record = []

    volume_dict={}
    surface_dict={}
    contact_dict={}

    if is_debug:
        print('configuration',config_arg)
        print('cell list ',cell_list)

    # if not is_calculate_contact_file:
    #     print('loading ', contact_saving_path)
    #     with open(contact_saving_path,'rb') as handle:
    #         contact_mesh_dict = pickle.load(handle)

    # print(cell_list)
    weight_surface=1.2031
    count_ratio_tmp=[]
    for cell_key in cell_list:
        if cell_key != 0:
            cell_mask=np.logical_xor(ndimage.binary_dilation(volume == cell_key),(volume == cell_key))
            # if is_debug:
            #     print('-------',cell_key,'---------')
            #     print('surface num',(cell_mask==1).sum())
            #     print('inside sum',(volume == cell_key).sum())
            if (cell_mask==1).sum() > 15000:
                volume_dict[cell_key]=(volume == cell_key).sum()
                surface_dict[cell_key]=cell_mask.sum()*weight_surface # 1.2031... is derived by other papers
                irregularity=surface_dict[cell_key]**(1/2)/volume_dict[cell_key]**(1/3)
                if is_debug:
                    print('irregularity   ',irregularity)
                if irregularity< 2.199085:
                    print('impossible small surface', time_point,cell_key)
                for (cell1, cell2) in cell_conatact_pair_renew:
                    idx = str(cell1) + '_' + str(cell2)
                    # idx_test=
                    if cell_key not in (cell1, cell2) or idx in contact_dict.keys():
                        continue
                    # --------------------contact-----------------------------------------
                    contact_dict[idx]=len(contact_points_dict[idx])*weight_surface
            else:
                if is_debug:
                    print(cell_key,'surface point num',(cell_mask == 1).sum(),' inside point num(dia)',(ndimage.binary_dilation(volume == cell_key) == 1).sum())
                # ------------saving cell mesh---------------------

                tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_key) == 1)
                # print(len(tuple_tmp))
                sphere_list = np.concatenate(
                    (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
                adjusted_rate=0.01
                sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, adjusted_rate,
                                                                                     (len(tuple_tmp[0]), 3))
                m_mesh = generate_alpha_shape(sphere_list_adjusted, displaying=showCellMesh)

                alpha_v = 1

                if not m_mesh.is_watertight():
                    for i in range(10):
                        alpha_v = alpha_v+i*0.1
                        sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, adjusted_rate*(i+1),
                                                                                             (len(tuple_tmp[0]), 3))
                        m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=alpha_v,displaying=showCellMesh)
                        if is_debug:
                            print('watertight', m_mesh.is_watertight())
                            print(f"alpha={alpha_v:.3f}")
                            print('edge manifold boundary', m_mesh.is_edge_manifold(allow_boundary_edges=False))

                        if m_mesh.is_watertight():
                            break
                if is_debug:
                    print(cell_key,'=======mesh info=========', m_mesh)
                    print('edge manifold', m_mesh.is_edge_manifold(allow_boundary_edges=True))
                    print('edge manifold boundary', m_mesh.is_edge_manifold(allow_boundary_edges=False))
                    print('vertex manifold', m_mesh.is_vertex_manifold())
                    print('watertight', m_mesh.is_watertight())
                    print(f"alpha={alpha_v:.3f}")
                    print('volume====>',m_mesh.get_volume(), 'using  ',(volume == cell_key).sum())
                    print('surface area=======>',m_mesh.get_surface_area(),'weighted ', cell_mask.sum()*weight_surface)
                    # o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                    #                                   window_name=str(cell_key))
                # -----------------can not get watertight cell anyway-----------------------------------
                if not m_mesh.is_watertight():
                    print('no watertight mesh even after 10 times generation!!!', file_name,cell_key,'use point sum')
                    volume_dict[cell_key] = (volume == cell_key).sum()
                    surface_dict[cell_key] = cell_mask.sum() * weight_surface  # 1.154... is derived by 2/sqrt(3)
                    irregularity=(surface_dict[cell_key] ** (1 / 2) / volume_dict[cell_key] ** (1 / 3))
                    if is_debug:
                        print(irregularity)
                    if  irregularity< 2.199085:
                        print('impossible small surface!!!!!!!!!!!', time_point, cell_key)
                    for (cell1, cell2) in cell_conatact_pair_renew:
                        idx = str(cell1) + '_' + str(cell2)
                        # idx_test=
                        if cell_key not in (cell1, cell2) or idx in contact_dict.keys():
                            continue
                        # --------------------contact-----------------------------------------
                        contact_dict[idx] = len(contact_points_dict[idx]) * weight_surface
                    continue
                else: # watertight mesh
                    m_mesh = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(np.asarray(m_mesh.vertices).astype(int)), m_mesh.triangles)
                    volume_dict[cell_key]=(volume == cell_key).sum()
                    surface_dict[cell_key]=m_mesh.get_surface_area()

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
                        points_surface_area =len(contact_points_dict[idx])*weight_surface
                        contact_dict[idx]=alpha_surface_area if alpha_surface_area>points_surface_area else points_surface_area


                        if is_debug:
                            contact_mesh = deepcopy(m_mesh)
                            contact_mesh.remove_vertices_by_mask(contact_mask_not)
                            print(idx, '=======mesh info=========', m_mesh)
                            print('edge manifold', m_mesh.is_edge_manifold(allow_boundary_edges=True))
                            print('edge manifold boundary', m_mesh.is_edge_manifold(allow_boundary_edges=False))
                            print('vertex manifold', m_mesh.is_vertex_manifold())
                            # print('watertight', m_mesh.is_watertight())
                            print('surface area=======>', m_mesh.get_surface_area(),'while points num',points_surface_area)
                            # o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
                            #                                   mesh_show_wireframe=True)
                # -----------------CHECK IF THE SUM OF CONTACT SURFACE AREA-------------------------

                if is_debug:
                    this_cell_surface=surface_dict[cell_key]
                    contact_list_tmp = []

                    for (cell1, cell2) in cell_conatact_pair_renew:
                        idx_tmp = str(cell1) + '_' + str(cell2)
                        # idx_test=
                        if cell_key not in (cell1, cell2):
                            continue
                        contact_list_tmp.append(contact_dict[idx_tmp])
                    print('contact sum',sum(contact_list_tmp),'  surface area ' ,this_cell_surface)
                    ratio_this_cell=sum(contact_list_tmp)/this_cell_surface
                    if ratio_this_cell>1:
                        count_ratio_tmp.append(ratio_this_cell)
                        # print('impossible things!!!!')

    if is_debug:
        abnormal_contact_cell_ratio=len(count_ratio_tmp)/(len(cell_list)-1)
        print('volume dict ', volume_dict)
        print('surface dict', surface_dict)
        print('contact dict', contact_dict)
        print('-------------abnormal contact cell ratio!!!!====>  ', abnormal_contact_cell_ratio)
    # ------------saving volume surface and contact file for an embryo------------
    else:
        path_tmp=os.path.join(my_config.data_cell_mesh_and_contact,'stat', embryo_name)
        if not os.path.exists(path_tmp):
            os.mkdir(path_tmp)
        with open(os.path.join(path_tmp,file_name.split('.')[0] + '_volume.txt'),'wb+') as handle:
            pickle.dump(volume_dict, handle, protocol=4)
        with open(os.path.join(path_tmp,file_name.split('.')[0] + '_surface.txt'),'wb+') as handle:
            pickle.dump(surface_dict, handle, protocol=4)
        with open(os.path.join(path_tmp,file_name.split('.')[0] + '_contact.txt'),'wb+') as handle:
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
        path_tmp = my_config.data_path + r'Segmentation Results\SegmentedCell/Sample' + embryo_name + 'LabelUnified'
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
                    if contact_mask.sum() > 4:

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
                for cell_key in cell_list:
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


# def display_cell_mesh_contact_CMap(is_showing_cell_mesh=False, is_showing_cell_contact=True):
#     embryo_name, cell_name, tp = str(input()).split(',')
#     path_tmp = os.path.join(my_config.data_CMAP_seg, embryo_name, 'SegCell',
#                             '{}_{}_segCell.nii.gz'.format(embryo_name, tp))
#     this_img = load_nitf2_img(path_tmp)
#     volume = this_img.get_fdata().astype(int)
#
#     num_cellname, cellname_num = get_cell_name_affine_table(path=my_config.data_CMAP_seg + r'name_dictionary.csv')
#     cell_idx = cellname_num[cell_name]
#
#     # -------------------
#     cell_mask = volume != 0
#     boundary_mask = (cell_mask == 0) & ndimage.binary_dilation(cell_mask)
#     [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
#     boundary_elements = []
#
#     # find boundary between cells
#     for (x, y, z) in zip(x_bound, y_bound, z_bound):
#         neighbors = volume[np.ix_(range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2))]
#         neighbor_labels = list(np.unique(neighbors))
#         neighbor_labels.remove(0)
#         if len(neighbor_labels) == 2:  # contact between two cells
#             boundary_elements.append(neighbor_labels)
#     # cell contact pairs
#     cell_contact_pairs = list(np.unique(np.array(boundary_elements), axis=0))
#     cell_conatact_pair_renew = []
#     contact_points_dict = {}
#     contact_area_dict = {}
#
#     for (label1, label2) in cell_contact_pairs:
#         contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1),
#                                       ndimage.binary_dilation(volume == label2))
#         contact_mask = np.logical_and(contact_mask, boundary_mask)
#         if contact_mask.sum() > 2:
#
#             cell_conatact_pair_renew.append((label1, label2))
#             str_key = str(label1) + '_' + str(label2)
#             contact_area_dict[str_key] = 0
#
#             point_position_x, point_position_y, point_position_z = np.where(contact_mask == True)
#
#             contact_points_list = []
#             for i in range(len(point_position_x)):
#                 contact_points_list.append([point_position_x[i], point_position_y[i], point_position_z[i]])
#             # print(str_key)
#             contact_points_dict[str_key] = contact_points_list
#
#     cell_list = np.unique(volume)
#     contact_mesh_dict = {}
#     showing_record = []
#
#     # print(cell_list)
#     for cell_key in cell_list:
#         if cell_key != 0:
#             # print(os.path.exists(cellMesh_file_saving_path))
#             # print('calculating and saving', path_tmp, file_name,cell_key, ' surface')
#             tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_key) == 1)
#             sphere_list = np.concatenate(
#                 (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
#             sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, 0.01,
#                                                                                  (len(tuple_tmp[0]), 3))
#             m_mesh = generate_alpha_shape(sphere_list_adjusted, displaying=showCellMesh)
#             if not m_mesh.is_watertight():
#                 for i in range(10):
#                     sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, 0.01 * (i + 1),
#                                                                                          (len(tuple_tmp[0]), 3))
#                     m_mesh = generate_alpha_shape(sphere_list_adjusted, alpha_value=1 + i * 0.1,
#                                                   displaying=showCellMesh)
#                     if m_mesh.is_watertight():
#                         break
#             if not m_mesh.is_watertight():
#                 print('no watertight mesh even after 10 times generation!!!', file_name, cell_key)
#                 continue
#             # print('saving mesh')
#             # is_contact_file = True
#             # ============contact surface detection========================
#             cell_vertices = np.asarray(m_mesh.vertices).astype(int)
#             # ====================contact file ============================
#
#             for (cell1, cell2) in cell_conatact_pair_renew:
#                 idx = str(cell1) + '_' + str(cell2)
#                 # idx_test=
#                 if cell_key not in (cell1, cell2) or idx in contact_mesh_dict.keys():
#                     continue
#
#                 # --------------------contact-----------------------------------------
#                 # print('calculating, saving or showing',path_tmp, file_name, idx, ' contact surface')
#
#                 # build a mask to erase not contact points
#                 # enumerate each points in contact surface
#                 contact_mask_not = [True for i in range(len(cell_vertices))]
#                 contact_vertices_loc_list = []
#                 for [x, y, z] in contact_points_dict[idx]:
#                     # print(x,y,z)
#                     contact_vertices_loc = np.where(np.prod(cell_vertices == [x, y, z], axis=-1))
#                     if len(contact_vertices_loc[0]) != 0:
#                         contact_vertices_loc_list.append(contact_vertices_loc[0][0])
#                         contact_mask_not[contact_vertices_loc[0][0]] = False
#                 contact_mesh_dict[idx] = contact_vertices_loc_list
#
#                 if showCellContact:
#                     contact_mesh = deepcopy(m_mesh)
#                     contact_mesh.remove_vertices_by_mask(contact_mask_not)
#                     o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
#                                                       mesh_show_wireframe=True)
#
#
#         else:
#
#             for (cell1, cell2) in cell_conatact_pair_renew:
#                 contact_mask_not = [True for i in range(len(cell_vertices))]
#                 idx = str(cell1) + '_' + str(cell2)
#                 if cell_key not in (cell1, cell2) or idx in showing_record:
#                     continue
#                 print('reading or showing', idx, ' contact surface')
#                 print(showing_record)
#                 showing_record.append(idx)
#                 for value_ in contact_mesh_dict[idx]:
#                     contact_mask_not[value_] = False
#
#                 if showCellContact:
#                     contact_mesh = deepcopy(m_mesh)
#                     contact_mesh.remove_vertices_by_mask(contact_mask_not)
#                     o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True,
#                                                       mesh_show_wireframe=True)
#
#     print('sum of cell contact area', sum(contact_sur_area))
#     print('list of cell contact area', contact_sur_area)



def calculate_cell_points_CShaper():
    embryo_names = [str(i).zfill(2) for i in range(4, 21)]

    for embryo_name in embryo_names:
        # ------------------------calculate surface points using dialation for each cell --------------------
        path_tmp = my_config.data_path + r'Segmentation Results\SegmentedCell/Sample' + embryo_name + 'LabelUnified'
        for file_name in os.listdir(path_tmp):
            if os.path.isfile(os.path.join(path_tmp, file_name)):
                t0 = time()
                print(path_tmp, file_name)
                this_img = load_nitf2_img(os.path.join(path_tmp, file_name))
                img_arr = this_img.get_fdata().astype(int)

                cell_points = export_dia_cell_points_json(img_arr)

                dia_cell_saving = os.path.join(my_config.data_path + r'cell_dia_points', 'Sample' + embryo_name)
                if not os.path.exists(dia_cell_saving):
                    os.mkdir(dia_cell_saving)

                with open(os.path.join(dia_cell_saving, file_name.split('.')[0] + '.json'), 'w') as fp:
                    json.dump(cell_points, fp)

                print("done in %0.3fs" % (time() - t0))

#
# def calculate_cell_surface_points_CShaper():
#     embryo_names = [str(i).zfill(2) for i in range(4, 21)]
#
#     for embryo_name in embryo_names:
#         # ------------------------calculate surface points using dialation for each cell --------------------
#         path_tmp = my_config.data_path + r'Segmentation Results\SegmentedCell/Sample' + embryo_name + 'LabelUnified'
#         for file_name in os.listdir(path_tmp):
#             if os.path.isfile(os.path.join(path_tmp, file_name)):
#                 t0 = time()
#                 print(path_tmp, file_name)
#                 this_img = load_nitf2_img(os.path.join(path_tmp, file_name))
#                 img_arr = this_img.get_fdata().astype(int)
#
#                 cell_points = export_dia_cell_surface_points_json(img_arr)
#
#                 dia_surface_saving = os.path.join(config.data_path + r'cell_dia_surface', 'Sample' + embryo_name)
#                 if not os.path.exists(dia_surface_saving):
#                     os.mkdir(dia_surface_saving)
#
#                 with open(os.path.join(dia_surface_saving, file_name.split('.')[0] + '.json'), 'w') as fp:
#                     json.dump(cell_points, fp)
#
#                 print("done in %0.3fs" % (time() - t0))
#
#
# def display_contact_points_CShaper():
#     # Sample20,ABplpapapa,150
#     # Sample20,Dpaap,158
#     # Sample20,ABalaapa,078
#     # Sample20,ABa,005
#     # Sample20,MSp,035
#     print('waiting type you input: samplename and timepoints for embryogenesis')
#     embryo_name, cell_name, tp = str(input()).split(',')
#
#     num_cell_name, cell_num = get_cell_name_affine_table()
#     this_cell_keys = cell_num[cell_name]
#
#     with open(os.path.join(r'./DATA/cshaper_contact_data', embryo_name + '_' + tp + '_segCell.json'), ) as fp:
#         data = json.load(fp)
#     display_key_list = []
#     for idx in data.keys():
#         print(idx, len(data[idx]))
#         # if re.match('^' + str(this_cell_keys) + '_\d', idx) or re.match('\d_' + str(this_cell_keys) + '$', idx):
#         label1_2 = idx.split('_')
#         if str(this_cell_keys) in label1_2:
#             display_key_list.append(idx)
#
#     # fig_contact_info = plt.figure()
#     # plt.axis('off')
#     item_count = 1
#     print('contact number', len(display_key_list))
#     for idx in display_key_list:
#         if item_count > 9:
#             break
#
#         # if len(data[idx]) < 30:
#         #     continue
#         draw_points_list = []
#         print(idx)
#         for item_str in data[idx]:
#             x, y, z = item_str.split('_')
#             x, y, z = int(x), int(y), int(z)
#             draw_points_list.append([x, y, z])
#
#         [x_tmp, y_tmp, z_tmp] = np.amax(np.array(draw_points_list), axis=0)
#         contact_mask = np.zeros((x_tmp * 2, y_tmp * 2, z_tmp * 2))
#         for [x_tmp, y_tmp, z_tmp] in draw_points_list:
#             contact_mask[x_tmp, y_tmp, z_tmp] = True
#         verts, faces, _, _ = marching_cubes(contact_mask)
#         contact_mesh = o3d.geometry.TriangleMesh(o3d.cpu.pybind.utility.Vector3dVector(verts),
#                                                  o3d.cpu.pybind.utility.Vector3iVector(faces))
#
#         print(idx, '  marching cubes method surface area:', mesh_surface_area(verts, faces) / 2)
#
#         contact_mesh.compute_vertex_normals()
#         o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
#
#         # a = idx.split('_')
#         # a.remove(str(this_cell_keys))
#         # contact_cell_num = int(a[0])
#         # ax = fig_contact_info.add_subplot(3, 3, item_count, projection='3d')
#         # draw_3D_points(np.array(draw_points_list), fig_name=cell_name + '_' + num_cell_name[contact_cell_num], ax=ax)
#
#         # dfs = pd.read_excel(static.cell_fate_path, sheet_name=None)['CellFate']
#         # fate_cell = dfs[dfs['Name'] == cell_name + '\'']['Fate'].values[0].split('\'')[0]
#
#         item_count += 1
#     # plt.show()
#
#
# def display_contact_alpha_surface_CShaper():
#     # Sample20,ABplpapapa,150
#     # Sample20,ABalaapa,078
#     # Sample20,ABa,005
#     # Sample20,MSp,035
#     print('waiting type you input: sample name and time points for embryogenesis analysis')
#     embryo_name, cell_name, tp = str(input()).split(',')
#
#     num_cell_name, cell_num = get_cell_name_affine_table()
#     this_cell_keys = cell_num[cell_name]
#
#     # -------getting all data points including dilation  points -- for generate alpha shape--------
#     with open(os.path.join(r'./DATA/cell_dia_points', embryo_name + '_' + tp + '_segCell.json')) as fp:
#         cell_data = json.load(fp)
#     cell_points_building_as = []
#     # print(cell_data.keys())
#     for item_str in cell_data[str(this_cell_keys)]:
#         x, y, z = item_str.split('_')
#         x, y, z = float(x) + uniform(0, 0.001), float(y) + uniform(0, 0.001), float(
#             z) + uniform(0, 0.001)
#         cell_points_building_as.append([x, y, z])
#     cell_points_building_as = np.array(cell_points_building_as)
#     # print(cell_points_building_as)
#     m_mesh = generate_alpha_shape(cell_points_building_as)
#     # ---------------------------finished generating alpha shape -------------------------------
#
#     with open(os.path.join(r'./DATA/cell_dia_surface', embryo_name, embryo_name + '_' + tp + '_segCell.json'),
#               'rb') as fp:
#         surface_data = json.load(fp)
#
#     with open(os.path.join(r'./DATA/cshaper_contact_data', embryo_name, embryo_name + '_' + tp + '_segCell.json'),
#               'rb') as fp:
#         surface_contact_data = json.load(fp)
#
#     get_contact_surface_mesh(this_cell_keys, surface_data, surface_contact_data, m_mesh)



if __name__ == "__main__":
    detect_outer_cells()
    # calculate_cell_surface_and_contact_points(is_calculate_cell_mesh=False, is_calculate_contact_file=False,
    #                                           showCellMesh=True, showCellContact=True)
    # display_cell_mesh_contact_CMap()

    # calculate_cell_surface_and_contact_points_CMap()
