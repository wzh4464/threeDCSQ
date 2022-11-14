# -------other's package------
import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import KMeans
import multiprocessing as mp
from tqdm import tqdm


# --------user's package------------

import static.config as my_config
from test_shape_analysis import calculate_cell_surface_and_contact_points
from utils.shape_model import generate_alpha_shape
import utils.data_io as data_io

def re_assign_CMap_wrong_dividing_cell():

    label_name_dict = pd.read_csv(my_config.data_label_name_dictionary, header=0, index_col=0).to_dict()['0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}

    with open(os.path.join('tem_files', 'SegCellTimeCombinedLabelUnified_'+'wrong_division_cells.pikcle'), 'rb') as fp:
        wrong_divison_file = pickle.load(fp)

    current_tp=1
    # step_folder_name = 'SegCellTimeCombinedLabelUnifiedPost1'

    # print(wrong_divison_file)
    reassign_dict={}
    for [embryo_name,cell_name,tp_current] in wrong_divison_file:
        origin_seg_embryo=nib.load(os.path.join(my_config.data_linux_CMAP_seg, embryo_name, 'SegCell',
                                                  '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp_current).zfill(3)))).get_data()
        combine_embryo_path=os.path.join(my_config.data_linux_CMAP_seg, embryo_name, 'SegCellTimeCombinedLabelUnified',
                     '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp_current).zfill(3)))

        if (embryo_name,tp_current) in reassign_dict.keys():
            combine_embryo_path = os.path.join(my_config.data_linux_CMAP_seg, embryo_name,
                                               'SegCellTimeCombinedLabelUnifiedPost1',
                                               '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp_current).zfill(3)))
        reassign_dict[(embryo_name,tp_current)]=1

        combined_unified_seg_embryo = nib.load(combine_embryo_path).get_data()
        cell_label=name_label_dict[cell_name]
        cell1_label=name_label_dict[cell_name+'a']
        cell2_label=name_label_dict[cell_name+'p']
        output_seg_cell = combined_unified_seg_embryo.copy()

        output_seg_cell[output_seg_cell==cell_label]=0
        cell1_pos_mask=(origin_seg_embryo == cell1_label)
        output_seg_cell[cell1_pos_mask]=cell1_label
        cell2_pos_mask = (origin_seg_embryo == cell2_label)
        output_seg_cell[cell2_pos_mask] = cell2_label
        print(origin_seg_embryo.shape,combined_unified_seg_embryo.shape,np.unique(combined_unified_seg_embryo==cell_label,return_counts=True),
              np.unique(cell1_pos_mask,return_counts=True),np.unique(cell2_pos_mask,return_counts=True))

        save_combine_embryo_path = os.path.join(my_config.data_linux_CMAP_seg, embryo_name,
                                           'SegCellTimeCombinedLabelUnifiedPost1',
                                           '{}_{}_segCell.nii.gz'.format(embryo_name, str(tp_current).zfill(3)))
        print(embryo_name,tp_current,cell_name,' ------> ', cell1_label, cell2_label)
        data_io.nib_save(output_seg_cell,save_combine_embryo_path)


def save_combine_wrong_dividing_cell_CMap():
    max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    # max_times = [195, 140, 155]

    embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
                    '200117plc1pop1ip3']
    label_name_dict = pd.read_csv(my_config.data_label_name_dictionary, header=0, index_col=0).to_dict()['0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}
    list_wrong_division=[]
    step_folder_name = 'SegCellTimeCombinedLabelUnifiedPost1'

    for embyro_idx, embryo_name in enumerate(embryo_names):
        # path_tmp = os.path.join(my_config.data_CMAP_seg, embryo_name,'SegCellTimeCombinedLabelUnifiedPost1')
        # if not os.path.exists(path_tmp):
        #     os.mkdir(path_tmp)
        for tp_current in tqdm(range(100,max_times[embyro_idx]+1),desc='working on CMap {} seperate region {}'.format(step_folder_name,embryo_name)):
            combined_unified_seg_by_jf=os.path.join(my_config.data_linux_CMAP_seg, embryo_name,step_folder_name ,
                                                    '{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))
            # post_seg_by_zelin=os.path.join(path_tmp,'{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))

            with open(os.path.join(my_config.data_linux_frozen_tem_data, 'tem_division_folder',embryo_name, 'DivisionCell',
                                   '{}_{}_division.txt'.format(embryo_name, str(tp_current).zfill(3)))) as f:
                dividing_information = f.readlines()
            division_cell_list=dividing_information[0].split(',')
            if division_cell_list[0]=='\n':
                continue
            division_cell_list_int=np.array(division_cell_list).astype(int)
            # print('{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)),division_cell_list_int)

            volume = nib.load(combined_unified_seg_by_jf).get_fdata().astype(int)
            cell_list = np.unique(volume)
            for cell_key in cell_list:
                # if (volume==cell_key).sum()>4096:
                #     continue

                if cell_key != 0 and cell_key in division_cell_list_int:

                    tuple_tmp = np.where(ndimage.binary_dilation(volume == cell_key) == 1)
                    # print(len(tuple_tmp))
                    sphere_list = np.concatenate(
                        (tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
                    adjusted_rate = 0.01
                    sphere_list_adjusted = sphere_list.astype(float) + np.random.uniform(0, adjusted_rate,
                                                                                         (len(tuple_tmp[0]), 3))
                    alpha_v = 1.5
                    m_mesh = generate_alpha_shape(sphere_list_adjusted,alpha_value=alpha_v)
                    # print(len(m_mesh.cluster_connected_triangles()[2]))

                    # alpha_v = 1

                    if len(m_mesh.cluster_connected_triangles()[2])>1:
                        print('wrong dividing ',cell_key,label_name_dict[cell_key]+' {}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))
                        list_wrong_division.append([embryo_name,label_name_dict[cell_key],tp_current])
                        # o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                        #                                   window_name=label_name_dict[cell_key]+' {}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))

                    # else:
                    #     print('correct dividing ',cell_key, label_name_dict[cell_key] + ' {}_{}_segCell.nii.gz'.format(embryo_name, str(tp_current).zfill(3)))
                    #     o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                    #                                       window_name=label_name_dict[
                    #                                                       cell_key] + ' {}_{}_segCell.nii.gz'.format(
                    #                                           embryo_name, str(tp_current).zfill(3)))
    print(list_wrong_division)
    with open(os.path.join('tem_files',step_folder_name+'_wrong_division_cells.pikcle'), 'wb') as fp:
        pickle.dump(list_wrong_division, fp)

# def pre_wrong_dividing_cell_CMap_label_generation():
#     label_name_dict = pd.read_csv(os.path.join(my_config.data_linux_CMAP_seg_gui, 'name_dictionary.csv'), header=0,
#                                   index_col=0).to_dict()['0']
#     name_label_dict = {label: name for name, label in label_name_dict.items()}
#
#     with open(os.path.join('tem_files','wrong_division_cells.pikcle'), "rb") as fp:  # Unpickling
#         list_wrong_division = pickle.load(fp)
#
#     label_wrong_division_list=[]
#     for value_item in list_wrong_division:
#         this_embryo_name = value_item[0]
#         cell_name_list = value_item[1]
#         this_embryo_tp = value_item[2]
#
#         cell_label=name_label_dict[cell_name_list]
#         label_wrong_division_list.append([this_embryo_name,cell_label,this_embryo_tp])
#     print(list_wrong_division)
#     with open(os.path.join('tem_files','wrong_division_cells_label.pikcle'), 'wb') as fp:
#         pickle.dump(label_wrong_division_list, fp)



def update_wrong_dividing_cell_stat_CMap():
    # todo : carefully, after this, all stat for CMAP will be update
    label_name_dict = pd.read_csv(os.path.join(my_config.data_linux_CMAP_seg_gui, 'name_dictionary.csv'), header=0,
                                  index_col=0).to_dict()['0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}

    with open(os.path.join('tem_files', 'SegCellTimeCombinedLabelUnified_'+'wrong_division_cells.pikcle'), "rb") as fp:  # Unpickling
        list_wrong_division = pickle.load(fp)

    last_embryo_name = 'starttt'
    working_list=[]
    for idx_item, value_item in enumerate(list_wrong_division):
        this_embryo_name = value_item[0]
        cell_name_list = [value_item[1]]
        this_embryo_tp = value_item[2]
        if this_embryo_tp == list_wrong_division[idx_item - 1][2]:
            continue

        for i in range(1, 10):
            # print(list_wrong_division[idx_item + i])
            if idx_item + i < len(list_wrong_division) and this_embryo_tp == list_wrong_division[idx_item + i][2]:
                cell_name_list.append(list_wrong_division[idx_item + i][1])
            else:
                break
        if this_embryo_name != last_embryo_name:
            # ace_file_path = os.path.join(my_config.cmap_data_original, this_embryo_name,
            #                              "CD" + this_embryo_name + ".csv")  # pixel x y z
            # cd_file = data_io.read_new_cd(ace_file_path)
            # df_cd_file = pd.DataFrame(data=cd_file.values[:, 1:], index=cd_file['Cell & Time'],
            #                           columns=cd_file.columns[1:])
            # print(df_cd_file)

            last_embryo_name = this_embryo_name
        working_list.append([this_embryo_name,this_embryo_tp])

    configs = []
    for [this_embryo_name,this_embryo_tp] in tqdm(working_list, desc="Compose configs"):
        config_tmp = {}
        config_tmp['time_point'] = this_embryo_tp
        config_tmp["embryo_name"] = this_embryo_name
        config_tmp['path_embryo']= os.path.join(my_config.data_linux_CMAP_seg, this_embryo_name, 'SegCellTimeCombinedLabelUnifiedPost1')
        config_tmp['showCellMesh'] = False
        configs.append(config_tmp)

    mpPool = mp.Pool(15)
    print(configs)
    # mpPool = mp.Pool(9)

    for idx_, _ in enumerate(
            tqdm(mpPool.imap_unordered(calculate_cell_surface_and_contact_points, configs), total=len(working_list),
                 desc="re-caculate the wrong division embryo stat")):
        #
        pass


if __name__ == "__main__":
    # pre_wrong_dividing_cell_CMap_label_generation()
    # save_combine_wrong_dividing_cell_CMap()
    update_wrong_dividing_cell_stat_CMap()
    # re_assign_CMap_wrong_dividing_cell()