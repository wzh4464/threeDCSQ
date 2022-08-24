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


def pre_combine_wrong_dividing_cell_CMap():
    max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    # max_times = [195, 140, 155]

    embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
                    '200117plc1pop1ip3']
    label_name_dict = pd.read_csv(my_config.data_label_name_dictionary, header=0, index_col=0).to_dict()['0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}
    list_wrong_division=[]
    for embyro_idx, embryo_name in enumerate(embryo_names):
        # path_tmp = os.path.join(my_config.data_CMAP_seg, embryo_name,'SegCellTimeCombinedLabelUnifiedPost1')
        # if not os.path.exists(path_tmp):
        #     os.mkdir(path_tmp)
        for tp_current in range(100,max_times[embyro_idx]+1):
            combined_unified_seg_by_jf=os.path.join(my_config.data_linux_CMAP_seg, embryo_name, 'SegCellTimeCombinedLabelUnified',
                                                    '{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))
            # post_seg_by_zelin=os.path.join(path_tmp,'{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))

            with open(os.path.join(my_config.data_linux_CMAP_seg_gui, embryo_name, 'DivisionCell', '{}_{}_division.txt'.format(embryo_name, str(tp_current).zfill(3)))) as f:
                dividing_information = f.readlines()
            division_cell_list=dividing_information[0].split(',')
            if division_cell_list[0]=='\n':
                continue
            division_cell_list_int=np.array(division_cell_list).astype(int)
            print('{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)),division_cell_list_int)

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
                    alpha_v = 1
                    m_mesh = generate_alpha_shape(sphere_list_adjusted,alpha_value=alpha_v)
                    # print(len(m_mesh.cluster_connected_triangles()[2]))

                    # alpha_v = 1

                    if len(m_mesh.cluster_connected_triangles()[2])>1:
                        print('wrong dividing ',cell_key,label_name_dict[cell_key]+' {}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))
                        list_wrong_division.append([embryo_name,label_name_dict[cell_key],tp_current])
                        o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                                                          window_name=label_name_dict[cell_key]+' {}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))

                    # else:
                    #     print('correct dividing ',cell_key, label_name_dict[cell_key] + ' {}_{}_segCell.nii.gz'.format(embryo_name, str(tp_current).zfill(3)))
                    #     o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                    #                                       window_name=label_name_dict[
                    #                                                       cell_key] + ' {}_{}_segCell.nii.gz'.format(
                    #                                           embryo_name, str(tp_current).zfill(3)))
        print(list_wrong_division)
    with open(os.path.join('tem_files','wrong_division_cells.pikcle'), 'wb') as fp:
        pickle.dump(list_wrong_division, fp)

def pre_wrong_dividing_cell_CMap_label_generation():
    label_name_dict = pd.read_csv(os.path.join(my_config.data_linux_CMAP_seg_gui, 'name_dictionary.csv'), header=0,
                                  index_col=0).to_dict()['0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}

    with open(os.path.join('tem_files','wrong_division_cells.pikcle'), "rb") as fp:  # Unpickling
        list_wrong_division = pickle.load(fp)

    label_wrong_division_list=[]
    for value_item in list_wrong_division:
        this_embryo_name = value_item[0]
        cell_name_list = value_item[1]
        this_embryo_tp = value_item[2]

        cell_label=name_label_dict[cell_name_list]
        label_wrong_division_list.append([this_embryo_name,cell_label,this_embryo_tp])
    print(list_wrong_division)
    with open(os.path.join('tem_files','wrong_division_cells_label.pikcle'), 'wb') as fp:
        pickle.dump(label_wrong_division_list, fp)


def combine_wrong_dividing_cell_CMap():
    label_name_dict = pd.read_csv(os.path.join(my_config.data_linux_CMAP_seg_gui, 'name_dictionary.csv'), header=0, index_col=0).to_dict()['0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}


    with open(os.path.join('tem_files','wrong_division_cells.pikcle'), "rb") as fp:  # Unpickling
        list_wrong_division = pickle.load(fp)

    last_embryo_name='starttt'
    for idx_item,value_item in enumerate(list_wrong_division):
        this_embryo_name=value_item[0]
        cell_name_list=[value_item[1]]
        this_embryo_tp=value_item[2]
        if this_embryo_tp==list_wrong_division[idx_item-1][2]:
            continue

        for i in range(1,10):
            # print(list_wrong_division[idx_item + i])
            if idx_item + i< len(list_wrong_division) and this_embryo_tp == list_wrong_division[idx_item +i][2]:
                cell_name_list.append(list_wrong_division[idx_item +i][1])
            else:
                break
        if this_embryo_name != last_embryo_name:
            ace_file_path = os.path.join(my_config.cmap_data_original, this_embryo_name, "CD" + this_embryo_name + ".csv")  # pixel x y z
            cd_file=data_io.read_new_cd(ace_file_path)
            df_cd_file=pd.DataFrame(data=cd_file.values[:,1:],index=cd_file['Cell & Time'],columns=cd_file.columns[1:])
            # print(df_cd_file)

            last_embryo_name=this_embryo_name
        print('working on ',this_embryo_name,cell_name_list,this_embryo_tp)
        combined_unified_seg_by_jf = os.path.join(my_config.data_linux_CMAP_seg, this_embryo_name,
                                                  'SegCellTimeCombinedLabelUnified',
                                                  '{}_{}_segCell.nii.gz'.format(this_embryo_name, str(this_embryo_tp).zfill(3)))
        combined_embryo=nib.load(combined_unified_seg_by_jf).get_fdata().astype(int)

        for dealing_cell_name in cell_name_list:
            dealing_cell_label=name_label_dict[dealing_cell_name]
            # daughter_cell1,daughter_cell2=dealing_cell_label+1,dealing_cell_label+2
            # daughter_cell1_name=label_name_dict[daughter_cell1]
            # daughter_cell2_name=label_name_dict[daughter_cell2]
            print(dealing_cell_name,dealing_cell_name+'a',dealing_cell_name+'p')

            tuple_tmp=np.where(ndimage.binary_dilation(combined_embryo == dealing_cell_label) == 1)
            numpy_pos=np.concatenate((tuple_tmp[0][:, None], tuple_tmp[1][:, None], tuple_tmp[2][:, None]), axis=1)
            kmean_cluster_pos = KMeans(n_clusters=2,init='random').fit(numpy_pos)
            # center_cell = np.mean(numpy_pos,axis=0)
            # print(numpy_pos,center_cell)
            # print(numpy_pos-center_cell)

            # ==========plotting testtttttttttttttttttttttt====================
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(numpy_pos[:,0], numpy_pos[:,1], numpy_pos[:,2],c=kmean_cluster_pos.labels_)
            plt.show()
            #
            # daughter_cell1_label=name_label_dict[dealing_cell_name+'a']
            # tuple_tmp1 = np.where(ndimage.binary_dilation(combined_embryo == daughter_cell1_label) == 1)
            # # print(len(tuple_tmp))
            # print(np.concatenate(
            #     (tuple_tmp1[0][:, None], tuple_tmp1[1][:, None], tuple_tmp1[2][:, None]), axis=1))
            # cell1_center = np.mean(np.concatenate(
            #     (tuple_tmp1[0][:, None], tuple_tmp1[1][:, None], tuple_tmp1[2][:, None]), axis=1),axis=0)
            # print(cell1_center,df_cd_file.loc[dealing_cell_name+'a'+':'+str(this_embryo_tp)][:3])
            #
            # daughter_cell2_label = name_label_dict[dealing_cell_name + 'p']
            # tuple_tmp2 = np.where(ndimage.binary_dilation(combined_embryo == daughter_cell2_label) == 1)
            # # print(len(tuple_tmp))
            # cell2_center = np.mean(np.concatenate(
            #     (tuple_tmp2[0][:, None], tuple_tmp2[1][:, None], tuple_tmp2[2][:, None]), axis=1), axis=0)
            # print(cell2_center, df_cd_file.loc[dealing_cell_name + 'p' + ':' + str(this_embryo_tp)][:3])

def update_wrong_dividing_cell_stat_CMap():
    # todo : carefully, after this, all stat for CMAP will be update
    label_name_dict = pd.read_csv(os.path.join(my_config.data_linux_CMAP_seg_gui, 'name_dictionary.csv'), header=0,
                                  index_col=0).to_dict()['0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}

    with open(os.path.join('tem_files', 'wrong_division_cells.pikcle'), "rb") as fp:  # Unpickling
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
        config_tmp['path_embryo']= os.path.join(my_config.data_linux_CMAP_seg, this_embryo_name, 'SegCellTimeCombined')
        config_tmp['showCellMesh'] = False
        configs.append(config_tmp)

    mpPool = mp.Pool(30)
    print(configs)
    # mpPool = mp.Pool(9)

    for idx_, _ in enumerate(
            tqdm(mpPool.imap_unordered(calculate_cell_surface_and_contact_points, configs), total=len(working_list),
                 desc="re-caculate the wrong division embryo stat")):
        #
        pass


if __name__ == "__main__":
    # pre_wrong_dividing_cell_CMap_label_generation()
    # pre_combine_wrong_dividing_cell_CMap()
    update_wrong_dividing_cell_stat_CMap()