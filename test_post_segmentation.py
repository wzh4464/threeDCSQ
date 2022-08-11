# -------other's package------
import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import pandas as pd
import open3d as o3d


# --------user's package------------

import static.config as my_config
from utils.shape_model import generate_alpha_shape


def combine_wrong_dividing_cell_CMap():
    max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    # max_times = [195, 140, 155]

    embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
                    '200117plc1pop1ip3']
    label_name_dict = pd.read_csv(os.path.join(my_config.data_label_name_dictionary), header=0, index_col=0).to_dict()['0']
    name_label_dict = {label: name for name, label in label_name_dict.items()}
    for embyro_idx, embryo_name in enumerate(embryo_names):
        path_tmp = os.path.join(my_config.segmentation_by_jf, embryo_name,'SegCellTimeCombinedLabelUnifiedPost1')
        if not os.path.exists(path_tmp):
            os.mkdir(path_tmp)
        for tp_current in range(100,max_times[embyro_idx]+1):
            combined_unified_seg_by_jf=os.path.join(my_config.segmentation_by_jf,embryo_name,
                                                    'SegCellTimeCombinedLabelUnified',
                                                    '{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))
            post_seg_by_zelin=os.path.join(path_tmp,'{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))

            with open(os.path.join(my_config.data_CMAP_seg_gui,embryo_name,'DivisionCell','{}_{}_division.txt'.format(embryo_name,str(tp_current).zfill(3)))) as f:
                dividing_information = f.readlines()
            division_cell_list=dividing_information[0].split(',')
            if division_cell_list[0]=='\n':
                continue
            division_cell_list_int=np.array(division_cell_list).astype(int)
            print('{}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)),division_cell_list_int)

            volume = nib.load(combined_unified_seg_by_jf).get_fdata().astype(int)
            cell_list = np.unique(volume)
            for cell_key in cell_list:
                if (volume==cell_key).sum()>4096:
                    continue

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
                        o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                                                          window_name=label_name_dict[cell_key]+' {}_{}_segCell.nii.gz'.format(embryo_name,str(tp_current).zfill(3)))
                    else:
                        print('correct dividing ',cell_key, label_name_dict[cell_key] + ' {}_{}_segCell.nii.gz'.format(embryo_name, str(tp_current).zfill(3)))
                        o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                                                          window_name=label_name_dict[
                                                                          cell_key] + ' {}_{}_segCell.nii.gz'.format(
                                                              embryo_name, str(tp_current).zfill(3)))

if __name__ == "__main__":
    combine_wrong_dividing_cell_CMap()