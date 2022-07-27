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

from scipy import ndimage
# from skimage.measure import marching_cubes, mesh_surface_area
from tqdm import tqdm

import static.config as my_config
from utils.cell_func import get_cell_name_affine_table
from utils.general_func import load_nitf2_img
from utils.shape_preprocess import export_dia_cell_points_json
from utils.shape_model import generate_alpha_shape, get_contact_surface_mesh


contact_path=r"/home/home/ProjectCode/LearningCell/MembProjectCode/statistics/191108plc1p1/191108plc1p1_contact.csv"
volume_path=r"/home/home/ProjectCode/LearningCell/MembProjectCode/statistics/191108plc1p1/191108plc1p1_volume.csv"
surface_path=r"/home/home/ProjectCode/LearningCell/MembProjectCode/statistics/191108plc1p1/191108plc1p1_surface.csv"



df_contact=pd.read_csv(contact_path,header=[0,1],index_col=0)
df_volume=pd.read_csv(volume_path,header=0,index_col=0)
df_surface=pd.read_csv(surface_path,header=0,index_col=0)

pd_number = pd.read_csv(r"/home/home/ProjectCode/LearningCell/MembProjectCode/dataset/number_dictionary.csv", names=["name", "label"])
number_dict = pd.Series(pd_number.label.values, index=pd_number.name).to_dict()

# print(df_volume)
# print(df_surface)
# print(df_contact)
# print(number_dict)

irregularity_list=[]
ratio_list=[]
# for tp in df_volume.index:
#     for cell_name in df_volume.columns:
#         # print()
#
#         if df_volume[cell_name].notnull().loc[tp]:
#             # print(df_volume.at[tp, cell_name])
#             irre=df_surface.at[tp,cell_name]**(1/2)/df_volume.at[tp,cell_name]**(1/3)
#             if irre<2.199999:
#                 print('impossible irregulartiy', tp, cell_name,irre)
#             elif irre>3:
#                 print('strange irregulartiy', tp, cell_name,irre)

contact_dict={}

for tp in df_contact.index:
    for (cell_name1,cell_name2) in df_contact.columns:
        # print(cell_name1,cell_name2)
        if df_contact[(cell_name1,cell_name2)].notnull().loc[tp]:
            contact_value_tmp=df_contact.at[tp,(cell_name1,cell_name2)]
            if (tp,cell_name1) in contact_dict.keys():
                contact_dict[(tp,cell_name1)].append(contact_value_tmp)
            else:
                contact_dict[(tp, cell_name1)]=[contact_value_tmp]

            if (tp,cell_name2) in contact_dict.keys():
                contact_dict[(tp,cell_name2)].append(contact_value_tmp)
            else:
                contact_dict[(tp, cell_name2)]=[contact_value_tmp]
print(contact_dict)
for tp in df_surface.index:
    for cell_name in df_surface.columns:
        # print()

        if df_surface[cell_name].notnull().loc[tp]:
            ratio_tmp=round(sum(contact_dict[(tp, cell_name)])/df_surface.at[tp, cell_name],2)
            ratio_list.append(ratio_tmp)
            if ratio_tmp>1:
                print(tp,cell_name,ratio_tmp)

print(np.unique(np.array(ratio_list),return_counts=True))


# try:
#     with open(volume_path, 'rb') as handle:
#         volume = pickle.load(handle)
# except:
#     print('open failed ', volume_path)
#
# embryo_path="/home/home/ProjectCode/LearningCell/MembProjectCode/gui/191108plc1p1/SegCell/191108plc1p1_179_segCell.nii.gz"
# embryo_=nib.load(embryo_path).get_fdata().astype(int).transpose([2, 1, 0])
# print(np.unique(embryo_,return_counts=True))

# max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
#     # max_times = [205, 205, 255, 195, 195, 185]
#
# embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
#                     '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
#                     '200117plc1pop1ip3']
#
#
# embryo_name='191108plc1p1'
# max_time=205
# path_tmp = os.path.join(r'/home/home/ProjectCode/LearningCell/MembProjectCode/gui/', embryo_name, 'SegCell')
#
#
# for tp in range(1, max_time + 1):
#     # ------------------------calculate surface points using dialation for each cell --------------------
#     # for file_name in os.listdir(path_tmp):
#         # if os.path.isfile(os.path.join(path_tmp, file_name)):
#     frame_this_embryo = str(tp).zfill(3)
#     file_name=embryo_name+'_'+frame_this_embryo+'_segCell.nii.gz'
#
#     volume = nib.load(os.path.join(path_tmp, file_name)).get_fdata().astype(int).transpose([2, 1, 0])
#
#     print(tp,len(np.unique(volume).tolist()))