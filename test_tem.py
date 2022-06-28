import json
import os
import pickle
from copy import deepcopy
from random import uniform

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

max_times = [205, 205, 255, 195, 195, 185, 220, 195, 195, 195, 140, 155]
    # max_times = [205, 205, 255, 195, 195, 185]

embryo_names = ['191108plc1p1', '200109plc1p1', '200113plc1p2', '200113plc1p3', '200322plc1p2', '200323plc1p1',
                    '200326plc1p3', '200326plc1p4', '200122plc1lag1ip1', '200122plc1lag1ip2', '200117plc1pop1ip2',
                    '200117plc1pop1ip3']


embryo_name='191108plc1p1'
max_time=205
path_tmp = os.path.join(r'/home/home/ProjectCode/LearningCell/MembProjectCode/gui/', embryo_name, 'SegCell')


for tp in range(1, max_time + 1):
    # ------------------------calculate surface points using dialation for each cell --------------------
    # for file_name in os.listdir(path_tmp):
        # if os.path.isfile(os.path.join(path_tmp, file_name)):
    frame_this_embryo = str(tp).zfill(3)
    file_name=embryo_name+'_'+frame_this_embryo+'_segCell.nii.gz'

    volume = nib.load(os.path.join(path_tmp, file_name)).get_fdata().astype(int).transpose([2, 1, 0])

    print(tp,len(np.unique(volume).tolist()))