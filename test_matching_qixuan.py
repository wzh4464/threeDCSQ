import nibabel as nib
import numpy as np
import glob
import os

import pandas as pd

from experiment.export_tif.utils import nib_load

packed_data = r'F:\packed membrane nucleus 3d niigz'
embryo_name1='200311plc1p1'
embryo_name2='200311plc1p3'
two_embryos_cell_number = {'200311plc1p1': {}, '200311plc1p3': {}}
target_cell_numbers = list(np.arange(4,360))
for cell_num in target_cell_numbers:
    for key in two_embryos_cell_number.keys():
        two_embryos_cell_number[key][cell_num] = []

for key_emb_name in two_embryos_cell_number.keys():
    annotated_nuc_niigzs = sorted(glob.glob(os.path.join(packed_data, key_emb_name, 'AnnotatedNuc', '*nii.gz')))
    for annotated_nuc in annotated_nuc_niigzs:
        tp_this = os.path.basename(annotated_nuc).split('.')[0].split('_')[1]
        cell_number = len(np.unique(nib_load(annotated_nuc))) - 1
        if cell_number in target_cell_numbers:
            two_embryos_cell_number[key_emb_name][cell_number].append(tp_this)

print(two_embryos_cell_number)

for cell_num in target_cell_numbers:
    for key_emb_name in two_embryos_cell_number.keys():
        if len(two_embryos_cell_number[key_emb_name][cell_num])<1:
            continue
        nuc_pos_pd = pd.DataFrame(columns=['cell', 'x', 'y', 'z'])

        list_this_num = two_embryos_cell_number[key_emb_name][cell_num]
        tp_this = list_this_num[len(list_this_num) // 2-1]
        annotated_nuc_arr = nib_load(os.path.join(packed_data, key_emb_name, 'AnnotatedNuc',
                                                  '{}_{}_annotatedNuc.nii.gz'.format(key_emb_name, tp_this)))
        cell_labels = np.unique(annotated_nuc_arr)[1:]
        for cell_label_this in cell_labels:
            position_nuc_this = np.where(annotated_nuc_arr == cell_label_this)
            point_index = len(position_nuc_this[0]) // 2
            x = position_nuc_this[0][point_index]
            y = position_nuc_this[1][point_index]
            z = position_nuc_this[2][point_index]
            nuc_pos_pd.loc[len(nuc_pos_pd)] = [cell_label_this, x, y, z]
        nuc_pos_pd.to_csv(os.path.join(
            r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\08paper qixuan matching',
            '{}cell_{}cell.csv'.format(str(cell_num).zfill(3),key_emb_name)),index=False)
