#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import dependency library

import numpy as np
import pandas as pd
from static import config

from scipy import ndimage
from collections import Counter
import csv

import os

# import user defined library

import utils.general_func as general_f



def get_cell_name_affine_table(path=config.cell_shape_analysis_data_path + r'name_dictionary_no_name.csv'):
    """

    :return: a set of NO. to name LIST and name to NO. DICTIONARY:
    zero first, but actually there are no zero, remember to plus 1
    """
    label_name_dict = pd.read_csv(path, index_col=0).to_dict()['0']
    name_label_dict = {value: key for key, value in label_name_dict.items()}

    return label_name_dict, name_label_dict


def nii_get_cell_surface(img_arr, cell_key):

    # with the original image data
    img_arr_dilation = ndimage.binary_dilation(img_arr == cell_key)
    # print(np.unique(img_arr,return_counts=True))
    # img_df_erosion = pd.DataFrame(img_arr_erosion[100:150, 150:200, 100])
    # print(img_df_erosion) me

    surface_data_result = np.logical_xor(img_arr_dilation, (img_arr == cell_key))
    # be careful!
    surface_loc = np.array(np.where(surface_data_result)).T

    return surface_loc, np.mean(surface_loc, axis=0)


# np.set_printoptions(threshold=100000)

def nii_count_volume_surface(this_image):
    """

    :param this_image: the nii image from 3D image, count volume and surface
    :return: volume counter, surface counter
    """

    img_arr = this_image.get_data()
    img_arr_shape = img_arr.shape
    img_arr_count_shape = np.prod(img_arr_shape)

    struc_element = ndimage.generate_binary_structure(3, -1)

    # ---------------- erosion ----------------
    # with the original image data
    img_arr_erosion = ndimage.grey_erosion(img_arr, footprint=struc_element)

    surface_data_result = img_arr - img_arr_erosion

    cnt1 = Counter(np.reshape(img_arr, img_arr_count_shape))
    del cnt1[0]
    cnt2 = Counter(np.reshape(surface_data_result, img_arr_count_shape))
    del cnt2[0]

    return cnt1, cnt2


def nii_count_contact_surface(this_image):
    img_arr = this_image.get_data()
    img_arr_shape = img_arr.shape
    img_arr_count = np.prod(img_arr_shape)
    cnt = Counter(np.reshape(img_arr, img_arr_count))
    print(type(cnt))


def count_volume_surface_normalization_tocsv(path_tmp):
    """
    normalization coefficient= (volume/10000)**(1/3)
    :param path_tmp:
    :return:
    """
    name_list, _ = get_cell_name_affine_table()
    data_embryo_time_slices = pd.DataFrame(columns=['volume', 'surface', 'normalized_c'])

    for temporal_embryo in os.listdir(path_tmp):
        if os.path.isfile(os.path.join(path_tmp, temporal_embryo)):
            img = general_f.load_nitf2_img(os.path.join(path_tmp, temporal_embryo))

            volume_counter, surface_counter = nii_count_volume_surface(img)
            time_point = str.split(temporal_embryo, '_')[1]
            print(path_tmp, time_point)

            for cell_index in volume_counter:
                cell_name = name_list[cell_index]
                data_embryo_time_slices.at[time_point + '::' + cell_name, 'volume'] = volume_counter[cell_index]
                data_embryo_time_slices.at[time_point + '::' + cell_name, 'surface'] = surface_counter[cell_index]
                data_embryo_time_slices.at[time_point + '::' + cell_name, 'normalized_c'] = (volume_counter[
                                                                                                 cell_index] / 10000) ** (
                                                                                                    1 / 3)
    embryo_name = os.path.split(path_tmp)[-1]
    data_embryo_time_slices.to_csv(os.path.join(config.dir_my_data_volume_surface, embryo_name + '.csv'))
