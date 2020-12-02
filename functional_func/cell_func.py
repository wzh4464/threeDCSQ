import nibabel as nib
import numpy as np
import pandas as pd
import config

from scipy import ndimage
from collections import Counter
from nibabel.viewers import OrthoSlicer3D
import csv

import os


def get_cell_name_affine_table():
    """

    :return: a set of NO. to name LIST and name to NO. DICTIONARY:
    zero first, but actually there are no zero, remember to plus 1
    """
    name_list = []
    NO_dic = {}
    with open(os.path.join('./DATA', 'name_dictionary.csv'), newline='') as name_table:
        name_reader = csv.reader(name_table, delimiter=' ', quotechar='|')
        name_list.append('background')
        for row in name_reader:
            NO_dic[row[0].split(',')[1]] = len(name_list)
            name_list.append(row[0].split(',')[1])

    return name_list, NO_dic


def nii_get_cell_surface(this_image, save_name=None):
    img_arr = this_image.get_data()
    # ---------------- erosion ----------------
    struct_element = ndimage.generate_binary_structure(3, -1)

    # with the original image data
    img_arr_erosion = ndimage.grey_erosion(img_arr, footprint=struct_element)
    # img_df_erosion = pd.DataFrame(img_arr_erosion[100:150, 150:200, 100])
    # print(img_df_erosion)

    surface_data_result = img_arr - img_arr_erosion

    membrane_img = nib.Nifti2Image(surface_data_result.astype(np.int16), np.diag([1, 1, 1, 1]))
    # OrthoSlicer3D(membrane_img.dataobj).show()

    if save_name is not None:
        # nib.save(membrane_img, os.path.join(config.dir_my_data, 'membrane' + save_name))
        membrane_img.to_filename(os.path.join(config.dir_my_data, 'membrane' + save_name))
    return membrane_img


# np.set_printoptions(threshold=100000)

def nii_count_volume_surface(this_image):
    '''
    :param this_image: the nii image from 3D image, count volume and surface
    :return: nil
    '''

    # OrthoSlicer3D(this_image.dataobj).show()
    img_arr = this_image.get_data()
    img_arr_shape = img_arr.shape
    img_arr_count = np.prod(img_arr_shape)

    # print(img_arr.shape)
    # img_df = pd.DataFrame(img_arr[100:150, 150:200, 100])
    # print(img_df)

    struc_element = ndimage.generate_binary_structure(3, -1)

    # ---------------- dilation ----------
    # with the original image data
    # print(struc_element)
    # img_arr_dilation = ndimage.grey_dilation(img_arr, footprint=struc_element)
    # img_df_dilation = pd.DataFrame(img_arr_dilation[100:150, 150:200, 100])
    # print(img_df_dilation)
    # img_dilation = nib.Nifti2Image(img_arr_dilation, np.diag([1, 1, 1, 1]))
    # OrthoSlicer3D(img_dilation.dataobj).show()

    # ---------------- erosion ----------------
    # with the original image data
    img_arr_erosion = ndimage.grey_erosion(img_arr, footprint=struc_element)
    # img_df_erosion = pd.DataFrame(img_arr_erosion[100:150, 150:200, 100])
    # print(img_df_erosion)

    surface_data_result = img_arr - img_arr_erosion

    cnt1 = Counter(np.reshape(img_arr, img_arr_count))
    # print(cnt1)
    cnt2 = Counter(np.reshape(surface_data_result, img_arr_count))
    # print(cnt)

    return cnt1, cnt2


def nii_count_contact_surface(this_image):
    img_arr = this_image.get_data()
    img_arr_shape = img_arr.shape
    img_arr_count = np.prod(img_arr_shape)
    cnt = Counter(np.reshape(img_arr, img_arr_count))
    print(type(cnt))
