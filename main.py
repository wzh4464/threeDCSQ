import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh

import csv
import os
import config

import particular_func.SH_represention as SH_func

import particular_func.SH_analyses as SH_A_func


def main():
    print("start cell shape analysis")

    # img_1 = general_f.load_nitf2_img(os.path.join(config.dir_segemented, 'Embryo04_000_segCell.nii.gz'))
    # _ = cell_f.nii_get_cell_surface(img_1, save_name='Embryo04_000_segCell.nii.gz')

    # img_2 = general_f.show_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + 'Embryo04_001_segCell.nii.gz'))
    # R_func.build_R_array_for_embryo(128)

    # for file_name in os.listdir(config.dir_segemented_tmp1):
    #     if os.path.isfile(os.path.join(config.dir_segemented_tmp1,file_name)):
    #         SH_func.get_SH_coeffient_from_surface_points(embryo_path=config.dir_segemented_tmp1, sample_N=100, lmax=49,
    #                                                      file_name=file_name)

    SH_A_func.analysis_with_img(embryo_path=config.dir_segemented_tmp1, file_name='Embryo04_053_segCell.nii.gz')


if __name__ == '__main__':
    main()
    # print(__name__)
