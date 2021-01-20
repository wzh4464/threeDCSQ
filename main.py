import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh

import csv
import os
import config

import particular_func.SH_represention as SH_func

import particular_func.SH_analyses as SH_A_func

import functional_func.cell_func as cell_f
import functional_func.general_func as general_f
import pandas as pd

import functional_func.draw_func as draw_f

import test_scripts


def main():
    print("start cell shape analysis")

    # ------------------------------R fibonacci representation------------------------------------------------
    # img_1 = general_f.load_nitf2_img(os.path.join(config.dir_segemented, 'Embryo04_000_segCell.nii.gz'))
    # _ = cell_f.nii_get_cell_surface(img_1, save_name='Embryo04_000_segCell.nii.gz')

    # img_2 = general_f.show_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + 'Embryo04_001_segCell.nii.gz'))
    # R_func.build_R_array_for_embryo(128)
    # ---------------------------------------------------------------------------------------------------------

    # # ------------------------------calculate SHC for each cell ----------------------------------------------
    # path_tmp=r'./DATA/SegmentCellUnified04-20/Sample20LabelUnified'
    # for file_name in os.listdir(path_tmp):
    #     if os.path.isfile(os.path.join(path_tmp,file_name)):
    #         print(path_tmp)
    #         SH_func.get_SH_coeffient_from_surface_points(embryo_path=path_tmp, sample_N=100, lmax=49,
    #                                                      file_name=file_name)
    # # -------------------------------------------------------------------------------------------------------

    # # ------------------------------do contraction with sh expand and shc expand------------------------------
    #
    # path_tmp = r'./DATA/Embryo04LabelUnified_C/'
    #
    # SH_func.get_SH_coeffient_from_surface_points(embryo_path=path_tmp, sample_N=100, lmax=49,
    #                                              file_name='Embryo04_009_segCell.nii.gz')
    # p = multiprocessing.Process(target=SH_A_func.analysis_calculate_error_contrast,
    #                             args=(path_tmp, 'Embryo04_009_segCell.nii.gz', 'draw',))
    # p.start()
    #
    # SH_A_func.analysis_calculate_error_contrast(embryo_path=config.dir_segemented_tmp1,
    #                                             file_name='Embryo04_009_segCell.nii.gz', behavior='draw')
    #
    # # ---------------------------------------------------------------------------------------------------------

    # # ------------------------------calculate volume and surface of the cells of one embryo ----------------
    # path_tmp = r'./DATA/SegmentCellUnified04-20/Sample20LabelUnified'
    # cell_f.count_volume_surface_normalization_tocsv(path_tmp)
    # # -----------------------------------------------------------------------------------------------------

    # SH_A_func.analysis_time_domain_k_means(embryo_path=config.dir_segemented_tmp1,l_degree=10)

    # =======we always use l_degree= 25 coefficients are 676 to analyse===============
    path_tmp = r'./DATA/SegmentCellUnified04-20/Sample05LabelUnified'
    # SH_A_func.analysis_SHPCA_One_embryo(embryo_path=path_tmp, l_degree=25, is_show_PCA=True)

    # draw_f.draw_comparison_SHcPCA_SH(embryo_path=path_tmp, l_degree=25)

    SH_A_func.analysis_SHcPCA_maximum_clustering(embryo_path=path_tmp, l_degree=25)


if __name__ == '__main__':
    main()
    # print(__name__)
