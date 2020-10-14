import test_scripts
import particular_func.R_matrix_represention as R_func




def main():
    print("start cell shape analysis")

    # img_1 = general_f.load_nitf2_img(os.path.join(config.dir_segemented, 'Embryo04_000_segCell.nii.gz'))
    # _ = cell_f.nii_get_cell_surface(img_1, save_name='Embryo04_000_segCell.nii.gz')

    # img_2 = general_f.show_nitf2_img(os.path.join(config.dir_my_data, 'membrane' + 'Embryo04_001_segCell.nii.gz'))
    R_func.build_R_array_for_embryo(100)



if __name__ == '__main__':
    main()
    # print(__name__)
