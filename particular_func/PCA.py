import particular_func.SH_analyses as sh_analysis
from pyshtools import SHCoeffs
from functional_func.draw_func import draw_3D_points
from matplotlib import pyplot as plt
import functional_func.general_func as general_f


def draw_PCA(sh_PCA):
    sh_PCA_mean = sh_PCA.mean_
    component_index = 0
    for component in sh_PCA.components_:
        print('components  ', component[:20])
        # print("inverse log::",inverse_log_expand[:50])

        fig = plt.figure()

        shc_instance_3 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + -5 * component)))
        shc_instance_2 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + -3 * component)))
        shc_instance_1 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + -1 * component)))
        shc_instance_0 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + 0 * component)))
        shc_instance1 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + 1 * component)))
        shc_instance2 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + 3 * component)))
        shc_instance3 = SHCoeffs.from_array(sh_analysis.collapse_flatten_clim(list(sh_PCA_mean + 5 * component)))

        sh_reconstruction = sh_analysis.do_reconstruction_for_SH(30, shc_instance_3)
        axes_tmp = fig.add_subplot(2, 3, 1, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-5),
                       ax=axes_tmp)
        sh_reconstruction = sh_analysis.do_reconstruction_for_SH(30, shc_instance_2)
        axes_tmp = fig.add_subplot(2, 3, 2, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-3),
                       ax=axes_tmp)
        sh_reconstruction = sh_analysis.do_reconstruction_for_SH(30, shc_instance_1)
        axes_tmp = fig.add_subplot(2, 3, 3, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(-1),
                       ax=axes_tmp)

        sh_reconstruction = sh_analysis.do_reconstruction_for_SH(30, shc_instance1)
        axes_tmp = fig.add_subplot(2, 3, 4, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(1), ax=axes_tmp)
        sh_reconstruction = sh_analysis.do_reconstruction_for_SH(30, shc_instance2)
        axes_tmp = fig.add_subplot(2, 3, 5, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(3), ax=axes_tmp)
        sh_reconstruction = sh_analysis.do_reconstruction_for_SH(30, shc_instance3)
        axes_tmp = fig.add_subplot(2, 3, 6, projection='3d')
        draw_3D_points(sh_reconstruction, fig_name=str(component_index) + 'Delta ' + str(5), ax=axes_tmp)

        plt.show()

        component_index += 1


def read_PCA_file(PCA_file_path):
    PCA_df = general_f.read_csv_to_df(PCA_file_path)
    pca_means = PCA_df.loc['mean'][1:]
    PCA_df.drop(index='mean', inplace=True)
    pca_explained = PCA_df['explained_variation']
    PCA_df.drop(columns='explained_variation', inplace=True)

    return pca_means, pca_explained, PCA_df

