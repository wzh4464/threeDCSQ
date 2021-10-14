import functional_func.general_func as general_f

import config
import os


def read_shc(embryo_path, degree=25):
    embryo_name = os.path.basename(embryo_path).split('.')[0]
    df_shc = general_f.read_csv_to_df(
        os.path.join(config.dir_my_data_SH_time_domain_csv, embryo_name + '_l_' + str(degree) + '.csv'))
    return df_shc


def read_shc_norm(embryo_path, degree=25):
    embryo_name = os.path.basename(embryo_path).split('.')[0]
    df_shc = general_f.read_csv_to_df(
        os.path.join(config.dir_my_data_SH_time_domain_csv, embryo_name + '_l_' + str(degree) + '_norm.csv'))
    return df_shc


def read_shcpca(embryo_path, degree=17):
    embryo_name = os.path.basename(embryo_path).split('.')[0]
    df_shc = general_f.read_csv_to_df(
        os.path.join(config.dir_my_data_SH_PCA_csv, embryo_name + '_SHcPCA' + str(degree ** 2) + '.csv'))
    return df_shc


def read_shcpca_norm(embryo_path, degree=17):
    embryo_name = os.path.basename(embryo_path).split('.')[0]
    df_shc = general_f.read_csv_to_df(
        os.path.join(config.dir_my_data_SH_PCA_csv, embryo_name + '_SHcPCA' + str(degree ** 2) + '_norm.csv'))
    return df_shc


