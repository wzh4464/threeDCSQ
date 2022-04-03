import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift

from static.dict import cell_fate_map
from utils.machine_learning import cluster_acc


def cluster_lifespan_for_embryo(df_avg_lifespan_feature_value,y_fate, cluter_dimension, cluster_num_predict=8 ):
    """
    clustering with the feature vector and PCA feature vector
    :param df_feature_vector:
    :param PCA_value_arr:
    :param tree_this_embryo:
    :param this_cell_fate_dict:
    :param time_limit_minutes_start:
    :return:
    """
    # =======================================start cluster==================================================
    y_fate = np.array(y_fate)

    randomlist = np.random.randint(low=0, high=cluster_num_predict, size=len(y_fate))

    print(np.unique(y_fate, return_counts=True))
    print(np.unique(randomlist, return_counts=True))
    print('Random cluster', cluster_acc(y_fate, randomlist, cluster_num_predict))

    # ------------KMEANS --------------------------------------------
    cluster_arr = df_avg_lifespan_feature_value.values[:, :cluter_dimension]
    y_kmeans_estimation = KMeans(n_clusters=cluster_num_predict, tol=1e-6).fit_predict(cluster_arr)
    print('Kmeans', np.unique(y_kmeans_estimation, return_counts=True))
    print(cluster_acc(y_fate, y_kmeans_estimation, cluster_num_predict))
    # ------Mean shift , a centroid clustering algorithms
    meanshift = MeanShift(bandwidth=0.6, cluster_all=False).fit_predict(cluster_arr)
    print('Mean shift', np.unique(meanshift, return_counts=True), cluster_acc(y_fate, meanshift, cluster_num_predict))

    # DBSCN have been proved useless
    # y_fea_agglo = AgglomerativeClustering(n_clusters=cluster_num_predict).fit_predict(cluster_arr)
    # print('ward',cluster_acc(y_fate, y_fea_agglo, cluster_num_predict),np.unique(y_fea_agglo, return_counts=True))
    y_fea_agglo = AgglomerativeClustering(n_clusters=cluster_num_predict, linkage='average').fit_predict(
        cluster_arr)
    print('average', cluster_acc(y_fate, y_fea_agglo, cluster_num_predict),
          np.unique(y_fea_agglo, return_counts=True))
    y_fea_agglo = AgglomerativeClustering(n_clusters=cluster_num_predict, linkage='complete').fit_predict(
        cluster_arr)
    print('maximum', cluster_acc(y_fate, y_fea_agglo, cluster_num_predict),
          np.unique(y_fea_agglo, return_counts=True))
    # y_fea_agglo = AgglomerativeClustering(n_clusters=cluster_num_predict, linkage='single').fit_predict(cluster_arr)
    # print('single', cluster_acc(y_fate, y_fea_agglo, cluster_num_predict),
    #       np.unique(y_fea_agglo, return_counts=True))

    print('Real distribution', np.unique(y_fate, return_counts=True))
    # =======================================stop cluster==================================================

