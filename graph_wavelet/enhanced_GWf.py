import igraph
import pandas as pd
import numpy as np
import sympy
import math


# from utils.general_func import read_csv_to_df


def embryo_enhanced_graph_wavelet(nei_cell_feature,cell_nei_matrix,phi_j_h_list,C_j_vk):

    fea_list=[]

    idx_fea=0
    for ith,ith_list in enumerate(cell_nei_matrix):
        tmp_average_fea_list=[]
        for _ in ith_list:
            tmp_average_fea_list.append(nei_cell_feature[idx_fea])
            idx_fea+=1
        # print(ith_list)
        # print(ith)
        fea_list.append(list(phi_j_h_list[ith]*np.mean(np.array(tmp_average_fea_list),axis=0)))

    enhanced_fea=C_j_vk*np.sum(np.array(fea_list),axis=0)
    # print(nei_cell_feature[0])
    # print(enhanced_fea)
    return enhanced_fea




def phi_j_h_wavelet(j,h,wavelet='Haar'):
    x = sympy.symbols('x')
    # Haar function
    f = sympy.Piecewise((1, x < 0.5), (-1, x>= 0.5))
    if wavelet == 'MexicanHat':
        f = 2 / math.sqrt(3) * (math.pi) ** (-1 / 4) * (1 - x ** 2) * (math.e) ** (-x ** 2 / 2)

    return sympy.integrate(f,(x,h/(j+1),(h+1)/(j+1)))

def get_kth_neighborhood_graph(star,hop,graph):
    neighborhood_dict={}
    for i in range(hop+1):
        if i==0:
            neighborhood_dict[i]= {star}
        else:
            neighborhood_dict[i]=set()
            for pre_node in neighborhood_dict[i-1]:
                neighborhood_dict[i].union(set(pre_node.neighbors()))
    return neighborhood_dict

# if __name__ == "__main__":
#     embryo_enhanced_graph_wavelet(None,None,wavelet='MexicanHat')