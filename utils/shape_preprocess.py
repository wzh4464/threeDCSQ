#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import dependency library

from scipy import ndimage
import numpy as np
from skimage.measure import mesh_surface_area


# import user defined library



def export_dia_cell_surface_points_json(volume):
    """
    json format
    dictionary {key:cell_number,value: list[surface point string: x_y_z]}
    :param volume:
    :return:
    """
    surface_dict = {}
    cell_number = np.unique(volume)
    # cell_number.remove(0)
    for cell_idx in cell_number:
        if cell_idx != 0:
            surface_mask = np.logical_xor(ndimage.binary_dilation(volume == cell_idx), (volume == cell_idx))
            point_position_x, point_position_y, point_position_z = np.where(surface_mask == True)
            surface_points_list = []
            for i in range(len(point_position_x)):
                surface_points_list.append(
                    str(point_position_x[i]) + '_' + str(point_position_y[i]) + '_' + str(point_position_z[i]))
            surface_dict[str(cell_idx)] = surface_points_list
            print(str(cell_idx), 'surface points', len(point_position_x))
    return surface_dict


def export_dia_cell_points_json(volume):
    """
    FOR THE alpha shape!!
    json format
    dictionary {key:cell_number,value: list[surface point string: x_y_z]}
    :param volume:
    :return:
    """
    cell_dict = {}
    cell_number = np.unique(volume)
    # cell_number.remove(0)
    for cell_idx in cell_number:
        if cell_idx != 0:
            cell_point_mask = ndimage.binary_dilation(volume == cell_idx)
            point_position_x, point_position_y, point_position_z = np.where(cell_point_mask == True)
            cell_points_list = []
            for i in range(len(point_position_x)):
                cell_points_list.append((point_position_x[i],point_position_y[i],point_position_z[i]))
            cell_dict[cell_idx] = cell_points_list
            print(cell_idx, 'surface points', len(point_position_x))
    return cell_dict


# def get_contact_area(volume):
#     '''
#     Get the contact volume surface of the segmentation. The segmentation results should be watershed segmentation results
#     with a ***watershed line***.
#
#     json format:
#     dictionary {key:cell1_number_cell2_number,value: list[contact surface point string: x_y_z]}
#
#     :param volume: segmentation result
#     :return boundary_elements_uni: pairs of SegCell which contacts with each other
#     :return contact_area: the contact surface area corresponding to that in the the boundary_elements.
#     '''
#
#     # get the boundary for the whole embryo, including the boundary between cells
#     cell_mask = volume != 0
#     boundary_mask = (cell_mask == 0) & ndimage.binary_dilation(cell_mask)
#     [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
#     boundary_elements = []
#
#     # find boundary between cells
#     for (x, y, z) in zip(x_bound, y_bound, z_bound):
#         neighbors = volume[np.ix_(range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2))]
#         neighbor_labels = list(np.unique(neighbors))
#         neighbor_labels.remove(0)
#         if len(neighbor_labels) == 2:  # contact between two cells
#             boundary_elements.append(neighbor_labels)
#     # cell contact pairs
#     cell_contact_pairs = list(np.unique(np.array(boundary_elements), axis=0))
#     contact_area = []
#     cell_conatact_pair_renew = []
#     contact_points_dict = {}
#     for (label1, label2) in cell_contact_pairs:
#         contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1),
#                                       ndimage.binary_dilation(volume == label2))
#         contact_mask = np.logical_and(contact_mask, boundary_mask)
#         if contact_mask.sum() > 4:
#
#             verts, faces, _, _ = marching_cubes(contact_mask)
#             area = mesh_surface_area(verts, faces) / 2
#             contact_area.append(area)
#             cell_conatact_pair_renew.append((label1, label2))
#
#             point_position_x, point_position_y, point_position_z = np.where(contact_mask == True)
#
#             contact_points_list = []
#             for i in range(len(point_position_x)):
#                 contact_points_list.append((point_position_x[i],point_position_y[i],point_position_z[i]))
#             # str_key = str(label1) + '_' + str(label2)
#             print((label1, label2))
#             contact_points_dict[(label1, label2)] = contact_points_list
#     # print(contact_points_dict)
#     return cell_conatact_pair_renew, contact_area, contact_points_dict
