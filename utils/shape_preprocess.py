# import dependency library
import sys
import shutil
from tqdm import tqdm
from scipy import ndimage
import multiprocessing as mp
import numpy as np
from skimage.measure import marching_cubes_lewiner, mesh_surface_area


# import user defined library

def export_points_json(volume):
    surface_dict = {}
    cell_number = np.unique(volume)
    # cell_number.remove(0)
    for cell_idx in cell_number:
        if cell_idx != 0:
            # surface_mask = np.logical_xor(ndimage.binary_dilation(volume == cell_idx), (volume == cell_idx))
            surface_mask = ndimage.binary_dilation(volume == cell_idx)
            point_position_x, point_position_y, point_position_z = np.where(surface_mask == True)
            surface_points_list = []
            for i in range(len(point_position_x)):
                surface_points_list.append(
                    str(point_position_x[i]) + '_' + str(point_position_y[i]) + '_' + str(point_position_z[i]))
            surface_dict[str(cell_idx)] = surface_points_list
            print(cell_idx,'surface points', len(point_position_x))
    return surface_dict


def get_contact_area(volume):
    '''
    Get the contact volume surface of the segmentation. The segmentation results should be watershed segmentation results
    with a ***watershed line***.
    :param volume: segmentation result
    :return boundary_elements_uni: pairs of SegCell which contacts with each other
    :return contact_area: the contact surface area corresponding to that in the the boundary_elements.
    '''

    cell_mask = volume != 0
    boundary_mask = (cell_mask == 0) & ndimage.binary_dilation(cell_mask)
    [x_bound, y_bound, z_bound] = np.nonzero(boundary_mask)
    boundary_elements = []
    for (x, y, z) in zip(x_bound, y_bound, z_bound):
        neighbors = volume[np.ix_(range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2))]
        neighbor_labels = list(np.unique(neighbors))
        neighbor_labels.remove(0)
        if len(neighbor_labels) == 2:
            boundary_elements.append(neighbor_labels)
    boundary_elements_uni = list(np.unique(np.array(boundary_elements), axis=0))
    contact_area = []
    boundary_elements_uni_new = []
    contact_points_dict = {}
    for (label1, label2) in boundary_elements_uni:
        contact_mask = np.logical_and(ndimage.binary_dilation(volume == label1),
                                      ndimage.binary_dilation(volume == label2))
        contact_mask = np.logical_and(contact_mask, boundary_mask)
        if contact_mask.sum() > 4:

            verts, faces, _, _ = marching_cubes_lewiner(contact_mask)
            area = mesh_surface_area(verts, faces) / 2
            contact_area.append(area)
            boundary_elements_uni_new.append((label1, label2))

            point_position_x, point_position_y, point_position_z = np.where(contact_mask == True)

            contact_points_list = []
            for i in range(len(point_position_x)):
                contact_points_list.append(
                    str(point_position_x[i]) + '_' + str(point_position_y[i]) + '_' + str(point_position_z[i]))
            str_key = str(label1) + '_' + str(label2)
            contact_points_dict[str_key] = contact_points_list
    return boundary_elements_uni_new, contact_area, contact_points_dict
