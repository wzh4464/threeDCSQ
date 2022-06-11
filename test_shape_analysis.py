import os
import pickle

from time import time

from static import config
from utils.general_func import load_nitf2_img
from utils.shape_preprocess import get_contact_area


def calculate_cell_contact_points():
    embryo_names = [str(i).zfill(2) for i in range(4, 21)]

    for embryo_name in embryo_names:
        # ------------------------calculate surface points using dialation for each cell --------------------
        path_tmp = config.data_path + r'Segmentation Results\SegmentedCell/Sample' + embryo_name + 'LabelUnified'
        for file_name in os.listdir(path_tmp):
            if os.path.isfile(os.path.join(path_tmp, file_name)):
                t0 = time()

                print(path_tmp, file_name)

                this_img = load_nitf2_img(os.path.join(path_tmp, file_name))

                img_arr = this_img.get_fdata().astype(int)

                _, _, contact_points_dict = get_contact_area(img_arr)

                contact_saving_path = os.path.join(config.data_cell_mesh_and_contact ,'Sample' + embryo_name)
                if not os.path.exists(contact_saving_path):
                    os.mkdir(contact_saving_path)

                with open(os.path.join(contact_saving_path, file_name.split('.')[0] + '.pickle'), 'wb+') as handle:
                    pickle.dump(contact_points_dict, handle,protocol=pickle.HIGHEST_PROTOCOL)

                # load()
                # you can find out the method about loading json in python
                print("done in %0.3fs" % (time() - t0))

                # print(contact_points_dict)

        # -------------------------------------------------------------------------------------------------------
#
#
# def calculate_cell_points():
#     embryo_names = [str(i).zfill(2) for i in range(4, 21)]
#
#     for embryo_name in embryo_names:
#         # ------------------------calculate surface points using dialation for each cell --------------------
#         path_tmp = config.data_path + r'Segmentation Results\SegmentedCell/Sample' + embryo_name + 'LabelUnified'
#         for file_name in os.listdir(path_tmp):
#             if os.path.isfile(os.path.join(path_tmp, file_name)):
#                 t0 = time()
#                 print(path_tmp, file_name)
#                 this_img = load_nitf2_img(os.path.join(path_tmp, file_name))
#                 img_arr = this_img.get_fdata().astype(int)
#
#                 cell_points = export_dia_cell_points_json(img_arr)
#
#                 dia_cell_saving = os.path.join(config.data_path + r'cell_dia_points', 'Sample' + embryo_name)
#                 if not os.path.exists(dia_cell_saving):
#                     os.mkdir(dia_cell_saving)
#
#                 with open(os.path.join(dia_cell_saving, file_name.split('.')[0] + '.json'), 'w') as fp:
#                     json.dump(cell_points, fp)
#
#                 print("done in %0.3fs" % (time() - t0))
#
#
# def calculate_cell_surface_points():
#     embryo_names = [str(i).zfill(2) for i in range(4, 21)]
#
#     for embryo_name in embryo_names:
#         # ------------------------calculate surface points using dialation for each cell --------------------
#         path_tmp = config.data_path + r'Segmentation Results\SegmentedCell/Sample' + embryo_name + 'LabelUnified'
#         for file_name in os.listdir(path_tmp):
#             if os.path.isfile(os.path.join(path_tmp, file_name)):
#                 t0 = time()
#                 print(path_tmp, file_name)
#                 this_img = load_nitf2_img(os.path.join(path_tmp, file_name))
#                 img_arr = this_img.get_fdata().astype(int)
#
#                 cell_points = export_dia_cell_surface_points_json(img_arr)
#
#                 dia_surface_saving = os.path.join(config.data_path + r'cell_dia_surface', 'Sample' + embryo_name)
#                 if not os.path.exists(dia_surface_saving):
#                     os.mkdir(dia_surface_saving)
#
#                 with open(os.path.join(dia_surface_saving, file_name.split('.')[0] + '.json'), 'w') as fp:
#                     json.dump(cell_points, fp)
#
#                 print("done in %0.3fs" % (time() - t0))
#
#
# def display_contact_points():
#     # Sample20,ABplpapapa,150
#     # Sample20,Dpaap,158
#     # Sample20,ABalaapa,078
#     # Sample20,ABa,005
#     # Sample20,MSp,035
#     print('waiting type you input: samplename and timepoints for embryogenesis')
#     embryo_name, cell_name, tp = str(input()).split(',')
#
#     num_cell_name, cell_num = get_cell_name_affine_table()
#     this_cell_keys = cell_num[cell_name]
#
#     with open(os.path.join(r'./DATA/cshaper_contact_data', embryo_name + '_' + tp + '_segCell.json'), ) as fp:
#         data = json.load(fp)
#     display_key_list = []
#     for idx in data.keys():
#         print(idx, len(data[idx]))
#         # if re.match('^' + str(this_cell_keys) + '_\d', idx) or re.match('\d_' + str(this_cell_keys) + '$', idx):
#         label1_2 = idx.split('_')
#         if str(this_cell_keys) in label1_2:
#             display_key_list.append(idx)
#
#     # fig_contact_info = plt.figure()
#     # plt.axis('off')
#     item_count = 1
#     print('contact number', len(display_key_list))
#     for idx in display_key_list:
#         if item_count > 9:
#             break
#
#         # if len(data[idx]) < 30:
#         #     continue
#         draw_points_list = []
#         print(idx)
#         for item_str in data[idx]:
#             x, y, z = item_str.split('_')
#             x, y, z = int(x), int(y), int(z)
#             draw_points_list.append([x, y, z])
#
#         [x_tmp, y_tmp, z_tmp] = np.amax(np.array(draw_points_list), axis=0)
#         contact_mask = np.zeros((x_tmp * 2, y_tmp * 2, z_tmp * 2))
#         for [x_tmp, y_tmp, z_tmp] in draw_points_list:
#             contact_mask[x_tmp, y_tmp, z_tmp] = True
#         verts, faces, _, _ = marching_cubes(contact_mask)
#         contact_mesh = o3d.geometry.TriangleMesh(o3d.cpu.pybind.utility.Vector3dVector(verts),
#                                                  o3d.cpu.pybind.utility.Vector3iVector(faces))
#
#         print(idx, '  marching cubes method surface area:', mesh_surface_area(verts, faces) / 2)
#
#         contact_mesh.compute_vertex_normals()
#         o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
#
#         # a = idx.split('_')
#         # a.remove(str(this_cell_keys))
#         # contact_cell_num = int(a[0])
#         # ax = fig_contact_info.add_subplot(3, 3, item_count, projection='3d')
#         # draw_3D_points(np.array(draw_points_list), fig_name=cell_name + '_' + num_cell_name[contact_cell_num], ax=ax)
#
#         # dfs = pd.read_excel(static.cell_fate_path, sheet_name=None)['CellFate']
#         # fate_cell = dfs[dfs['Name'] == cell_name + '\'']['Fate'].values[0].split('\'')[0]
#
#         item_count += 1
#     # plt.show()
#
#
# def display_contact_alpha_surface():
#     # Sample20,ABplpapapa,150
#     # Sample20,ABalaapa,078
#     # Sample20,ABa,005
#     # Sample20,MSp,035
#     print('waiting type you input: sample name and time points for embryogenesis analysis')
#     embryo_name, cell_name, tp = str(input()).split(',')
#
#     num_cell_name, cell_num = get_cell_name_affine_table()
#     this_cell_keys = cell_num[cell_name]
#
#     # -------getting all data points including dilation  points -- for generate alpha shape--------
#     with open(os.path.join(r'./DATA/cell_dia_points', embryo_name + '_' + tp + '_segCell.json')) as fp:
#         cell_data = json.load(fp)
#     cell_points_building_as = []
#     # print(cell_data.keys())
#     for item_str in cell_data[str(this_cell_keys)]:
#         x, y, z = item_str.split('_')
#         x, y, z = float(x) + uniform(0, 0.001), float(y) + uniform(0, 0.001), float(
#             z) + uniform(0, 0.001)
#         cell_points_building_as.append([x, y, z])
#     cell_points_building_as = np.array(cell_points_building_as)
#     # print(cell_points_building_as)
#     m_mesh = generate_alpha_shape(cell_points_building_as)
#     # ---------------------------finished generating alpha shape -------------------------------
#
#     with open(os.path.join(r'./DATA/cell_dia_surface', embryo_name, embryo_name + '_' + tp + '_segCell.json'),
#               'rb') as fp:
#         surface_data = json.load(fp)
#
#     with open(os.path.join(r'./DATA/cshaper_contact_data', embryo_name, embryo_name + '_' + tp + '_segCell.json'),
#               'rb') as fp:
#         surface_contact_data = json.load(fp)
#
#     get_contact_surface_mesh(this_cell_keys, surface_data, surface_contact_data, m_mesh)
#

if __name__ == "__main__":
    calculate_cell_contact_points()