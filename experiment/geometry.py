import numpy as np
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def read_solid_file(solid_file_path):
    with open(solid_file_path) as f:
        content = [i.strip() for i in f.readlines()]

    vertices = []
    faces = []

    reading_type = 0
    for index, line_item in enumerate(content):
        if line_item == 'VERTICES:':
            continue
        elif content[index - 1] == 'VERTICES:' or reading_type == 1:
            reading_type = 1

            list_tmp = []
            for item in line_item.split(' '):
                # print(type(item))
                list_tmp.append(float(item))
            vertices.append(list_tmp)

            if content[index + 1] == 'FACES:':
                reading_type = 2
        elif line_item != 'FACES:' and reading_type == 2:
            list_tmp = []
            for item in line_item.split(' '):
                list_tmp.append(int(item))
            faces.append(list_tmp)
    return vertices, faces


def get_face_equation(points):
    """

    :param points: give me a list at least three points
    :return:ax+by+cz+d=0
    https://blog.csdn.net/u012463389/article/details/50755220

    """
    v1 = points[0]
    v2 = points[1]
    v3 = points[2]
    a = (v2[1] - v1[1]) * (v3[2] - v1[2]) - (v3[1] - v1[1]) * (v2[2] - v1[2])
    b = (v2[2] - v1[2]) * (v3[0] - v1[0]) - (v3[2] - v1[2]) * (v2[0] - v1[0])
    c = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])

    d = -a * v1[0] - b * v1[1] - c * v1[2]

    return a, b, c, d


def random_sample_on_surface(vertices_2d, vertices_order):
    vertices_2d = np.array(vertices_2d)
    # print(vertices_2d)
    # print(vertices_order)
    x_max = max(vertices_2d[:, 0])
    x_min = min(vertices_2d[:, 0])
    y_max = max(vertices_2d[:, 1])
    y_min = min(vertices_2d[:, 1])
    z_max = max(vertices_2d[:, 2])
    z_min = min(vertices_2d[:, 2])

    a, b, c, d = get_face_equation(vertices_2d[:3])

    face_sample = []
    points_for_polygon = []

    # print('plane a b c d ', a, b, c, d)

    if c == 0:
        # b!=0 (a==0 and c==0) or (c==0)
        # generate point on x z
        for index_tmp, _ in enumerate(vertices_order):
            points_for_polygon.append((vertices_2d[index_tmp][0], vertices_2d[index_tmp][2]))
        # print('generating polygon on xz plane===>', points_for_polygon)
        polygon = Polygon(points_for_polygon)
        # print('finish generating polygon on xz plane, begin to generate uniform points on xy plane')
        while len(face_sample) <= 1000:
            x = random.uniform(x_min, x_max)
            z = random.uniform(z_min, z_max)
            # print(x,y)
            if polygon.contains(Point(x, z)):
                y = (-a * x - c * z - d) / b
                face_sample.append([x, y, z])
    elif b == 0 and c == 0:
        # a!=0
        # generate point on y z
        for index_tmp, _ in enumerate(vertices_order):
            points_for_polygon.append((vertices_2d[index_tmp][1], vertices_2d[index_tmp][2]))
        # print('generating polygon on yz plane===>', points_for_polygon)
        polygon = Polygon(points_for_polygon)
        # print('finish generating polygon on yz plane, begin to generate uniform points on xy plane')
        while len(face_sample) <= 1000:
            y = random.uniform(y_min, y_max)
            z = random.uniform(z_min, z_max)
            # print(x,y)
            if polygon.contains(Point(y, z)):
                x = (-b * y - c * z - d) / a
                face_sample.append([x, y, z])
    else:
        # generate point on x y
        for index_tmp, _ in enumerate(vertices_order):
            points_for_polygon.append((vertices_2d[index_tmp][0], vertices_2d[index_tmp][1]))
        # print('generating polygon on xy plane===>', points_for_polygon)
        polygon = Polygon(points_for_polygon)
        # print('finish generating polygon on xy plane, begin to generate uniform points on xy plane')
        while len(face_sample) <= 1000:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            # print(x,y)
            if polygon.contains(Point(x, y)):
                z = (-a * x - b * y - d) / c
                face_sample.append([x, y, z])
    # print('finish generate sample points')
    # z = (-a * x - b * y - d) / c
    return face_sample


def draw_3D_points_geometry(points_data, fig_name="DEFAULT", fig_size=(10, 10), ax=None):
    x = points_data[:, 0]  # first column of the 2D matrix
    y = points_data[:, 1]
    z = points_data[:, 2]

    if ax == None:
        fig = plt.figure(figsize=fig_size)
        ax = Axes3D(fig)
        ax.scatter3D(x, y, z, cmap='BuRd', marker='o')
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        ax.set_title(fig_name)

        plt.title(fig_name)
        plt.show()
    else:
        ax.scatter3D(x, y, z, cmap='BuRd', marker='o')
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        ax.set_title(fig_name)


def get_sample_on_geometric_object(tmp_path):
    vertices, faces = read_solid_file(tmp_path)
    # print(vertices, faces)
    shape_sample = []
    print('generating random points on this regular object surface')
    for index, _ in enumerate(tqdm(faces)):
        vertex_point_list = []
        # print('item tmp!!', index)
        for item_tmp in faces[index]:
            vertex_point_list.append(vertices[item_tmp])
        shape_sample = shape_sample + random_sample_on_surface(vertex_point_list, faces[index])
    # print(shape_sample)
    # draw_3D_points_geometry(np.array(shape_sample))
    return np.array(shape_sample)


regular_polyhedron_list = [
    'unit-regular-tetrahedron.solid',  # 四面体
    'unit-cube.solid',  # 六面体
    'unit-regular-octahedron.solid',  # 八面体
    'unit-regular-dodecahedron.solid',  # 十二面体
    'unit-regular-icosahedron.solid',  # 20面体
]

if __name__ == "__main__":
    print('geometric test')
    # Create a new plot
    tmp_path = r'./DATA/template_shape_stl/unit-regular-tetrahedron.solid'
    get_sample_on_geometric_object(tmp_path)
