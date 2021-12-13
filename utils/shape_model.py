# import dependency library


import open3d as o3d
import numpy as np
import PIL.Image
import IPython.display
import os
import urllib.request
import tarfile
import gzip
import zipfile
import shutil
import sys


# import user defined library


def get_armadillo_mesh():
    armadillo_path = r"../test_data/Armadillo.ply"
    if not os.path.exists(armadillo_path):
        print("downloading armadillo mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
        urllib.request.urlretrieve(url, armadillo_path + ".gz")
        print("extract armadillo mesh")
        with gzip.open(armadillo_path + ".gz", "rb") as fin:
            with open(armadillo_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        os.remove(armadillo_path + ".gz")
    mesh = o3d.io.read_triangle_mesh(armadillo_path)
    mesh.compute_vertex_normals()
    return mesh


def get_bunny_mesh():
    bunny_path = r"../test_data/Bunny.ply"
    if not os.path.exists(bunny_path):
        print("downloading bunny mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        urllib.request.urlretrieve(url, bunny_path + ".tar.gz")
        print("extract bunny mesh")
        with tarfile.open(bunny_path + ".tar.gz") as tar:
            tar.extractall(path=os.path.dirname(bunny_path))
        shutil.move(
            os.path.join(
                os.path.dirname(bunny_path),
                "bunny",
                "reconstruction",
                "bun_zipper.ply",
            ),
            bunny_path,
        )
        os.remove(bunny_path + ".tar.gz")
        shutil.rmtree(os.path.join(os.path.dirname(bunny_path), "bunny"))
    mesh = o3d.io.read_triangle_mesh(bunny_path)
    print(np.asarray(mesh.vertices).tolist())
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.vertices))

    return mesh


def generate_alpha_shape(points_np: np.array, displaying: bool = False, alpha_value: float = 0.88,
                         view_name: str = 'default'):
    '''

    :param points_np:
    :param displaying:
    :param alpha_value: bigger than 0.85, sqrt(3)/2, the delaunay triangulation would be successful.
        Because of the random bias, we need to add more than 0.01 to the 0.866
    :param view_name:
    :return:
    '''
    pcd = o3d.geometry.PointCloud()
    # print(np.mean(points_np, axis=0))
    # print(points_np)
    pcd.points = o3d.utility.Vector3dVector(points_np)
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd])

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)

    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # o3d.visualization.draw_geometries([pcd, tetra_mesh])

    # pcd.estimate_normals()
    # radii = [0.1,0.5, 1, 2, 4,8,16,32]
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector(radii))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)
    if displaying:

        print('edge manifold', mesh.is_edge_manifold(allow_boundary_edges=True))
        print('edge manifold boundary', mesh.is_edge_manifold(allow_boundary_edges=False))
        print('vertex manifold', mesh.is_vertex_manifold())
        # print('self intersection ', mesh.is_self_intersecting())
        # print('watertight', mesh.is_watertight())
        print(f"alpha={alpha_value:.3f}")

        # mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        print(len(np.asarray(mesh.get_non_manifold_edges())))
        vertex_colors = 0.75 * np.ones((len(mesh.vertices), 3))
        for boundary in mesh.get_non_manifold_edges():
            for vertex_id in boundary:
                vertex_colors[vertex_id] = [1, 0, 0]
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        print(len(np.asarray(mesh.get_non_manifold_vertices())))

        o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                                          window_name=view_name)
    return mesh


if __name__ == '__main__':
    # mesh = get_bunny_mesh()
    # pcd = mesh.sample_points_poisson_disk(3000)
    mesh = get_armadillo_mesh()
    pcd = mesh.sample_points_poisson_disk(750)
    print(np.asarray(pcd.points))
    # print(np.max(np.asarray(pcd.points)), np.min(np.asarray(pcd.points)), np.average(np.asarray(pcd.points)))
    # o3d.visualization.draw_geometries_with_editing([pcd])
    alpha = 35
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    print('edge manifold', mesh.is_edge_manifold(allow_boundary_edges=True))
    print('edge manifold boundary', mesh.is_edge_manifold(allow_boundary_edges=False))
    print('vertex manifold', mesh.is_vertex_manifold())
    print('self intersection ', mesh.is_self_intersecting())
    print('watertight', mesh.is_watertight())
    print(f"alpha={alpha:.3f}")

    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # bbox = o3d.geometry.AxisAlignedBoundingBox()
    # bbox.min_bound = [-1, -1, -1]
    # bbox.max_bound = [0, 0, 0]
    # mesh = mesh.crop(bbox)
    # het_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    #
    vertex_colors = 0.75 * np.ones((len(mesh.vertices), 3))
    for boundary in mesh.get_non_manifold_edges():
        for vertex_id in boundary:
            vertex_colors[vertex_id] = [1, 0, 0]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True, mesh_show_wireframe=True)

    generate_alpha_shape(np.random.uniform(size=(10, 3)), displaying=True, alpha_value=1)
