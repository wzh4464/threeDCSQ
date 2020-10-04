from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def draw_3D_curve(points_data, fig_name="DEFAULT", fig_size=(6, 6)):
    fig = plt.figure(figsize=fig_size)
    ax = Axes3D(fig)

    x = points_data[:, 0]  # first column of the 2D matrix
    y = points_data[:, 1]
    z = points_data[:, 2]

    # ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_trisurf(x, y, z, cmap=plt.cm.Spectral,antialiased=True)

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title(fig_name)
    plt.show()


def draw_3D_points(points_data, fig_name="DEFAULT", fig_size=(6, 6)):
    x = points_data[:, 0]  # first column of the 2D matrix
    y = points_data[:, 1]
    z = points_data[:, 2]

    fig = plt.figure(figsize=fig_size)
    ax = Axes3D(fig)

    # divide as three different color so that we can see them more easily
    third_of_points_num = int(len(x) / 3)
    ax.scatter(x[:third_of_points_num], y[:third_of_points_num], z[:third_of_points_num], c='y')  # 绘制数据点
    ax.scatter(x[third_of_points_num:2 * third_of_points_num], y[third_of_points_num:2 * third_of_points_num],
               z[third_of_points_num:2 * third_of_points_num], c='r')
    ax.scatter(x[2 * third_of_points_num:len(x)], y[2 * third_of_points_num:len(x)], z[2 * third_of_points_num:len(x)],
               c='g')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.title(fig_name)
    plt.show()
