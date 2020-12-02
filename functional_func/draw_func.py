from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import scipy.spatial as sci_spatial
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.tri import Triangulation


def generate_2D_Z_ARRAY(x, y, z):
    x_num = len(x)
    y_num = len(y)
    Z = np.zeros((x_num, y_num), dtype=np.float64)
    for i in range(x_num):
        for j in range(y_num):
            if i == j:
                Z[i][j] = z[i]
            else:
                Z[i][j] = np.float64(0)
    return Z


def draw_3D_curve_with_lines(points_data, fig_name="DEFAULT", fig_size=(6, 6)):
    fig = plt.figure(figsize=fig_size)
    ax = Axes3D(fig)

    # x = points_data[:, 0]  # first column of the 2D matrix
    # y = points_data[:, 1]
    # z = points_data[:, 2]

    num_points_half = int(len(points_data) / 2)
    num_points_quarter = int(len(points_data) / 4)

    points_data = points_data[points_data[:, 0].argsort()]
    # print(points_data.shape)
    # print(points_data)
    # print(points_data[0:num_points_half, :])

    points_data_xnegative = points_data[:num_points_half, :]
    points_data_xnegative = points_data_xnegative[points_data_xnegative[:, 1].argsort()]
    points_data_xnegative_ynegative = points_data_xnegative[:num_points_quarter]
    points_data_xnegative_ynegative = points_data_xnegative_ynegative[points_data_xnegative_ynegative[:, 2].argsort()]
    points_data_xnegative_ypositive = points_data_xnegative[num_points_quarter:]
    points_data_xnegative_ypositive = points_data_xnegative_ypositive[points_data_xnegative_ypositive[:, 2].argsort()]

    points_data_xpositive = points_data[num_points_half:]
    points_data_xpositive = points_data_xpositive[points_data_xpositive[:, 1].argsort()]
    points_data_xpositive_ynegative = points_data_xpositive[:num_points_quarter]
    # points_data_xpositive_ynegative = points_data_xpositive_ynegative[points_data_xpositive_ynegative[:, 2].argsort()]
    points_data_xpositive_ypositive = points_data_xpositive[num_points_quarter:]
    points_data_xpositive_ypositive = points_data_xpositive_ypositive[points_data_xpositive_ypositive[:, 2].argsort()]

    x = points_data_xnegative_ynegative[:, 0]  # first column of the 2D matrix
    y = points_data_xnegative_ynegative[:, 1]
    z = points_data_xnegative_ynegative[:, 2]
    ax.plot3D(x, y, z, 'blue')

    x = points_data_xnegative_ypositive[:, 0]  # first column of the 2D matrix
    y = points_data_xnegative_ypositive[:, 1]
    z = points_data_xnegative_ypositive[:, 2]
    ax.plot3D(x, y, z, 'grey')

    x = points_data_xpositive_ynegative[:, 0]  # first column of the 2D matrix
    y = points_data_xpositive_ynegative[:, 1]
    z = points_data_xpositive_ynegative[:, 2]
    ax.scatter3D(x, y, z, c=z, cmap="Greens")

    x = points_data_xpositive_ypositive[:, 0]  # first column of the 2D matrix
    y = points_data_xpositive_ypositive[:, 1]
    z = points_data_xpositive_ypositive[:, 2]
    ax.plot3D(x, y, z, 'red')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title(fig_name)
    plt.show()


def draw_3D_curve_with_triangle(points_data, fig_name="DEFAULT", fig_size=(10, 10)):
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(fig_name)

    ax = Axes3D(fig)

    x = points_data[:, 0]  # first column of the 2D matrix
    y = points_data[:, 1]
    z = points_data[:, 2]
    ax.plot_trisurf(x, y, z)

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_title(fig_name)
    # plt.title(fig_name)
    plt.show()


def draw_3D_points_in_new_coordinate(points, center=None):
    if center is None:
        center = [0, 0, 0]
    points_new = points - center
    draw_3D_points(points_new)
    return points_new


def draw_3D_points(points_data, fig_name="DEFAULT", fig_size=(10, 10)):
    x = points_data[:, 0]  # first column of the 2D matrix
    y = points_data[:, 1]
    z = points_data[:, 2]

    fig = plt.figure(figsize=fig_size)
    ax = Axes3D(fig)

    ax.scatter3D(x, y, z, cmap='BuRd',marker='o')

    # # Add x, y gridlines
    # ax.grid(b=True, color='grey',
    #         linestyle='-.', linewidth=0.3,
    #         alpha=0.2)
    #
    # # Creating color map
    # my_cmap = plt.get_cmap('hsv')
    #
    # # Creating plot
    # sctt = ax.scatter3D(x, y, z,
    #                     alpha=0.8,
    #                     c=(x + y + z),
    #                     cmap=my_cmap,
    #                     marker='^')
    # fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.title(fig_name)
    plt.show()
