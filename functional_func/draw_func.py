from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import scipy.spatial as sci_spatial
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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


def draw_3D_curve(points_data, fig_name="DEFAULT", fig_size=(6, 6)):
    fig = plt.figure(figsize=fig_size)
    ax = Axes3D(fig)

    # x = points_data[:, 0]  # first column of the 2D matrix
    # y = points_data[:, 1]
    # z = points_data[:, 2]

    num_points_half = int(len(points_data) / 2)
    num_points_quarter = int(len(points_data) / 4)

    points_data = points_data[points_data[:, 0].argsort()]
    print(points_data.shape)
    print(points_data)
    print(points_data[0:num_points_half, :])

    points_data_xnegative = points_data[:num_points_half, :]
    points_data_xnegative = points_data_xnegative[points_data_xnegative[:, 1].argsort()]
    points_data_xnegative_ynegative = points_data_xnegative[:num_points_quarter]
    points_data_xnegative_ynegative = points_data_xnegative_ynegative[points_data_xnegative_ynegative[:, 2].argsort()]
    points_data_xnegative_ypositive = points_data_xnegative[num_points_quarter:]
    points_data_xnegative_ypositive = points_data_xnegative_ypositive[points_data_xnegative_ypositive[:, 2].argsort()]

    points_data_xpositive = points_data[num_points_half:]
    points_data_xpositive = points_data_xpositive[points_data_xpositive[:, 1].argsort()]
    points_data_xpositive_ynegative = points_data_xpositive[:num_points_quarter]
    points_data_xpositive_ynegative = points_data_xpositive_ynegative[points_data_xpositive_ynegative[:, 2].argsort()]
    points_data_xpositive_ypositive = points_data_xpositive[num_points_quarter:]
    points_data_xpositive_ypositive = points_data_xpositive_ypositive[points_data_xpositive_ypositive[:, 2].argsort()]

    print(points_data_xpositive_ypositive)

    x = points_data_xnegative_ynegative[:, 0]  # first column of the 2D matrix
    y = points_data_xnegative_ynegative[:, 1]
    z = points_data_xnegative_ynegative[:, 2]
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

    x = points_data_xnegative_ypositive[:, 0]  # first column of the 2D matrix
    y = points_data_xnegative_ypositive[:, 1]
    z = points_data_xnegative_ypositive[:, 2]
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

    x = points_data_xpositive_ynegative[:, 0]  # first column of the 2D matrix
    y = points_data_xpositive_ynegative[:, 1]
    z = points_data_xpositive_ynegative[:, 2]
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

    x = points_data_xpositive_ypositive[:, 0]  # first column of the 2D matrix
    y = points_data_xpositive_ypositive[:, 1]
    z = points_data_xpositive_ypositive[:, 2]

    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(x, y, z, xi, yi)

    surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.jet,
                           linewidth=1, antialiased=True)

    ax.set_zlim3d(np.min(Z), np.max(Z))
    fig.colorbar(surf)


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
