import math
import numpy as np


# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
def fibonacci_sphere(num_points=162,radius=1):
    points = []
    # https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html#:~:text=Summary,three%20points%20to%20the%20sphere.
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians - 2.39999 radian (compared to degrees 137)

    # latitude direction  pi ---?  polar angle
    # longitude direction 2pi  ---?azimuth angle
    for i in range(num_points):
        # y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        # radius = math.sqrt(1 - y * y)  # radius at y
        #
        # theta = phi * i  # golden angle increment
        #
        # x = math.cos(theta) * radius
        # z = math.sin(theta) * radius

        lat = math.asin(-1.0 + 2.0 * float(i / (num_points + 1)))
        lon = phi * i

        x = math.cos(lon) * math.cos(lat)*radius
        y = math.sin(lon) * math.cos(lat)*radius
        z = math.sin(lat)*radius

        points.append([x, y, z])

    points_np = np.array(points)

    return points_np


def fibonacci_spiral_disc(num_points, density_inverse=10):
    points = []

    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians - 2.39999 radian (compared to degrees 137)

    for i in range(num_points):
        radius = math.sqrt(i) * density_inverse
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        y = math.sin(theta) * radius

        points.append([x, y])

    return points
