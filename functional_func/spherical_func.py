import math
import numpy as np


# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
def fibonacci_sphere(num_points=162, radius=1):
    points_cartesian = []
    points_spherical = []
    # https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html#:~:text=Summary,three%20points%20to%20the%20sphere.
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians - 2.39999 radian (compared to degrees 137)

    # latitude direction  pi ---?  polar angle ------from x, not from z
    # longitude direction 2pi  ---?azimuth angle

    for i in range(num_points):
        lat = math.asin(-1.0 + 2.0 * float(i / (num_points + 1)))
        # lat = lat % (-math.pi) if lat < 0 else lat % math.pi
        # lat = lat % (2*math.pi)
        lon = phi * i

        x = math.cos(lon) * math.cos(lat) * radius
        y = math.sin(lon) * math.cos(lat) * radius
        z = math.sin(lat) * radius

        points_cartesian.append([x, y, z])
        # points_spherical.append([radius, lat, lon])

    points_np_cartesian = np.array(points_cartesian)
    # points_np_spherical = np.array(points_spherical)
    return points_np_cartesian

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


def average_lat_lon_sphere(radian_distance=0.05, radius=1):
    points = []
    num_points_lon = int(2 / radian_distance)
    num_points_lat = int(2 / radian_distance)
    for i in range(num_points_lon):
        for j in range(num_points_lat):
            x = math.cos(radian_distance * i * math.pi) * math.cos(radian_distance * j * math.pi) * radius
            y = math.sin(radian_distance * i * math.pi) * math.cos(radian_distance * j * math.pi) * radius
            z = math.sin(radian_distance * j * math.pi) * radius
            points.append([x, y, z])
    points = np.array(points)
    return points
