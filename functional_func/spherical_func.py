import math
import numpy as np
from functional_func.general_func import read_csv_to_df


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


def calculate_R_with_average_with_my_locate_method(points_sorted_by_lat, points_sorted_by_lon, lat_phi, lon_theta,
                                                   probable_interval_num,
                                                   average_points=5):
    """
    not work so well. a lot of problem.
    :param points_sorted_by_lat: the surface points sorted by lat direction -- pass by reference, don't worry about the redundancy.
    :param points_sorted_by_lon: the surface points sorted by lon direction
    :param lat_phi: the ray (the sample) latitude value
    :param lon_theta: the ray (the sample) longitude value
    :param probable_interval_num: the proportion of the samples number and all surface points
    :param average_points: how many neighboring points to calculate to represent this sample(with phi and theta)
    :return: the sampling result R=f(phi,theta)
    """
    FLAG_AS_GOTO = False

    # flag to find the ray lat in points
    points_num_this_cell = len(points_sorted_by_lat)
    prob_ray_lat_index_in_points = int(lat_phi / math.pi * points_num_this_cell)
    # print(prob_ray_lat_index_in_points)
    # -----------------deal with the border issues---------------------------- #
    if prob_ray_lat_index_in_points + probable_interval_num >= points_num_this_cell:
        prob_ray_lat_index_in_points -= (probable_interval_num + 1)
        FLAG_AS_GOTO = True
    elif prob_ray_lat_index_in_points - probable_interval_num <= 0:
        prob_ray_lat_index_in_points += (probable_interval_num + 1)
        FLAG_AS_GOTO = True
    elif lat_phi <= points_sorted_by_lat[0][1]:
        prob_ray_lat_index_in_points = probable_interval_num
        FLAG_AS_GOTO = True
    elif lat_phi >= points_sorted_by_lat[points_num_this_cell - 1][1]:
        prob_ray_lat_index_in_points = points_num_this_cell - (probable_interval_num + 1)
        FLAG_AS_GOTO = True
    # ---------------------------------------------------------------------- #

    if FLAG_AS_GOTO is False:
        # print(prob_ray_lat_index_in_points)
        pro_ray_index_pos = (0, 0)
        if points_sorted_by_lat[prob_ray_lat_index_in_points - probable_interval_num][1] < lat_phi < \
                points_sorted_by_lat[prob_ray_lat_index_in_points + probable_interval_num][1]:
            pro_ray_index_pos = (0, 0)
            prob_ray_lat_index_in_points = prob_ray_lat_index_in_points - probable_interval_num
        elif points_sorted_by_lat[prob_ray_lat_index_in_points + probable_interval_num][1] <= lat_phi:
            pro_ray_index_pos = (1, 0)
            prob_ray_lat_index_in_points = prob_ray_lat_index_in_points + probable_interval_num
        elif points_sorted_by_lat[prob_ray_lat_index_in_points - probable_interval_num][1] >= lat_phi:
            pro_ray_index_pos = (0, 1)
            prob_ray_lat_index_in_points = prob_ray_lat_index_in_points - probable_interval_num

        while 1:
            print(prob_ray_lat_index_in_points)
            print("finding", lat_phi)
            if pro_ray_index_pos == (0, 0):
                if points_sorted_by_lat[prob_ray_lat_index_in_points][1] > lat_phi:
                    break
                else:
                    prob_ray_lat_index_in_points += 1
            elif pro_ray_index_pos == (1, 0):
                if points_sorted_by_lat[prob_ray_lat_index_in_points][1] > lat_phi:
                    break
                else:
                    prob_ray_lat_index_in_points += 1
            elif pro_ray_index_pos == (0, 1):
                if points_sorted_by_lat[prob_ray_lat_index_in_points][1] < lat_phi:
                    break
                else:
                    prob_ray_lat_index_in_points -= 1
            else:
                print('======lat====finding---ray----====error====================================')
    # print(ray_lat)
    # print(points_at_spherical_lat_phi[prob_ray_lat_index_in_points])
    # flag to find the ray lon in points
    FLAG_AS_GOTO = False
    prob_ray_lon_index_in_points = int(lon_theta / (math.pi * 2) * points_num_this_cell)
    # -----------------deal with the border issues---------------------------- #
    if prob_ray_lon_index_in_points + probable_interval_num >= points_num_this_cell:
        prob_ray_lon_index_in_points -= (probable_interval_num + 1)
        FLAG_AS_GOTO = True
    elif prob_ray_lon_index_in_points - probable_interval_num <= 0:
        prob_ray_lon_index_in_points += (probable_interval_num + 1)
        FLAG_AS_GOTO = True
    elif lon_theta <= points_sorted_by_lon[0][2]:
        prob_ray_lon_index_in_points = probable_interval_num
        FLAG_AS_GOTO = True
    elif lon_theta >= points_sorted_by_lon[points_num_this_cell - 1][2]:
        prob_ray_lon_index_in_points = points_num_this_cell - (probable_interval_num + 1)
        FLAG_AS_GOTO = True
    # ---------------------------------------------------------------------- #

    if FLAG_AS_GOTO is False:
        pro_ray_index_pos = (0, 0)
        if points_sorted_by_lon[prob_ray_lon_index_in_points - probable_interval_num][2] < lon_theta < \
                points_sorted_by_lon[prob_ray_lon_index_in_points + probable_interval_num][2]:
            pro_ray_index_pos = (0, 0)
            prob_ray_lon_index_in_points = prob_ray_lon_index_in_points - probable_interval_num
        elif points_sorted_by_lon[prob_ray_lon_index_in_points + probable_interval_num][2] <= lon_theta:
            pro_ray_index_pos = (1, 0)
            prob_ray_lon_index_in_points = prob_ray_lon_index_in_points + probable_interval_num
        elif points_sorted_by_lon[prob_ray_lon_index_in_points - probable_interval_num][2] >= lon_theta:
            pro_ray_index_pos = (0, 1)
            prob_ray_lon_index_in_points = prob_ray_lon_index_in_points - probable_interval_num

        while 1:
            if pro_ray_index_pos == (0, 0):
                if points_sorted_by_lon[prob_ray_lon_index_in_points][2] > lon_theta:
                    break
                else:
                    prob_ray_lon_index_in_points += 1
            elif pro_ray_index_pos == (1, 0):
                if points_sorted_by_lon[prob_ray_lon_index_in_points][2] > lon_theta:
                    break
                else:
                    prob_ray_lon_index_in_points += 1
            elif pro_ray_index_pos == (0, 1):
                if points_sorted_by_lon[prob_ray_lon_index_in_points][2] < lon_theta:
                    break
                else:
                    # if prob_ray_lon_index_in_points==0:
                    #     print('----------------------')
                    #     print(points_at_spherical_lon_theta[prob_ray_lon_index_in_points][2])
                    #     print(ray_lon)
                    #     print('----------------------')

                    prob_ray_lon_index_in_points -= 1
            else:
                print('======lon====finding---ray----====error====================================')
    # print(lon_theta)
    # print(points_sorted_by_lon[prob_ray_lon_index_in_points])

    prob_points_set = np.vstack((points_sorted_by_lon[
                                 prob_ray_lon_index_in_points - probable_interval_num:prob_ray_lon_index_in_points + probable_interval_num,
                                 :], points_sorted_by_lat[
                                     prob_ray_lat_index_in_points - probable_interval_num:prob_ray_lat_index_in_points + probable_interval_num,
                                     :]))
    # print(prob_points_set.shape[0])
    prob_set_distance = (prob_points_set[:, 1] - lat_phi) ** 2 + (prob_points_set[:, 2] - lon_theta) ** 2
    prob_points_set = np.hstack((prob_points_set, prob_set_distance.reshape((prob_set_distance.shape[0], 1))))
    # print(prob_points_set[:, 3].argsort())
    prob_points_set = prob_points_set[prob_points_set[:, 3].argsort()]
    # print(prob_points_set)
    # print(prob_points_set[:surface_average_num, 0])
    return np.average(prob_points_set[:average_points, 0])


def calculate_R_with_lat_lon(spherical_points, lat_phi, lon_theta, average_points=5):
    '''

    :param spherical_points: all surface points
    :param lat_phi: the co-latitude radius
    :param lon_theta: the longitude radius
    :param average_points: how many closest point to calculate
    :return:
    '''
    distance_vector = (spherical_points[:, 1] - lat_phi) ** 2 + (spherical_points[:, 2] - lon_theta) ** 2
    # print(distance_vector.shape)
    # print(spherical_points[:,0].shape)
    distance_matrix = np.vstack((spherical_points[:, 0], distance_vector)).T
    # print(distance_matrix.shape)
    distance_matrix = distance_matrix[distance_matrix[:, 1].argsort()]
    # print(distance_matrix.shape)
    # print(distance_matrix[0:average_points,0])
    return np.mean(distance_matrix[0:average_points, 0])


def sort_by_phi_theta(points_at_spherical):
    # from small-> large
    points_at_spherical_phi = points_at_spherical[points_at_spherical[:, 1].argsort()]
    points_at_spherical_theta = points_at_spherical[points_at_spherical[:, 2].argsort()]
    return points_at_spherical_phi, points_at_spherical_theta


def normalize_SHc(path_original_SHc_saving_csv, df_embryo_volume_surface_slices, path_normalized_SHc_saving_csv):
    df_SHc = read_csv_to_df(path_original_SHc_saving_csv)

    for index_tmp in df_embryo_volume_surface_slices.index:
        this_normalized_coefficient = df_embryo_volume_surface_slices.loc[index_tmp][2]
        normalization_tmp = df_SHc.loc[index_tmp] / this_normalized_coefficient
        df_SHc.loc[index_tmp] = normalization_tmp
    df_SHc.to_csv(path_normalized_SHc_saving_csv)
    return df_SHc
