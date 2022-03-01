r"""
Running this file will read the raw file with all WRF 500m datapoints, that can be found in:
O:\Utbygging\Fagress\BFA40 Konstruksjoner\10 Faggrupper\01 Metocean\Data\Vinddata\3d\Vind\KVT_Bjornafjorden_10m_ws10m_xyt.nc4

This file was also copied locally (could not read it directly in the O: directory). It can be found in my computer at:
C:\\Users\\bercos\\PycharmProjects\\Metocean\\masts_10min\\WRF_500m_all\\KVT_Bjornafjorden_10m_ws10m_xyt.nc4

Then, only the relevant datapoints, close to the bridge pints, are collected and saved in a 'mini-grid' dataset, saved to:
C:\\Users\\bercos\\PycharmProjects\\Metocean\\WRF_500_interpolated\\WRF_500m_minigrid.nc
"""

import os
import numpy as np
import netCDF4  # necessary to open the raw data. https://unidata.github.io/netcdf4-python/
import matplotlib.pyplot as plt

def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi

create_minigrid = False  # if True, when running this file, the miniggrid is created
earth_R = 6371*1000  # m. Globally averaged radius of the Earth. https://en.wikipedia.org/wiki/Earth_radius
lat_mid_Bj = np.deg2rad(60.1086)  # Latitude at the middle of the Bjørnafjord. Used to obtain the Earth circunference at that latitude
earth_circunf_R_at_lat = earth_R * np.cos(lat_mid_Bj) # IMPORTANT: In Norway (lat ~ 60deg), going 1 deg west is about half the distance (cos(60)) than going 1 deg North! earth_circunf_R_at_lat concerns the circunference when the Earth is cut at lat=60deg
lat_lon_aspect_ratio = 1/np.cos(lat_mid_Bj)  # correct aspect ratio for plots, since 1 deg lon corresponds to a different distance than 1 deg lat
n_bridge_WRF_nodes = 11

def bridge_WRF_nodes_coor_func(n_bridge_WRF_nodes = n_bridge_WRF_nodes, bridge_R = 5000, bridge_L = 5000, bridge_chord_yaw = rad(10), bridge_south_coor = np.array([rad(60.0855), rad(5.3705)]), unit='deg'):
    """
    :param n_bridge_WRF_nodes: Number of discrete points along the bridge axis to interpolate the WRF data.
    :param bridge_R: m. Bridge radius
    :param bridge_L: m. Bridge arc length
    :param bridge_chord_yaw: Bridge chord yaw with respect to the North-South alignment
    :param bridge_south_coor: latitude, longitude. https://www.google.pt/maps/@60.12861577,5.41361577,200m
    :param unit: return coordinates in 'deg' or 'rad'
    :return: The coordinate angles (latitudes and longitudes) of each bridge node where the WRF data will be interpolated
    """
    bridge_arc_delta_angle = bridge_L / bridge_R  # rad. "Aperture" angle of the whole bridge arc.
    bridge_chord = np.sin(bridge_arc_delta_angle / 2) * bridge_R * 2  # m.
    bridge_north_coor = np.array([bridge_south_coor[0] + np.cos(bridge_chord_yaw)*bridge_chord/earth_R,  # latitude, longitude. print(deg(bridge_north_coor)).
                                  bridge_south_coor[1] + np.sin(bridge_chord_yaw)*bridge_chord/earth_circunf_R_at_lat])  # e.g.(yaw=7deg): https://www.google.pt/maps/@60.12861577  5.41361577,200m
    bridge_WRF_nodes_angle = np.cumsum([0] + [bridge_arc_delta_angle / (n_bridge_WRF_nodes-1)]*(n_bridge_WRF_nodes-1))
    angle_horiz_to_R_centre = rad(90) - (rad(180) - rad(90) - bridge_arc_delta_angle/2 - bridge_chord_yaw)  # angle between a latidude line and the line crossing south point and bridge R center
    bridge_R_center_coor = np.array([bridge_south_coor[0] + np.sin(angle_horiz_to_R_centre)*bridge_R/earth_R,  # coordinates of the center of the imaginary bridge circle which encompasses the curved axis
                                     bridge_south_coor[1] - np.cos(angle_horiz_to_R_centre)*bridge_R/earth_circunf_R_at_lat])
    bridge_WRF_nodes_coor = np.array([[bridge_R_center_coor[0] + np.sin(-angle_horiz_to_R_centre+i)*bridge_R/earth_R,  # coordinates of the center of the imaginary bridge circle which encompasses the curved axis
                                       bridge_R_center_coor[1] + np.cos(-angle_horiz_to_R_centre+i)*bridge_R/earth_circunf_R_at_lat] for i in bridge_WRF_nodes_angle])
    assert np.allclose(bridge_north_coor, bridge_WRF_nodes_coor[-1]), "Last node does not have expected coordinates. There's an error somewhere. Check plot bellow"
    if unit == 'rad':
        return bridge_WRF_nodes_coor
    elif unit == 'deg':
        return deg(bridge_WRF_nodes_coor)

def create_minigrid_data_func():
    # Getting raw data. To see names of 'variables' do: print(dataset.variables)
    # folder_loc = os.path.join(os.getcwd(), r'WRF_500m_all/KVT_Bjornafjorden_10m_ws10m_xyt.nc4')  # original raw data file
    folder_loc = os.path.join(r'C:\Users\bercos\PycharmProjects\Metocean\WRF_500m_all\KVT_Bjornafjorden_10m_ws10m_xyt.nc4')
    # folder_loc = r'O:\Utbygging\Fagress\BFA40 Konstruksjoner\10 Faggrupper\01 Metocean\Data\Vinddata\3d\Vind\KVT_Bjornafjorden_10m_ws10m_xyt.nc4'  # this doesn't work?
    dataset = netCDF4.Dataset(folder_loc)

    # Large-grid: All available datapoint locations, in latitudes and longitudes
    lats = dataset.variables['latitude'][:].data  # latitudes
    lons = dataset.variables['longitude'][:].data  # longitudes
    lats_lons = {'lats': lats, 'lons': lons}  # both
    bridge_WRF_nodes_coor = bridge_WRF_nodes_coor_func(unit='rad')

    max_bridge_lat, max_bridge_lon = np.max(bridge_WRF_nodes_coor, axis=0)  # maximum
    min_bridge_lat, min_bridge_lon = np.min(bridge_WRF_nodes_coor, axis=0)

    # Getting the "mini-grid" data. The "mini-grid" is a subset of all the WRF 500m points, as a tight layout around the bridge.
    window_margin = 500  # m. The window where to find WRF data covers the max and min bridge lats and lons, plus this margin. Larger -> more points to interpolate (useful for non-linear interpolations)
    lat_lon_window = np.array([[min_bridge_lat - window_margin/earth_R, max_bridge_lat + window_margin/earth_R],
                               [min_bridge_lon - window_margin/earth_circunf_R_at_lat, max_bridge_lon + window_margin/earth_circunf_R_at_lat]])
    cond_1 = deg(lat_lon_window[0, 0]) < lats_lons['lats']
    cond_2 = deg(lat_lon_window[1, 0]) < lats_lons['lons']
    cond_3 = deg(lat_lon_window[0, 1]) > lats_lons['lats']
    cond_4 = deg(lat_lon_window[1, 1]) > lats_lons['lons']
    cond_all = np.logical_and(np.logical_and(np.logical_and(cond_1,cond_2), cond_3), cond_4)
    idx_cond_all = np.array(np.where(cond_all)).transpose()
    lat_lon_mini_grid = np.array([[lats_lons['lats'][i,j], lats_lons['lons'][i,j]] for i,j in idx_cond_all])

    # Getting wind directions and wind speeds for each mini-grid node
    hours_since_datetime_min = dataset.variables['time'][:].data
    ws = np.array([dataset.variables['ws'][i,j].data for i,j in idx_cond_all])  # wind speeds
    wd = np.array([dataset.variables['wd'][i,j].data for i,j in idx_cond_all])  # wind directions

    # Saving new .nc file of the mini-grid
    n_mini_grid_nodes = lat_lon_mini_grid.shape[0]
    n_time_points = dataset.variables['ws'].shape[-1]
    minidataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500m_minigrid.nc'), 'w', format='NETCDF4')
    minidataset.createDimension('n_nodes', n_mini_grid_nodes)
    minidataset.createDimension('n_time_points' , n_time_points)
    minidataset_lats = minidataset.createVariable('latitudes' , 'f4', ('n_nodes',))  # f4: 32-bit signed floating point
    minidataset_lons = minidataset.createVariable('longitudes', 'f4', ('n_nodes',))  # f4: 32-bit signed floating point
    minidataset_ws = minidataset.createVariable('ws', 'f4', ('n_nodes','n_time_points',))  # f4: 32-bit signed floating point
    minidataset_wd = minidataset.createVariable('wd', 'f4', ('n_nodes','n_time_points',))  # f4: 32-bit signed floating point
    minidataset_time = minidataset.createVariable('time', 'i4', ('n_time_points',))  # i4: 32-bit signed integer
    minidataset_lats[:] = lat_lon_mini_grid[:,0]
    minidataset_lons[:] = lat_lon_mini_grid[:,1]
    minidataset_ws[:] = ws
    minidataset_wd[:] = wd
    minidataset_time[:] = hours_since_datetime_min
    minidataset['time'].description = """Number of hours since 01/01/0001 00:00:00 (use datetime.datetime.min + datetime.timedelta(hours=minidataset['time'])"""
    minidataset.close()

    # Plotting all nodes of the bridge and the mini-grid
    plt.scatter(deg(bridge_WRF_nodes_coor[:,1]), deg(bridge_WRF_nodes_coor[:,0]), c='black', alpha=0.7, label='Interpolation nodes')
    plt.scatter(lat_lon_mini_grid[:,1], lat_lon_mini_grid[:,0], c='blue', s=10, alpha=0.7, label='Mini-grid WRF-500m')
    plt.axhline(deg(max_bridge_lat), c='orange', alpha=0.5, linestyle='-.')
    plt.axhline(deg(min_bridge_lat), c='orange', alpha=0.5, linestyle='-.')
    plt.axvline(deg(max_bridge_lon), c='orange', alpha=0.5, linestyle='-.')
    plt.axvline(deg(min_bridge_lon), c='orange', alpha=0.5, linestyle='-.')
    plt.gca().set_aspect(lat_lon_aspect_ratio)  # Very important: going 1 deg to west is not the same distance in meters as going 1 deg north. The proportion is np.cos(lat_mid_Bj)
    # plt.legend()
    plt.show()

if create_minigrid:
    create_minigrid_data_func()


