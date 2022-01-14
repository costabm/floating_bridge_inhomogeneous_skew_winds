import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata
import netCDF4  # necessary to open the raw data. https://unidata.github.io/netcdf4-python/
from create_minigrid_data_from_raw_WRF_500_data import bridge_WRF_nodes_coor_func, lat_mid_Bj


def from_cos_sin_to_0_2pi(cosines, sines, out_units='rad'):
    # # To test this angle transformations, do:
    # test = np.deg2rad([-170, 170, 30, -30, -90, 370, -1, -180, 180])
    # test = np.arctan2(np.sin(test), np.cos(test))
    # test[test < 0] = abs(test[test < 0]) + 2 * (np.pi - abs(test[test < 0]))
    # print(np.rad2deg(test))
    atan2 = np.arctan2(sines, cosines)  # angles in interval -pi to pi
    atan2[atan2 < 0] = abs(atan2[atan2 < 0]) + 2 * ( np.pi - abs(atan2[atan2 < 0]) )
    if out_units == 'deg':
        atan2 = np.rad2deg(atan2)
    return atan2

generate_new_WRF_at_bridge_nodes_file = True
z_str = '60m'  # 10m, 19m, 60m. Height at which the WRF has been calculated (different heights -> different files)

if generate_new_WRF_at_bridge_nodes_file:
    # Getting the already pre-processed data (mini-grid of the relevant WRF 500m datapoints that are near the bridge)
    dataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', f'WRF_{z_str}_500m_minigrid.nc'), 'r')
    lats_grid = dataset['latitudes'][:].data
    lons_grid = dataset['longitudes'][:].data
    ws_grid = dataset['ws'][:].data
    wd_grid = dataset['wd'][:].data
    time = dataset['time'][:].data
    n_time_points = dataset.variables['time'].shape[-1]
    dataset.close()

    # Getting bridge nodes
    lats_bridge, lons_bridge = bridge_WRF_nodes_coor_func().transpose()
    n_bridge_nodes = len(lats_bridge)

    # Interpolating wind speeds and directions onto the bridge nodes
    print('Interpolation might take 5-10 min to run...')
    ws_interp = np.array([griddata(points=(lats_grid,lons_grid), values=ws_grid[:,t], xi=(lats_bridge, lons_bridge), method='linear') for t in range(n_time_points)]).transpose()
    wd_cos_interp = np.array([griddata(points=(lats_grid,lons_grid), values=np.cos(np.deg2rad(wd_grid[:,t])), xi=(lats_bridge, lons_bridge), method='linear') for t in range(n_time_points)]).transpose()
    wd_sin_interp = np.array([griddata(points=(lats_grid,lons_grid), values=np.sin(np.deg2rad(wd_grid[:,t])), xi=(lats_bridge, lons_bridge), method='linear') for t in range(n_time_points)]).transpose()
    wd_interp = from_cos_sin_to_0_2pi(wd_cos_interp, wd_sin_interp, out_units='deg')

    # Saving the newly obtained WRF dataset at the bridge nodes
    bridgedataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', f'WRF_{z_str}_at_bridge_nodes.nc'), 'w', format='NETCDF4')
    bridgedataset.createDimension('n_nodes', n_bridge_nodes)
    bridgedataset.createDimension('n_time_points', n_time_points)
    bridgedataset_lats = bridgedataset.createVariable('latitudes', 'f4', ('n_nodes',))  # f4: 32-bit signed floating point
    bridgedataset_lons = bridgedataset.createVariable('longitudes', 'f4', ('n_nodes',))  # f4: 32-bit signed floating point
    bridgedataset_ws = bridgedataset.createVariable('ws', 'f4', ('n_nodes', 'n_time_points',))  # f4: 32-bit signed floating point
    bridgedataset_wd = bridgedataset.createVariable('wd', 'f4', ('n_nodes', 'n_time_points',))  # f4: 32-bit signed floating point
    bridgedataset_time = bridgedataset.createVariable('time', 'i4', ('n_time_points',))  # i4: 32-bit signed integer
    bridgedataset_lats[:] = lats_bridge
    bridgedataset_lons[:] = lons_bridge
    bridgedataset_ws[:] = ws_interp
    bridgedataset_wd[:] = wd_interp
    bridgedataset_time[:] = time
    bridgedataset['time'].description = """Number of hours since 01/01/0001 00:00:00 (use datetime.datetime.min + datetime.timedelta(hours=bridgedataset['time'])"""
    bridgedataset.close()


# Reading the WRF dataset at the bridge nodes:
bridgedataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', f'WRF_{z_str}_at_bridge_nodes.nc'), 'r', format='NETCDF4')
lats_bridge = bridgedataset['latitudes'][:].data
lons_bridge = bridgedataset['longitudes'][:].data
ws_orig = bridgedataset['ws'][:].data  # original data
wd_orig = bridgedataset['wd'][:].data
time_orig = bridgedataset['time'][:].data
n_bridge_nodes = np.shape(ws_orig)[0]
ws_cols = [f'ws_{n:02}' for n in range(n_bridge_nodes)]
wd_cols = [f'wd_{n:02}' for n in range(n_bridge_nodes)]
df_WRF = pd.DataFrame(ws_orig.T, columns=ws_cols)
df_WRF = df_WRF.join(pd.DataFrame(wd_orig.T, columns=wd_cols))
df_WRF['hour'] = time_orig
# Deleting data with max(U) < U_treshold m/s:
U_tresh = 12  # m/s. threshold. Data rows where all datapoints are below threshold, are removed
idx_to_keep = df_WRF[ws_cols] >= U_tresh
idx_to_keep = idx_to_keep.all(axis='columns')  # choose .any() to remove rows with any value below treshold, or .all() to remove rows where all values are below treshold
df_WRF = df_WRF.loc[idx_to_keep].reset_index(drop=True)
# Checking datasamples with high variance of wd
ws = df_WRF[ws_cols]
wd = np.deg2rad(df_WRF[wd_cols])
wd_cos = np.cos(wd)
wd_sin = np.sin(wd)
wd_cos_var = np.var(wd_cos, axis=1)
wd_sin_var = np.var(wd_sin, axis=1)
idxs_sorted_by = {'ws_var': np.array(np.argsort(np.var(ws, axis=1))),
                  'ws_max': np.array(np.argsort(np.max(ws, axis=1))),
                  'wd_var': np.array(np.argsort(pd.concat([wd_cos_var, wd_sin_var], axis=1).max(axis=1)))}

# Plotting datasamples with high variance of wd
sort_by = 'wd_var'  # 'wd_var' for variance of the wind direction, or 'ws_var' for variance of the wind speeds
rank = -1 # choose 0 for lowest, choose -1 for highest. Index of the sorted (by variance) list of indexes
idx_to_plot = idxs_sorted_by[sort_by][rank]
ws_to_plot = df_WRF[ws_cols].iloc[idx_to_plot].to_numpy()
cm = matplotlib.cm.cividis
norm = matplotlib.colors.Normalize()
sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
ws_colors = cm(norm(ws_to_plot))
wd_to_plot = np.deg2rad(df_WRF[wd_cols].iloc[idx_to_plot].to_numpy())
plt.figure(figsize=(4,6), dpi=300)
plt.quiver(*np.array([lons_bridge, lats_bridge]), ws_to_plot * np.cos(wd_to_plot), ws_to_plot * np.sin(wd_to_plot),
           color=ws_colors, angles='uv', scale=100, width= 0.015, headlength=3, headaxislength=3)
cbar = plt.colorbar(sm)
cbar.set_label('U [m/s]')
plt.xlim(5.36, 5.40)
plt.ylim(60.082, 60.133)
plt.xlabel('Longitude [$\degree$]')
plt.ylabel('Latitude [$\degree$]')
plt.gca().set_aspect(1/np.cos(lat_mid_Bj), adjustable='box')
plt.tight_layout()
plt.show()



print('Confirm in the WRF data that 0 deg is North and then angles rotate clockwise')
