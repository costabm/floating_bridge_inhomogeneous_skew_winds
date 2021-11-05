import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
from scipy.interpolate import griddata
from transformations import from_cos_sin_to_0_2pi
import netCDF4  # necessary to open the raw data. https://unidata.github.io/netcdf4-python/
from create_minigrid_data_from_raw_WRF_500_data import bridge_WRF_nodes_coor_func


generate_new_WRF_at_bridge_nodes_file = False

if generate_new_WRF_at_bridge_nodes_file:
    # Getting the already pre-processed data (mini-grid of the relevant WRF 500m datapoints that are near the bridge)
    dataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', 'WRF_500m_minigrid.nc'), 'r')
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
    cos_wd_interp = np.array([griddata(points=(lats_grid,lons_grid), values=np.cos(np.deg2rad(wd_grid[:,t])), xi=(lats_bridge, lons_bridge), method='linear') for t in range(n_time_points)]).transpose()
    sin_wd_interp = np.array([griddata(points=(lats_grid,lons_grid), values=np.sin(np.deg2rad(wd_grid[:,t])), xi=(lats_bridge, lons_bridge), method='linear') for t in range(n_time_points)]).transpose()
    wd_interp = from_cos_sin_to_0_2pi(cos_wd_interp, sin_wd_interp, out_units='deg')

    # Saving the newly obtained WRF dataset at the bridge nodes
    bridgedataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', r'WRF_at_bridge_nodes.nc'), 'w', format='NETCDF4')
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
    bridgedataset['ws'].description = "m/s"
    bridgedataset['wd'].description = "0 = from North; 90 = from East"
    bridgedataset['time'].description = "Num of hours since matlabs beginning of time, but there is a 2-day mismatch between python and matlab (1 due to date convention + 1 due to base index convention?). Hence, in Python run: datetime.datetime.min + datetime.timedelta(hours=17522904) - datetime.timedelta(days=2). The first timestamp of this database (17522904) should correspond to 1-Jan-2000"
    #  NOTE ABOUT TIME: In the original dataset, there is also a time variable 'jdate'. These are the number of days since the Matlab convention "0-Jan-0000 (proleptic ISO calendar)".
    #  The first 'jdate' value is 730486. (the first 'time' value is 17522904)
    #  To confirm go to https://octave-online.net/ and run: datetime(730486, 'ConvertFrom', 'datenum')
    #  The first timestamp of this database should be 1-Jan-2000
    bridgedataset.close()


# Reading the WRF dataset at the bridge nodes:
bridgedataset = netCDF4.Dataset(os.path.join(os.getcwd(), r'WRF_500_interpolated', r'WRF_at_bridge_nodes.nc'), 'r', format='NETCDF4')
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
df_WRF['datetime'] = [datetime.datetime.min + datetime.timedelta(hours=int(time_orig[i])) - datetime.timedelta(days=2) for i in range(len(time_orig))]
# Deleting data with max(U) < U_treshold m/s:
U_tresh = 12  # m/s. threshold. Data rows where datapoints below threshold, are removed
idx_to_keep = df_WRF[ws_cols] >= U_tresh
idx_to_keep = idx_to_keep.any(axis='columns')  # choose .any() to keep rows with at least one value above treshold, or .all() to keep only rows where all values are above treshold
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
sort_by = 'wd_var'  # 'wd_var' for variance of the wind direction, or 'ws_var' for variance of the wind speeds, or 'ws_max'
rank = -5 # choose 0 for lowest, choose -1 for highest. Index of the sorted (by variance) list of indexes
idx_to_plot = idxs_sorted_by[sort_by][rank]
ws_to_plot = df_WRF[ws_cols].iloc[idx_to_plot].to_numpy()
cm = matplotlib.cm.cividis
norm = matplotlib.colors.Normalize()
sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
ws_colors = cm(norm(ws_to_plot))
wd_to_plot = np.deg2rad(df_WRF[wd_cols].iloc[idx_to_plot].to_numpy())
plt.figure(figsize=(4,6), dpi=300)
plt.quiver(*np.array([lons_bridge, lats_bridge]), -ws_to_plot * np.sin(wd_to_plot), -ws_to_plot * np.cos(wd_to_plot),
           color=ws_colors, angles='uv', scale=100, width= 0.015, headlength=3, headaxislength=3)
cbar = plt.colorbar(sm)
cbar.set_label('U [m/s]')
plt.title(f'Sort by: {sort_by}. Rank: {rank}. U_tresh: {U_tresh}')
plt.xlim(5.366, 5.386)
plt.ylim(60.082, 60.133)
plt.xlabel('Longitude [$\degree$]')
plt.ylabel('Latitude [$\degree$]')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

