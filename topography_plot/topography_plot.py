import copy
import os
import math
import pandas as pd
import numpy as np
import netCDF4  # necessary to open the raw data. https://unidata.github.io/netcdf4-python/
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from create_minigrid_data_from_raw_WRF_500_data import bridge_WRF_nodes_coor_func, earth_R, lat_mid_Bj, earth_circunf_R_at_lat, lat_lon_aspect_ratio, n_bridge_WRF_nodes
from orography import get_all_geotiffs_merged, synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33
from nonhomogeneity import NwAllCases
from pyproj import Transformer, CRS
from windrose import WindroseAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi

def find_idx_of_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def utmToLatLng(zone, easting, northing, northernHemisphere=True):
    if not northernHemisphere:
        northing = 10000000 - northing
    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996
    arc = northing / k0
    mu = arc / (a * (1 - np.power(e, 2) / 4.0 - 3 * np.power(e, 4) / 64.0 - 5 * np.power(e, 6) / 256.0))
    ei = (1 - np.power((1 - e * e), (1 / 2.0))) / (1 + np.power((1 - e * e), (1 / 2.0)))
    ca = 3 * ei / 2 - 27 * np.power(ei, 3) / 32.0
    cb = 21 * np.power(ei, 2) / 16 - 55 * np.power(ei, 4) / 32
    cc = 151 * np.power(ei, 3) / 96
    cd = 1097 * np.power(ei, 4) / 512
    phi1 = mu + ca * np.sin(2 * mu) + cb * np.sin(4 * mu) + cc * np.sin(6 * mu) + cd * np.sin(8 * mu)
    n0 = a / np.power((1 - np.power((e * np.sin(phi1)), 2)), (1 / 2.0))
    r0 = a * (1 - e * e) / np.power((1 - np.power((e * np.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * np.tan(phi1) / r0
    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2
    t0 = np.power(np.tan(phi1), 2)
    Q0 = e1sq * np.power(np.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * np.power(dd0, 4) / 24
    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * np.power(dd0, 6) / 720
    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * np.power(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * np.power(Q0, 2) + 8 * e1sq + 24 * np.power(t0, 2)) * np.power(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / np.cos(phi1)
    _a3 = _a2 * 180 / np.pi
    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / np.pi
    if not northernHemisphere:
        latitude = -latitude
    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3
    return latitude, longitude

lon_mosaic, lat_mosaic, imgs_mosaic = get_all_geotiffs_merged()  # They are actually Eastings and Northings, but lon and lan are used for compactness.

# OLD: Using Longitudes and Latitudes. Then, it does not match the other plots that already have Northings and Eastings
# print('Converting from (easting, northing) to (lons, lats). Takes up to 2min...')
# lat_mosaic, lon_mosaic = utmToLatLng(33, easting_mosaic, northing_mosaic)  # these idx were visually decided from mosaic_of_selected_bjornafjord_maps.PNG
# imgs_mosaic = imgs_mosaic
# print('Done!')

def lighten_color(color, amount=0.5):
    """
    Amount: 0 to 1. Use 0 for maximum light (white)!
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]) + (1,)  # + alpha

def plot_WRF_grids_on_topography():
    matplotlib.rcParams.update({'font.size': 7})
    lon_lims = [-41000, -28000]
    lat_lims = [6.694E6, 6.710E6]
    lon_lim_idxs = [np.where(lon_mosaic[0,:]==lon_lims[0])[0][0], np.where(lon_mosaic[0,:]==lon_lims[1])[0][0]]
    lat_lim_idxs = [np.where(lat_mosaic[:,0]==lat_lims[0])[0][0], np.where(lat_mosaic[:,0]==lat_lims[1])[0][0]]
    lon_mosaic_crop = lon_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
    lat_mosaic_crop = lat_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
    imgs_mosaic_crop = imgs_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
    # cmap_colors = np.vstack((colors.to_rgba('skyblue'), plt.get_cmap('magma_r')(np.linspace(0, 0.8, 255))))  # choose the cmap colors here
    cmap_colors = np.vstack((lighten_color(colors.to_rgba('skyblue'), amount=0.3), plt.get_cmap('gist_earth')(np.linspace(0.3, 1.0, 255))))  # choose the cmap colors here
    cmap = colors.LinearSegmentedColormap.from_list('my_terrain_map', colors=cmap_colors)
    # cmap = copy.copy(plt.get_cmap('gray_r'))
    imgs_mosaic_crop = np.ma.masked_where(imgs_mosaic_crop == 0, imgs_mosaic_crop)  # set mask where height is 0, to be converted to another color
    # cmap.set_bad(color='skyblue')  # color where height == 0
    plt.figure(dpi=600)
    resolution_decrease_times = 1  # use 1. or more just to speed up the code while debugging
    imshow = plt.tripcolor(lon_mosaic_crop.flatten()[::resolution_decrease_times], lat_mosaic_crop.flatten()[::resolution_decrease_times], imgs_mosaic_crop.flatten()[::resolution_decrease_times], zorder=0, cmap=cmap)
    cb = plt.colorbar(imshow, pad=0.02)
    cb.set_label('Height [m]')
    # Getting raw data. To see names of 'variables' do: print(dataset.variables)
    # folder_loc = os.path.join(os.getcwd(), r'WRF_500m_all/KVT_Bjornafjorden_10m_ws10m_xyt.nc4')  # original raw data file
    folder_loc = os.path.join(r'C:\Users\bercos\PycharmProjects\Metocean\WRF_500m_all\KVT_Bjornafjorden_10m_ws10m_xyt.nc4')
    # folder_loc = r'O:\Utbygging\Fagress\BFA40 Konstruksjoner\10 Faggrupper\01 Metocean\Data\Vinddata\3d\Vind\KVT_Bjornafjorden_10m_ws10m_xyt.nc4'  # this doesn't work?
    dataset = netCDF4.Dataset(folder_loc)
    # Large-grid: All available datapoint locations, in latitudes and longitudes
    lats = dataset.variables['latitude'][:].data  # latitudes
    lons = dataset.variables['longitude'][:].data  # longitudes
    bridge_WRF_nodes_coor = bridge_WRF_nodes_coor_func(unit='deg')
    # Converting from actually Latitudes and Longitudes, to Northings and Eastings
    crs = CRS.from_epsg(25833)  # obtained in https://register.geonorge.no/epsg-koder/euref89-utm-sone-33-2d/5c13f8fd-ef2f-4a3c-b8b1-c53a2fa8c812
    latlon2utm = Transformer.from_crs(crs.geodetic_crs, crs)
    lons, lats = latlon2utm.transform(lats, lons)
    bridge_WRF_nodes_coor = np.array(latlon2utm.transform(bridge_WRF_nodes_coor[:,0], bridge_WRF_nodes_coor[:,1])).T[:,[1,0]]
    lats_lons = {'lats': lats, 'lons': lons}  # both
    max_bridge_lat, max_bridge_lon = np.max(bridge_WRF_nodes_coor, axis=0)  # maximum
    min_bridge_lat, min_bridge_lon = np.min(bridge_WRF_nodes_coor, axis=0)
    # Getting the "mini-grid" data. The "mini-grid" is a subset of all the WRF 500m points, as a tight layout around the bridge.
    window_margin = 1500  # m. The window where to find WRF data covers the max and min bridge lats and lons, plus this margin. Larger -> more points to interpolate (useful for non-linear interpolations)
    lat_lon_window = np.array([[min_bridge_lat - window_margin, max_bridge_lat + window_margin],
                               [min_bridge_lon - window_margin, max_bridge_lon + window_margin]])
    cond_1 = lat_lon_window[0, 0] < lats_lons['lats']
    cond_2 = lat_lon_window[1, 0] < lats_lons['lons']
    cond_3 = lat_lon_window[0, 1] > lats_lons['lats']
    cond_4 = lat_lon_window[1, 1] > lats_lons['lons']
    cond_all = np.logical_and(np.logical_and(np.logical_and(cond_1,cond_2), cond_3), cond_4)
    idx_cond_all = np.array(np.where(cond_all)).transpose()
    lat_lon_mini_grid = np.array([[lats_lons['lats'][i,j], lats_lons['lons'][i,j]] for i,j in idx_cond_all])

    # Getting the "major-grid" data.
    major_lat_lims = np.array([lat_lims[0], lat_lims[1]])
    major_lon_lims = np.array([lon_lims[0], lon_lims[1]])
    cond_1 = major_lat_lims[0] < lats_lons['lats']
    cond_2 = major_lon_lims[0] < lats_lons['lons']
    cond_3 = major_lat_lims[1] > lats_lons['lats']
    cond_4 = major_lon_lims[1] > lats_lons['lons']
    cond_all = np.logical_and(np.logical_and(np.logical_and(cond_1,cond_2), cond_3), cond_4)
    idx_cond_all = np.array(np.where(cond_all)).transpose()
    lat_lon_major_grid = np.array([[lats_lons['lats'][i,j], lats_lons['lons'][i,j]] for i,j in idx_cond_all])
    # Getting the mast positions
    mast_positions = np.array([synn_EN_33, svar_EN_33, osp1_EN_33, osp2_EN_33])
    # Guessing the WRF 4km grid positions
    wrf4km_p_ref = np.array([-38200,6.70257E6])
    wrf4km_p1 = np.array([-34325, 6.70162E6])  # guessed visually, by comparing with the WRF grid picture in reports from Kjeller Vindteknikk
    wrf4km_dx, wrf4km_dy = np.abs(wrf4km_p1 - wrf4km_p_ref)
    dpoints = [-3, -2, -1, 0, 1, 2, 3]
    wrf4km_grid = np.array([[wrf4km_p_ref + np.array([i*wrf4km_dx+j*wrf4km_dy, j*wrf4km_dx-i*wrf4km_dy]) for i in dpoints] for j in dpoints]).reshape(len(dpoints)**2,2)
    wrf4km_grid = np.delete(wrf4km_grid, np.where(np.all(wrf4km_grid - wrf4km_p_ref == [0,0], axis=1))[0][0], axis=0)  # removing the ref point, which is represented separately with another marker
    # Plotting all nodes of the bridge and the mini-grid
    plt.scatter(bridge_WRF_nodes_coor[:,1], bridge_WRF_nodes_coor[:,0], c='black', marker='s', s=10, edgecolors='none', alpha=0.8, label='Interp. nodes')
    plt.scatter(lat_lon_major_grid[:, 1], lat_lon_major_grid[:, 0], c='black', s=2, marker='o', edgecolors='none', alpha=0.6, label='WRF 500m grid', zorder=0.9)
    plt.plot(bridge_WRF_nodes_coor[:, 1], bridge_WRF_nodes_coor[:, 0], c='black', alpha=0.6, label='Bridge axis', lw=0.5)
    # plt.scatter(lat_lon_mini_grid[:,1], lat_lon_mini_grid[:,0], c='dodgerblue', s=6, edgecolors='none', alpha=0.8, label='Mini-grid WRF-500m')
    plt.scatter(wrf4km_grid[:,0], wrf4km_grid[:,1], c='black', s=25, edgecolors='none', marker='^', alpha=0.8, label='WRF 4km grid')
    plt.scatter(wrf4km_p_ref[0],wrf4km_p_ref[1], c='black', s=70, edgecolors='none', marker='*', alpha=0.8, label='WRF 4km ref.')
    plt.scatter(mast_positions[:, 0], mast_positions[:, 1], c='none', marker='o', s=60, edgecolors='black', alpha=0.8, label='Wind masts')
    plt.xlim(lon_lims)
    plt.ylim(lat_lims)
    # plt.gca().set_xticks([-30000, -35000, -40000])
    plt.gca().set_aspect(1)  # Very important: going 1 deg to west is not the same distance in meters as going 1 deg north. The proportion is np.cos(lat_mid_Bj)
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.legend(loc=9, ncol=3, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig('plots/topography_and_WRF_plot.png')
    plt.show()

def plot_U_roses_on_topography():

    n_WRF_nodes = n_bridge_WRF_nodes

    bj_coors = np.array([[-34449.260, 6699999.046],
                         [-34098.712, 6700359.394],
                         [-33786.051, 6700752.909],
                         [-33514.390, 6701175.648],
                         [-33286.431, 6701623.380],
                         [-33104.435, 6702091.622],
                         [-32970.204, 6702575.689],
                         [-32885.057, 6703070.741],
                         [-32849.826, 6703571.830],
                         [-32864.842, 6704073.945],
                         [-32929.936, 6704572.075]])
    bj_idx_strs = [f'{i:02}' for i in range(n_WRF_nodes)]

    Nw_all = NwAllCases()
    sort_by = 'ws_max'
    n_Nw_cases = 'all'
    Nw_all.set_df_WRF(sort_by=sort_by, U_tresh=17.2)
    df_WRF = Nw_all.df_WRF


    # WINDROSES
    wd = df_WRF['wd_03']
    ws = df_WRF['ws_03']
    plt.figure(dpi=1000)
    ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=0.0, edgecolor='white', cmap=plt.get_cmap('Reds'), bins=[0,10,15,20], nsector=1)
    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
    plt.legend(prop={'size': 40})
    ax.get_legend().set_title("Wind speeds [m/s]:")
    ax.set_facecolor('red')
    plt.axis('off')
    plt.rcParams['legend.title_fontsize'] = 40
    plt.savefig('plots/wind_roses_on_topography_legend.png')
    plt.show()
    plt.close()




    # WINDROSES ON TOPOGRAPHY PLOTS
    cmap_colors = np.vstack((lighten_color(colors.to_rgba('skyblue'), amount=0.3), plt.get_cmap('gist_earth')(np.linspace(0.3, 1.0, 255))))  # choose the cmap colors here
    cmap = colors.LinearSegmentedColormap.from_list('my_terrain_map', colors=cmap_colors)
    # lon_lims = [-35000, -32000]
    # lat_lims = [6.6996E6, 6.705E6]
    lon_lims = [-35200, -32200]
    lat_lims = [6.6994E6, 6.7052E6]
    lon_lim_idxs = [np.where(lon_mosaic[0, :] == lon_lims[0])[0][0], np.where(lon_mosaic[0, :] == lon_lims[1])[0][0]]
    lat_lim_idxs = [np.where(lat_mosaic[:, 0] == lat_lims[0])[0][0], np.where(lat_mosaic[:, 0] == lat_lims[1])[0][0]]
    lon_mosaic_crop = lon_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
    lat_mosaic_crop = lat_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
    imgs_mosaic_crop = imgs_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]

    plt.figure(dpi=400)
    plt.title('Strong-wind roses')
    bbox = ((lon_mosaic_crop.min(), lon_mosaic_crop.max(),
             lat_mosaic_crop.min(), lat_mosaic_crop.max()))
    plt.imshow(imgs_mosaic_crop, extent=bbox, zorder=0, cmap=cmap, vmin=0, vmax=750)
    main_ax = plt.gca()
    wrax = {}
    # for pt_idx, pt in enumerate(bj_pt_strs):
    #     plt.scatter(bj_coors[pt_idx][0], bj_coors[pt_idx][1], marker='o', facecolors='black', edgecolors='black', s=10, label='Measurement location' if pt==0 else None)

    # plt.legend() #[handles[idx] for idx in order], [labels[idx] for idx in order], handletextpad=0.1)
    # plt.tight_layout(pad=0.05)


    for pt_idx, pt in enumerate(bj_idx_strs):
        if True:  # pt_idx%10==0: # if point index is even:
            ########### WIND ROSES
            wd = df_WRF['wd_'+pt]
            ws = df_WRF['ws_'+pt]

            wrax[pt] = inset_axes(main_ax,
                                  width=0.65,  # size in inches
                                  height=0.65,  # size in inches
                                  loc='center',  # center bbox at given position
                                  bbox_to_anchor=(bj_coors[pt_idx][0], bj_coors[pt_idx][1]),  # position of the axe
                                  bbox_transform=main_ax.transData,  # use data coordinate (not axe coordinate)
                                  axes_class=WindroseAxes)  # specify the class of the axe
            # # print(f'Min: {(Iu_min-Iu_min_all_pts)/(Iu_max_all_pts-Iu_min_all_pts)}')
            # # print(f'Max; {(Iu_max-Iu_min_all_pts)/(Iu_max_all_pts-Iu_min_all_pts)}')
            # wrax[pt].bar(wd, Iu, opening=1.0, nsector=360, cmap=truncate_colormap(matplotlib.pyplot.cm.Reds, (Iu_min - Iu_min_all_pts) / (Iu_max_all_pts - Iu_min_all_pts),
            #                                                                       (Iu_max - Iu_min_all_pts) / (Iu_max_all_pts - Iu_min_all_pts)))
            wrax[pt].bar(wd, ws, normed=True, opening=0.8, edgecolor='none', cmap=plt.get_cmap('Reds'), bins=[0, 10, 15, 20], nsector=12, alpha=0.8)
            wrax[pt].tick_params(labelleft=False, labelbottom=False)
            wrax[pt].patch.set_alpha(0)
            wrax[pt].axis('off')

    # cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=Iu_min_all_pts, vmax=Iu_max_all_pts), cmap=matplotlib.pyplot.cm.Reds), ax=main_ax, pad=0.02)
    # cb.set_label('$I_u$')
    main_ax.axis('off')
    plt.tight_layout()
    plt.savefig('plots/wind_roses_on_topography.png')
    plt.show()
    plt.close()

    pl
    plt.legend()
    plt.show()



