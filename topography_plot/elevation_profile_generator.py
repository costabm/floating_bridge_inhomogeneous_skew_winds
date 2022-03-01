import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator, RectBivariateSpline, griddata
from orography import get_all_geotiffs_merged
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import copy
import pandas as pd

lon_mosaic, lat_mosaic, imgs_mosaic = get_all_geotiffs_merged()

print('Building a K-D tree (takes approx. 1 minute)...')
my_tree = cKDTree(np.array([lon_mosaic.ravel(), lat_mosaic.ravel()]).T)


def get_point2_from_point1_dir_and_dist(point_1=[-34625., 6700051.], direction_deg=180, distance=5000):
    lon_1, lat_1 = point_1
    lon_2 = lon_1 + np.sin(np.deg2rad(direction_deg)) * distance
    lat_2 = lat_1 + np.cos(np.deg2rad(direction_deg)) * distance
    return np.array([lon_2, lat_2])


def elevation_profile_generator(point_1, point_2, step_distance=10, list_of_distances=False):
    """
    Args:
        point_1: e.g. [-35260., 6700201.]
        point_2: e.g. [-34501., 6690211.]
        step_distance: Should be larger or equal to the database grid resolution (in this case dtm10 -> 10meters)
        list_of_distances: e.g. False -> uses linear step distance. e.g. [i*(5+5*i) for i in range(45)] -> has non-even steps
    Returns: horizontal distances to point 1 and heights of each interpolated point between point 1 and 2
    """
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    tolerance_trick = 1  # meter. Force lat or lon to always change at least 1 meter to avoid error in our approach with np.arange
    if point_2[0] - point_1[0] == 0:
        point_2 = [point_2[0] + tolerance_trick, point_2[1]]
    if point_2[1] - point_1[1] == 0:
        point_2 = [point_2[0], point_2[1] + tolerance_trick]
    delta_lon = point_2[0] - point_1[0]
    delta_lat = point_2[1] - point_1[1]
    total_distance = np.sqrt(delta_lon**2 + delta_lat**2)  # SRSS
    if list_of_distances:
        new_dists = np.array(list_of_distances)
    else:
        n_steps = int(np.round(total_distance / step_distance)) + 1
        new_dists = np.linspace(0, total_distance, n_steps)
    new_lons = point_1[0] + new_dists / total_distance * delta_lon
    new_lats = point_1[1] + new_dists / total_distance * delta_lat
    # OLD VERSION DOWN
    # new_lons = np.linspace(point_1[0], point_2[0], n_steps)
    # new_lats = np.linspace(point_1[1], point_2[1], n_steps)
    new_lon_lat_idxs = my_tree.query(np.array([new_lons,new_lats]).T)[1]
    heights = imgs_mosaic.ravel()[new_lon_lat_idxs]
    return new_dists, heights


def plot_elevation_profile(point_1, point_2, step_distance, list_of_distances):
    dists, heights = elevation_profile_generator(point_1, point_2, step_distance=10, list_of_distances=False)
    sea_idxs = np.where(heights==0)[0]
    land_idxs = np.where(heights!=0)[0]

    plt.figure(dpi=400/0.85, figsize=(5*0.85,1.8*0.85))
    plt.title('Upstream terrain profile')
    plt.plot(dists[land_idxs], heights[land_idxs], c='peru' , linestyle='-', linewidth=1, zorder=1.9) #c='peru')
    plt.scatter(dists[land_idxs], heights[land_idxs], c='peru' , s=1, label='Ground') #c='peru')
    plt.scatter(dists[sea_idxs], heights[sea_idxs], c='skyblue', s=2, label='Sea', zorder=2)  # c='peru')
    plt.xlabel('Upstream distance [m]')
    plt.ylabel('Height [m]')
    plt.ylim([-10, 450])
    plt.yticks([0,200,400])
    plt.legend(markerscale=3, handletextpad=0.1)
    plt.tight_layout(pad=0.05)
    plt.savefig('plots/TerrainProfile_example.png')
    plt.show()

    new_dists, new_heights = elevation_profile_generator(point_1, point_2, step_distance=step_distance, list_of_distances=list_of_distances)
    new_sea_idxs = np.where(new_heights==0)[0]
    new_land_idxs = np.where(new_heights!=0)[0]

    df_mins_maxs = pd.read_csv('df_mins_maxs.csv')
    df_mins_maxs_Z_columns = [c for c in df_mins_maxs.columns if c[0]=='Z']
    Z_mins = np.array(df_mins_maxs[df_mins_maxs_Z_columns].loc[0])
    Z_maxs = np.array(df_mins_maxs[df_mins_maxs_Z_columns].loc[1])

    new_normalized_heights = (new_heights - Z_mins) / (Z_maxs - Z_mins)

    plt.figure(dpi=400/0.85, figsize=(5*0.85,1.8*0.85))
    plt.title('Z vector')
    # NEW:
    plt.plot(new_dists, new_normalized_heights, c='black', linestyle='--', alpha=0.6, linewidth=1) #c='peru')
    plt.scatter(new_dists, new_normalized_heights, c='black', s=3, label='Ground')  # c='peru')
    # OLD:
    # plt.plot(new_dists, new_heights, c='black', linestyle='--', alpha=0.6, linewidth=1) #c='peru')
    # plt.scatter(new_dists, new_heights, c='black', s=3, label='Ground')  # c='peru')
    plt.xlabel('Upstream distance [m]')
    plt.ylabel('Norm. height')
    plt.ylim([-0.05, 1.05])
    plt.yticks([0,1])
    ax = plt.gca()
    ax.set_yticklabels(['    0', '    1'])
    plt.tight_layout(pad=0.05)
    plt.savefig('plots/TerrainProfile_2_example.png')
    plt.show()

    plt.figure(dpi=400/0.85, figsize=(5*0.85,1.8*0.85))
    plt.title('R vector')
    plt.scatter(new_dists[new_sea_idxs], np.zeros(len(new_sea_idxs)), c='black', s=2, label='Ground')  # c='peru')
    plt.scatter(new_dists[new_land_idxs], np.ones(len(new_land_idxs)), c='black', s=2, label='Ground')  # c='peru')
    plt.xlabel('Upstream distance [m]')
    plt.ylabel('Norm. rough.')
    plt.yticks([0,1])
    ax = plt.gca()
    ax.set_yticklabels(['    0', '    1'])
    plt.tight_layout(pad=0.05)
    plt.savefig('plots/TerrainProfile_3_example.png')
    plt.show()


    lon_lims = [-50000, -20000]
    lat_lims = [6.685E6, 6.715E6]
    lon_lim_idxs = [np.where(lon_mosaic[0,:]==lon_lims[0])[0][0], np.where(lon_mosaic[0,:]==lon_lims[1])[0][0]]
    lat_lim_idxs = [np.where(lat_mosaic[:,0]==lat_lims[0])[0][0], np.where(lat_mosaic[:,0]==lat_lims[1])[0][0]]
    lon_mosaic_crop = lon_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
    lat_mosaic_crop = lat_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
    imgs_mosaic_crop = imgs_mosaic[lat_lim_idxs[1]:lat_lim_idxs[0], lon_lim_idxs[0]:lon_lim_idxs[1]]
    cmap = copy.copy(plt.get_cmap('magma_r'))
    imgs_mosaic_crop = np.ma.masked_where(imgs_mosaic_crop == 0, imgs_mosaic_crop)  # set mask where height is 0, to be converted to another color
    cmap.set_bad(color='skyblue')  # color where height == 0
    plt.figure(dpi=400)
    plt.title('Data sample example:\n Topography, location and fetch')
    bbox = ((lon_mosaic.min(),   lon_mosaic.max(),
             lat_mosaic.min(),  lat_mosaic.max()))
    bbox = ((lon_mosaic_crop.min(),   lon_mosaic_crop.max(),
             lat_mosaic_crop.min(),  lat_mosaic_crop.max()))
    # plt.xlim(bbox[0], bbox[1])
    # plt.ylim(bbox[2], bbox[3])
    imshow = plt.imshow(imgs_mosaic_crop, extent=bbox, zorder=0, cmap=cmap)
    plt.scatter(point_1[0], point_1[1], marker='o', facecolors='none', edgecolors='black', label='Measurement location')
    plt.scatter(point_2[0], point_2[1], marker='o', facecolors='none', edgecolors='black', s=4)
    plt.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], c='black', linestyle='--', label='Upstream fetch')
    cb = plt.colorbar(imshow)

    # plt.xlim([-50000, -20000])
    # plt.ylim([6.685E6, 6.715E6])
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    cb.set_label('Height [m]')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], handletextpad=0.1)
    plt.tight_layout(pad=0.05)
    plt.savefig('plots/2D_map_2_points_example.png')
    plt.show()
    pass


# TRASH
# plot_elevation_profile(point_1=[-35260., 6700201.], point_2=[-34501., 6690211.], step_distance=10, list_of_distances=None)



