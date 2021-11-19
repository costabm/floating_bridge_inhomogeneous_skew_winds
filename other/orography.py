"""
Created: June, 2021
Contact: bercos@vegvesen.no

Data gathered from:
https://hoydedata.no/LaserInnsyn/

Koordinatsystem
Alle prosjekter er lagret i EUREF89 og lokal UTM sone.
De nasjonale modellene er tilgjengelige i sone 33 for hele landet og 32 og 35 for områder der dette er aktuellt.
( https://hoydedata.no/LaserInnsyn/help_no/index.htm?context=130 )

Converting from UTM zone 33 to Lat Lon:
Option 1: https://www.engineeringtoolbox.com/utm-latitude-longitude-d_1370.html
Option 2 (favourite): http://rcn.montana.edu/Resources/Converter.aspx
Option 3 (gives slighty different results, inaccurate!?): https://stackoverflow.com/questions/343865/how-to-convert-from-utm-to-latlng-in-python-or-javascript

A direct elevation profile script can be found below, but it is slow and it doesn't work in the Bjornafjord coordinates tested:
https://www.geodose.com/2018/03/create-elevation-profile-generator-python.html

To plot the final result (a full picture from all individual geotiff files), run:
import matplotlib.pyplot as plt
lon_mosaic, lat_mosaic, imgs_mosaic = get_all_geotiffs_merged(fnames=fnames, fpositions=fpositions)
plt.figure(dpi=800)
plt.imshow(lat_mosaic)
plt.show()
"""

import numpy as np
from tifffile import tifffile
import os


# Manually introduce the names of the 4 tif files representing the dtm10 maps of Bjornafjord
fnames = ['dtm10_67m1_3_10m_z33.tif', 'dtm10_67m1_2_10m_z33.tif', 'dtm10_66m1_4_10m_z33.tif', 'dtm10_66m1_1_10m_z33.tif']  # top-left, top-right, bottom-left, bottom-right
fpositions = ['11', '12', '21', '22']   # 'row column' of each "tile" to compose the merged "mosaic"


# Mast coordinates Easting and Northing in UTM 32 (given in SBJ-01-C4-KJE-01-TN-005 Analysis of extreme wind at the fjord crosssing):
synn_EN_32 = [297558., 6672190.]
svar_EN_32 = [297966., 6666513.]
osp1_EN_32 = [292941., 6669471.]
osp2_EN_32 = [292989., 6669215.]
land_EN_32 = [298225., 6654672.]
neso_EN_32 = [303585., 6649851.]

# Mast coordinates, in UTM 33 (found iterativelly with http://rcn.montana.edu/Resources/Converter.aspx, by enforcing UTM 33 and matching the NATO UTM to that in UTM 32 -> 1 meter precision):
synn_EN_33 = [-34515., 6705758.]
svar_EN_33 = [-34625., 6700051.]
osp1_EN_33 = [-39375., 6703464.]
osp2_EN_33 = [-39350., 6703204.]
land_EN_33 = [-35446., 6688200.]
neso_EN_33 = [-30532., 6682896.]


def get_1_geotiff(tifpath, tfwpath, trim=True):
    """
    Reads one geotiff file
    Returns the coordinates in UTM zone 33 [longitudes, latitudes] of the centre of each pixel, as well as each pixel height
    """
    img = np.array(tifffile.imread(tifpath), dtype='float64')
    # removing the 1% overlapping cells.
    img[img<0] = 0  # some files have very large negative values that shoud just be 0
    if trim:
        img = img[:5000, :5000]  # trimming the overlapping part of the map
    img_descrip = [eval(i) for i in open(tfwpath, 'r').read().splitlines()]
    assert img.shape[0] == img.shape[1] and img_descrip[0] == abs(img_descrip[3]), "Warning: Image is not a square so code needs to be updated"
    img_dtm = img_descrip[0]
    img_lon = np.array([img_descrip[4] + x * img_dtm for x in range(img.shape[1])], dtype='float64')
    img_lat = np.array([img_descrip[5] - y * img_dtm for y in range(img.shape[1])], dtype='float64')
    img_lons, img_lats = np.meshgrid(img_lon, img_lat)
    return img_lons, img_lats, img


def get_all_geotiffs_merged(fnames=fnames, fpositions=fpositions, fpath=os.path.join(r'C:\Users\bercos\PycharmProjects\Metocean', 'basis', 'dtm10', 'data')):
    """
    Returns the matrix of longitudes, latitudes and heights, from all merged geotiff files (each one as a tile, with row and column number, to be merged into a mosaic)
    """
    positions = np.array([(int(rc[0]), int(rc[1])) for rc in fpositions])  # converting from strings to array
    n_cols = max(positions[:,1])
    n_rows = max(positions[:,0])
    imgs_lon_row = []
    imgs_lat_row = []
    imgs_row = []
    for row in range(n_rows):
        imgs_lon = []
        imgs_lat = []
        imgs = []
        for col in range(n_cols):
            file_idx = fpositions.index(str(row+1)+str(col+1))
            tifpath = os.path.join(fpath, fnames[file_idx])
            tfwpath = tifpath[:-3] + 'tfw'
            img_lons, img_lats, img = get_1_geotiff(tifpath, tfwpath, trim=True)
            imgs_lon.append(img_lons)
            imgs_lat.append(img_lats)
            imgs.append(img)
        imgs_lon_row.append(np.hstack(imgs_lon))
        imgs_lat_row.append(np.hstack(imgs_lat))
        imgs_row.append(np.hstack(imgs))
    lon_mosaic = np.vstack(imgs_lon_row)
    lat_mosaic = np.vstack(imgs_lat_row)
    imgs_mosaic = np.vstack(imgs_row)
    return lon_mosaic, lat_mosaic, imgs_mosaic


