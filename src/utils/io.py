import rasterio
import geopandas as gpd
import os
import glob


def load_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1).astype('float32')
        transform = src.transform
        crs = src.crs
    return array, transform, crs


def save_raster(array, transform, crs, save_path, dtype='float32'):
    with rasterio.open(
        save_path, 'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(array, 1)


def load_vector(path):
    return gpd.read_file(path)


def export_csv(df, path):
    df.to_csv(path, index=False, encoding="euc-kr")


def in_dir(folder):
    tif_files = glob.glob(os.path.join(folder, "*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {folder}")
    return tif_files[0]
