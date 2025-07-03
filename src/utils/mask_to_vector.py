from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from rasterio.crs import CRS
import numpy as np


def mask_to_vector(mask: np.ndarray, transform, crs=None, min_area=10) -> gpd.GeoDataFrame:
    if crs is None:
        raise ValueError("crs (좌표계)를 반드시 지정해야 합니다.")

    shapes_gen = shapes(mask.astype(np.uint8), mask=(mask > 0), transform=transform)

    geometries = []
    for geom, value in shapes_gen:
        if value == 1:
            polygon = shape(geom)
            if polygon.area >= min_area:
                geometries.append(polygon)

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    gdf["area"] = gdf.area

    return gdf
