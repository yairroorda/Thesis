from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import Polygon

from utils import get_logger, timed

logger = get_logger("Visualize")


@timed("Saving viewshed as GeoTIFF")
def save_viewshed_as_tif(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    visibility_values: np.ndarray,
    aoi: Polygon,
    resolution: float,
    output_path: Path,
) -> None:
    """
    Save viewshed as a GeoTIFF by assigning each point to its raster cell and
    storing the maximum visibility of all points that fall within that cell.
    """
    min_x, min_y, max_x, max_y = aoi.bounds

    # Calculate raster dimensions
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    visibility_values = np.asarray(visibility_values, dtype=np.float32)

    if x_coords.size == 0 or y_coords.size == 0 or visibility_values.size == 0:
        raise ValueError("No points or visibility values provided for raster export.")
    if not (x_coords.size == y_coords.size == visibility_values.size):
        raise ValueError("x_coords, y_coords, and visibility_values must have the same length.")

    # Shift the origin by half a cell so point are centered per pixel.
    transform = from_origin(min_x - (resolution / 2.0), max_y + (resolution / 2.0), resolution, resolution)

    aoi_mask = rasterize(
        [(aoi, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    raster_data = np.full((height, width), -1.0, dtype=np.float32)

    cols = np.rint((x_coords - min_x) / resolution).astype(np.int64)
    rows = np.rint((max_y - y_coords) / resolution).astype(np.int64)

    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    valid &= aoi_mask[rows.clip(0, height - 1), cols.clip(0, width - 1)]

    if np.any(valid):
        flat_indices = rows[valid] * width + cols[valid]
        np.maximum.at(raster_data.ravel(), flat_indices, visibility_values[valid])

    raster_data[~aoi_mask] = -1.0

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster_data.dtype,
        crs="EPSG:28992",  # Standard Dutch CRS (Amersfoort / RD New)
        transform=transform,
        nodata=-1.0,
    ) as dst:
        dst.write(raster_data, 1)

    logger.info(f"Saved 2D viewshed GeoTIFF to {output_path}")
