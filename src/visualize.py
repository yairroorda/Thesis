import json
from pathlib import Path
from typing import Literal

import numpy as np
import pdal
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from query_copc import AOIPolygon
from utils import get_logger, timed

logger = get_logger("Visualize")

type VoxelFile = Literal["copc", "vti"]


@timed("Saving viewshed as GeoTIFF")
def save_viewshed_as_tif(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    visibility_values: np.ndarray,
    aoi: AOIPolygon,
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


@timed("Saving viewshed as voxel grid")
def save_viewshed_as_voxel_grid(
    input_path: Path,
    aoi: AOIPolygon,
    resolution: float,
    file_type: VoxelFile,
    output_path: Path,
    input_crs: str = "EPSG:28992",
) -> None:
    """
    Save viewshed as a voxel grid by assigning each point to its voxel and
    storing the maximum visibility of all points that fall within that voxel.
    """
    reader_type = "readers.copc" if ".copc" in input_path.name.lower() else "readers.las"
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": reader_type, "filename": str(input_path)}]}))
    pipeline.execute()
    points = np.concatenate(pipeline.arrays)
    x_coords, y_coords, z_coords = points["X"], points["Y"], points["Z"]
    visibility_values = points["Visibility"]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_z, max_z = np.min(z_coords), np.max(z_coords)
    width = int(np.floor((max_x - min_x) / resolution)) + 1
    height = int(np.floor((max_y - min_y) / resolution)) + 1
    depth = int(np.floor((max_z - min_z) / resolution)) + 1

    transform = from_origin(min_x - (resolution / 2.0), max_y + (resolution / 2.0), resolution, resolution)

    if aoi.crs != input_crs:
        logger.debug(f"Reprojecting AOI from {aoi.crs} to {input_crs} for voxel grid export")
        aoi = aoi.to_crs(input_crs)

    aoi_mask = rasterize(
        [(aoi, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    voxel_data = np.full((depth, height, width), -1.0, dtype=np.float32)

    cols = np.floor((x_coords - min_x) / resolution).astype(np.int64)
    rows = np.floor((max_y - y_coords) / resolution).astype(np.int64)
    depths = np.floor((z_coords - min_z) / resolution).astype(np.int64)

    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width) & (depths >= 0) & (depths < depth)
    valid &= aoi_mask[rows.clip(0, height - 1), cols.clip(0, width - 1)]

    flat_indices = (depths[valid] * height * width) + (rows[valid] * width) + cols[valid]
    np.maximum.at(voxel_data.ravel(), flat_indices, visibility_values[valid])

    if file_type == "copc":
        mask = voxel_data >= 0
        z_idx, y_idx, x_idx = np.where(mask)
        arr = np.empty(mask.sum(), dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Visibility", "f4")])
        arr["X"] = min_x + (x_idx + 0.5) * resolution
        arr["Y"] = max_y - (y_idx + 0.5) * resolution
        arr["Z"] = min_z + (z_idx + 0.5) * resolution
        arr["Visibility"] = voxel_data[mask]
        pdal.Pipeline(json.dumps({"pipeline": [{"type": "writers.copc", "filename": str(output_path), "forward": "all", "extra_dims": "all"}]}), arrays=[arr]).execute()
        logger.info(f"Saved 3D viewshed voxel grid in COPC format to {output_path}")

    elif file_type == "vti":
        import pyvista as pv

        grid = pv.ImageData(
            dimensions=(width + 1, height + 1, depth + 1),
            spacing=(resolution, resolution, resolution),
            origin=(min_x, min_y, min_z),
        )
        grid.cell_data["visibility"] = np.transpose(voxel_data, (2, 1, 0)).ravel(order="F")
        grid.save(output_path)
        logger.info(f"Saved 3D viewshed voxel grid in VTI format to {output_path}")


if __name__ == "__main__":
    # Example usage:

    save_viewshed_as_voxel_grid(
        input_path=Path("data/Delft_bouwkunde/viewshed_3d.copc.laz"),
        aoi=AOIPolygon.get_from_file(Path("data/Delft_bouwkunde/aoi.geojson")),
        resolution=2.0,
        file_type="copc",
        output_path=Path("data/Delft_bouwkunde/viewshed_3d_voxel.copc"),
        input_crs="EPSG:28992",
    )
