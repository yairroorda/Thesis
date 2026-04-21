import json
from pathlib import Path
from typing import Literal

import numpy as np
import pdal
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from models import AOIPolygon, ProjectPaths, RunConfig, RunPaths
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
    run_paths: RunPaths,
    run_cfg: RunConfig,
    project_paths: ProjectPaths,
    file_type: VoxelFile = "copc",
) -> None:
    """
    Save viewshed as a voxel grid by assigning each point to its voxel and
    storing the maximum visibility of all points that fall within that voxel.
    """
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(run_paths.output_viewshed_copc_3d)}]}))
    pipeline.execute()
    chunks = pipeline.arrays
    if not chunks:
        arr = np.empty(0, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Visibility", "f4")])
        write_pipeline = {
            "pipeline": [
                {
                    "type": "writers.copc",
                    "filename": str(run_paths.output_viewshed_voxel_grid_3d),
                    "forward": "all",
                    "extra_dims": "all",
                }
            ]
        }
        pdal.Pipeline(json.dumps(write_pipeline), arrays=[arr]).execute()
        logger.info(f"Saved 3D viewshed voxel grid in COPC format to {run_paths.output_viewshed_voxel_grid_3d}")
        return

    min_x = min(np.min(chunk["X"]) for chunk in chunks)
    max_x = max(np.max(chunk["X"]) for chunk in chunks)
    min_y = min(np.min(chunk["Y"]) for chunk in chunks)
    max_y = max(np.max(chunk["Y"]) for chunk in chunks)
    min_z = min(np.min(chunk["Z"]) for chunk in chunks)
    max_z = max(np.max(chunk["Z"]) for chunk in chunks)
    width = int(np.floor((max_x - min_x) / run_cfg.resolution)) + 1
    height = int(np.floor((max_y - min_y) / run_cfg.resolution)) + 1
    depth = int(np.floor((max_z - min_z) / run_cfg.resolution)) + 1

    transform = from_origin(min_x - (run_cfg.resolution / 2.0), max_y + (run_cfg.resolution / 2.0), run_cfg.resolution, run_cfg.resolution)

    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs("EPSG:28992")
    aoi_mask = rasterize(
        [(aoi, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    voxel_max: dict[int, float] = {}
    for chunk in chunks:
        cols = np.floor((chunk["X"] - min_x) / run_cfg.resolution).astype(np.int64)
        rows = np.floor((max_y - chunk["Y"]) / run_cfg.resolution).astype(np.int64)
        depths = np.floor((chunk["Z"] - min_z) / run_cfg.resolution).astype(np.int64)
        vis = chunk["Visibility"]

        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width) & (depths >= 0) & (depths < depth)
        valid &= aoi_mask[rows.clip(0, height - 1), cols.clip(0, width - 1)]

        if not np.any(valid):
            continue

        flat_indices = (depths[valid] * height * width) + (rows[valid] * width) + cols[valid]
        valid_vis = vis[valid]
        order = np.argsort(flat_indices, kind="mergesort")
        flat_sorted = flat_indices[order]
        vis_sorted = valid_vis[order]
        unique_flat, first_idx = np.unique(flat_sorted, return_index=True)
        chunk_max = np.maximum.reduceat(vis_sorted, first_idx)

        for flat, value in zip(unique_flat.tolist(), chunk_max.tolist()):
            existing = voxel_max.get(flat)
            if existing is None or value > existing:
                voxel_max[flat] = float(value)

    if file_type == "copc":
        if not voxel_max:
            arr = np.empty(0, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Visibility", "f4")])
        else:
            unique_flat = np.fromiter(voxel_max.keys(), dtype=np.int64)
            max_vis = np.fromiter(voxel_max.values(), dtype=np.float32)

            voxels_per_layer = height * width
            z_idx = unique_flat // voxels_per_layer
            rem = unique_flat % voxels_per_layer
            y_idx = rem // width
            x_idx = rem % width

            arr = np.empty(unique_flat.size, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Visibility", "f4")])
            arr["X"] = min_x + (x_idx + 0.5) * run_cfg.resolution
            arr["Y"] = max_y - (y_idx + 0.5) * run_cfg.resolution
            arr["Z"] = min_z + (z_idx + 0.5) * run_cfg.resolution
            arr["Visibility"] = max_vis

        write_pipeline = {
            "pipeline": [
                {
                    "type": "writers.copc",
                    "filename": str(run_paths.output_viewshed_voxel_grid_3d),
                    "forward": "all",
                    "extra_dims": "all",
                }
            ]
        }
        pdal.Pipeline(json.dumps(write_pipeline), arrays=[arr]).execute()
        logger.info(f"Saved 3D viewshed voxel grid in COPC format to {run_paths.output_viewshed_voxel_grid_3d}")

    elif file_type == "vti":
        import pyvista as pv

        voxel_data = np.full((depth, height, width), -1.0, dtype=np.float32)
        if voxel_max:
            flat_indices = np.fromiter(voxel_max.keys(), dtype=np.int64)
            max_vis = np.fromiter(voxel_max.values(), dtype=np.float32)
            voxel_data.ravel()[flat_indices] = max_vis

        grid = pv.ImageData(
            dimensions=(width + 1, height + 1, depth + 1),
            spacing=(run_cfg.resolution, run_cfg.resolution, run_cfg.resolution),
            origin=(min_x, min_y, min_z),
        )
        grid.cell_data["visibility"] = np.transpose(voxel_data, (2, 1, 0)).ravel(order="F")
        output_vti_path = run_paths.folder / "viewshed_3d_voxel.vti"
        grid.save(output_vti_path)
        logger.info(f"Saved 3D viewshed voxel grid in VTI format to {output_vti_path}")


if __name__ == "__main__":
    project_root = Path("data/evaluate_vegetation_influence_veg_eval_01_1776769328")
    run_folder = project_root / "run_01_without"

    project_paths = ProjectPaths("evaluate_vegetation_influence_veg_eval_01_1776769328")
    project_paths.aoi = project_root / "aoi.geojson"

    run_cfg = RunConfig(resolution=2.0)
    run_paths = RunPaths(project_paths, "run_01_without")
    run_paths.output_viewshed_copc_3d = run_folder / "viewshed_3d.copc.laz"
    run_paths.output_viewshed_voxel_grid_3d = run_folder / "viewshed_3d.copc_voxel.laz"

    save_viewshed_as_voxel_grid(
        run_paths=run_paths,
        run_cfg=run_cfg,
        project_paths=project_paths,
        file_type="copc",
    )
