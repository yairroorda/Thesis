import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pdal

from calculate import hag_to_ortho
from models import AOIPolygon, Point, ProjectConfig, ProjectPaths, RunConfig, RunPaths
from utils import get_logger, load_profile, write_metadata
from visualize import save_viewshed_as_tif, write_to_copc


def setup(project_cfg: ProjectConfig, project_paths: ProjectPaths, run_cfg: RunConfig):
    # setup run paths
    run_paths = RunPaths(project_paths, run_cfg.name)

    # setup logging
    profile_cfg = load_profile(project_cfg.profile)
    run_logger = get_logger(name="Main", logfile_path=run_paths.run_log, level=profile_cfg.logging_level)
    run_logger.info(f"Running viewshed '{run_cfg.name}' in project '{project_paths.name}'")
    run_logger.info(f"LoS settings: mode={run_cfg.los_mode} radius={run_cfg.los_radius} min_radius={run_cfg.los_start_radius} max_radius={run_cfg.los_end_radius} step_length={run_cfg.los_step_length}")

    return run_paths, run_logger


def calculate_3d_flight_height(
    project_cfg: ProjectConfig,
    project_paths: ProjectPaths,
    run_cfg: RunConfig,
    threshold: float = 0,
    resolution: float = 1.0,
) -> RunPaths:

    run_paths, run_logger = setup(project_cfg, project_paths, run_cfg)

    start_time = perf_counter()

    # Read visibility points from the 3D viewshed COPC
    pipeline = {
        "pipeline": [
            {"type": "readers.copc", "filename": str(run_paths.output_viewshed_copc_3d)},
            {"type": "filters.range", "limits": f"Visibility({threshold}:)"},
        ]
    }
    pl = pdal.Pipeline(json.dumps(pipeline))
    pl.execute()

    # Get bounds
    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs)
    min_x, min_y, max_x, max_y = aoi.bounds

    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    # ---- Rasterize the AOI to create an exact boolean mask ----
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    transform = from_origin(min_x - (resolution / 2.0), max_y + (resolution / 2.0), resolution, resolution)
    aoi_mask = rasterize(
        [(aoi, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    # If there are visible points, calculate their cell indices and minimum Z
    if pl.arrays and pl.arrays[0].size > 0:
        arr = pl.arrays[0]

        cols = np.floor((arr["X"] - min_x) / resolution).astype(np.int64)
        rows = np.floor((max_y - arr["Y"]) / resolution).astype(np.int64)

        # Ensure points are within grid bounds AND inside the precise AOI mask
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        valid &= aoi_mask[rows.clip(0, height - 1), cols.clip(0, width - 1)]

        flat_indices = rows[valid] * width + cols[valid]
        valid_z = arr["Z"][valid]

        order = np.argsort(flat_indices, kind="mergesort")
        flat_sorted = flat_indices[order]
        z_sorted = valid_z[order]

        unique_flat, first_idx = np.unique(flat_sorted, return_index=True)
        min_z_per_cell = np.minimum.reduceat(z_sorted, first_idx)
    else:
        unique_flat = np.array([], dtype=np.int64)
        min_z_per_cell = np.array([], dtype=np.float64)

    # Find the flat indices of all cells that are physically inside the AOI polygon
    valid_mask_indices = np.where(aoi_mask.ravel())[0]

    # Calculate X/Y coordinates ONLY for those valid cells
    out_rows = valid_mask_indices // width
    out_cols = valid_mask_indices % width

    out_x = min_x + (out_cols + 0.5) * resolution
    out_y = max_y - (out_rows + 0.5) * resolution

    # Create ceiling points only for valid AOI cells
    ceiling_points = [Point(x, y, run_cfg.z_height) for x, y in zip(out_x, out_y)]
    ceiling_points_ortho = hag_to_ortho(ceiling_points, input_path=project_paths.input_copc)

    # Initialize the final Z array with the true ceiling heights
    output_z = np.array([pt.z for pt in ceiling_points_ortho], dtype=np.float64)

    # Calculate ground elevation by subtracting the known HAG from the orthometric ceiling
    ground_z = output_z - run_cfg.z_height

    # Overwrite cells that had visible points.
    if unique_flat.size > 0:
        insert_indices = np.searchsorted(valid_mask_indices, unique_flat)
        output_z[insert_indices] = min_z_per_cell

    # Save as a new COPC file WITH the HAG dimension added
    dtype = [("X", "<f8"), ("Y", "<f8"), ("Z", "<f8"), ("HAG", "<f8")]
    out_arr = np.empty(len(valid_mask_indices), dtype=dtype)
    out_arr["X"] = out_x
    out_arr["Y"] = out_y
    out_arr["Z"] = output_z
    out_arr["HAG"] = output_z - ground_z

    out_copc_path = run_paths.folder / "flight_height.copc.laz"
    write_to_copc(out_arr, out_copc_path, project_cfg)

    # save as GeoTIFF
    save_viewshed_as_tif(
        x_coords=out_arr["X"],
        y_coords=out_arr["Y"],
        visibility_values=out_arr["HAG"],
        aoi=aoi,
        resolution=resolution,
        output_path=run_paths.output_flight_height_tif,
    )

    write_metadata(
        run_cfg,
        project_paths,
        run_paths,
        project_cfg.profile,
        project_cfg.crs,
        start_time,
    )

    run_logger.info("Run completed")
    return run_paths
