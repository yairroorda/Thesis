import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tomllib

from calculate import Point, calculate_flight_height, calculate_intervisibility, calculate_viewshed_2d, calculate_viewshed_for_grid, export_grid_to_copc, generate_grid, only_save_viewable_volume, sample_polygon_boundary
from enhance_facades import generate_facades
from query_copc import AOIPolygon, get_pointcloud_aoi
from segment import classify_vegetation_rule_based
from utils import get_logger, prepare_run_folder, timed
from visualize import save_viewshed_as_tif


def load_config(profile: Optional[str] = None):
    config_path = Path(__file__).parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    profile = profile or os.environ.get("THESIS_PROFILE") or "production"
    if profile not in config:
        raise ValueError(f"Profile '{profile}' not found in config.toml")
    return config[profile], profile


def _path_map(run_folder: Path) -> dict[str, Path]:
    return {
        "input_copc": run_folder / "input.copc.laz",
        "classified_copc": run_folder / "classified.copc.laz",
        "rescaled_copc": run_folder / "rescaled.copc.laz",
        "facades_copc": run_folder / "facades.copc.laz",
        "target_point_copc": run_folder / "target_point.copc.laz",
        "grid_shell_copc": run_folder / "grid_points_3d_shell.copc.laz",
        "output_viewshed_copc_2d": run_folder / "viewshed_2d.copc.laz",
        "output_viewshed_copc_3d": run_folder / "viewshed_3d.copc.laz",
        "output_viewshed_tif": run_folder / "viewshed.tif",
        "output_flight_height_tif": run_folder / "flight_height.tif",
        "aoi_geojson": run_folder / "aoi.geojson",
        "metadata": run_folder / "metadata.json",
        "log": run_folder / "run.log",
    }


def _load_run_aoi(run_folder: Path, aoi_source: Optional[Path] = None) -> tuple[str, AOIPolygon]:
    """Load AOI for this run and return source CRS and AOI reprojected to EPSG:28992."""
    run_aoi_path = run_folder / "aoi.geojson"

    if aoi_source:
        if not aoi_source.exists():
            raise FileNotFoundError(f"aoi_source does not exist: {aoi_source}")
        shutil.copy2(aoi_source, run_aoi_path)
    elif not run_aoi_path.exists():
        aoi_user = AOIPolygon.get_from_user()
        aoi_user.save_to_file(run_aoi_path, crs="EPSG:4326")

    aoi_loaded = AOIPolygon.get_from_file(run_aoi_path)
    return aoi_loaded.crs, aoi_loaded.to_crs("EPSG:28992")


def _save_metadata(metadata_path: Path, metadata: dict) -> None:
    """Save metadata to JSON."""
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_run_target(run_folder: Path, target_source: Optional[Path] = None) -> tuple[Point, str]:
    """Load target for this run and return (target_nap, source_label)."""
    run_target_path = run_folder / "target_point.copc.laz"

    if target_source:
        if not target_source.exists():
            raise FileNotFoundError(f"target_source does not exist: {target_source}")
        shutil.copy2(target_source, run_target_path)
        return Point.get_from_file(run_target_path), str(target_source)

    # Same behavior as AOI source fallback: prompt user and persist in run folder.
    target = Point.get_from_user("Select target point for viewshed analysis")
    export_grid_to_copc([target], output_path=run_target_path)
    return target, "user"


@timed("Total processing")
def main(
    name: str = "test",
    dataset: Optional[list[str]] = None,
    classification_method: str = "myria3d",
    resolution: float = 0.5,
    los_mode: str = "fixed",
    los_radius: float = 0.15,
    los_start_radius: float = 0.15,
    los_end_radius: float = 0.15,
    los_step_length: float = 0.10,
    profile: Optional[str] = None,
    aoi_source: Optional[Path] = None,
    overwrite: bool = False,
    target_source: Optional[Path] = None,
    z_height: float = 50.0,
) -> None:

    start_time = time.perf_counter()
    config, active_profile = load_config(profile)
    log_level = config.get("logging_level", "INFO")

    if los_mode == "fixed":
        los_start_radius = los_radius
        los_end_radius = los_radius

    # Prepare isolated run folder under data
    run_name = name or "tmp"
    # run_folder = prepare_run_folder("data", run_name, overwrite=overwrite)
    run_folder = Path("data") / run_name
    paths = _path_map(run_folder)

    global logger
    logger = get_logger(name="Main", logfile_path=paths["log"], level=log_level)
    logger.info(f"Using config profile: {active_profile}")
    logger.debug(f"Config: {config}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Run folder: {run_folder}")
    logger.info(f"LoS settings: mode={los_mode} radius={los_radius} min_radius={los_start_radius} max_radius={los_end_radius} step_length={los_step_length}")

    # AOI is stored in each run folder and reprojected to EPSG:28992 for processing.
    source_aoi_crs, aoi = _load_run_aoi(run_folder=run_folder, aoi_source=aoi_source)

    output_copc_path = paths["input_copc"]
    get_pointcloud_aoi(aoi, aoi_crs="EPSG:28992", include=dataset, output_path=output_copc_path)

    # Classify vegetation
    output_classified_path = paths["classified_copc"]
    if classification_method == "myria3d":
        logger.info("Delegating vegetation classification to Myria3D Pixi environment")
        result = subprocess.run(f"pixi run -e myria3d python src/segment.py {run_name} {classification_method}", shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Myria3D classification failed with exit code {result.returncode}.")

    elif classification_method == "rule-based":
        classify_vegetation_rule_based(output_copc_path, output_classified_path)

    else:
        raise ValueError(f"Unknown classification method: {classification_method}")

    # Generate building facades from roof edges
    output_facades_path = paths["facades_copc"]
    generate_facades(output_classified_path, output_facades_path)

    target, target_source_used = _load_run_target(run_folder=run_folder, target_source=target_source)

    # Generate 2D viewshed
    export_grid_to_copc([target], output_path=paths["target_point_copc"])

    output_path = paths["output_viewshed_copc_2d"]
    _, _, visibility_points = calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        radius=los_radius,
        radius_mode=los_mode,
        min_radius=los_start_radius,
        max_radius=los_end_radius,
        step_length=los_step_length,
        resolution=resolution,
        input_path=output_facades_path,
        output_path=output_path,
        z_offset=0.3,
    )

    grid_points = int(len(visibility_points))

    save_viewshed_as_tif(
        x_coords=visibility_points["X"],
        y_coords=visibility_points["Y"],
        visibility_values=visibility_points["Visibility"],
        aoi=aoi,
        resolution=resolution,
        output_path=output_path.with_suffix(".tif"),
    )

    # Generate 3D viewshed
    # Only use the shell: top surface and vertical walls
    # Top surface (z = z_height)
    top_points = generate_grid(aoi, resolution, z_height=z_height, two_d=True)
    for pt in top_points:
        pt.z = z_height
    # Vertical walls: sample boundary at multiple heights
    boundary_points = sample_polygon_boundary(aoi, sample_distance=resolution, z_height=0.0)
    wall_zs = np.arange(0, z_height + resolution, resolution)
    wall_points = [Point(pt.x, pt.y, z) for pt in boundary_points for z in wall_zs]
    # Combine shell points
    grid_points = top_points + wall_points
    export_grid_to_copc(grid_points, output_path=paths["grid_shell_copc"])

    # Compute 3D viewshed for shell grid points
    chunk_size = 5
    viewshed_output_path = paths["output_viewshed_copc_3d"]
    calculate_viewshed_for_grid(
        target=target,
        grid_points=grid_points,
        cylinder_radius=los_radius,
        radius_mode=los_mode,
        min_radius=los_start_radius,
        max_radius=los_end_radius,
        step_length=los_step_length,
        input_path=output_facades_path,
        output_path=viewshed_output_path,
        intervisibility_func=calculate_intervisibility,
        chunk_size=chunk_size,
    )

    # Compute flight height ceiling from the 3D viewshed COPC
    flight_height_output_path = paths["output_flight_height_tif"]
    _, _, height_points = calculate_flight_height(
        aoi=aoi,
        resolution=resolution,
        input_path=viewshed_output_path,
        output_path=flight_height_output_path,
    )

    logger.info(f"Flight height GeoTIFF written to {flight_height_output_path}")

    grid_points = int(len(height_points))

    _save_metadata(
        paths["metadata"],
        {
            "name": run_name,
            "run_folder": str(run_folder),
            "profile": active_profile,
            "aoi_source_crs": source_aoi_crs,
            "EPSG": "EPSG:28992",
            "runtime_seconds": round(time.perf_counter() - start_time, 3),
            "classification_method": classification_method,
            "dataset": dataset,
            "target_source": target_source_used,
            "target_point": {"x": target.x, "y": target.y, "z": target.z},
            "los": {
                "mode": los_mode,
                "radius": los_radius,
                "min_radius": los_start_radius,
                "max_radius": los_end_radius,
                "step_length": los_step_length,
            },
            "resolution": resolution,
            "grid_points": grid_points,
            "files": {k: str(v) for k, v in paths.items()},
        },
    )

    only_save_viewable_volume(paths["output_viewshed_copc_3d"], paths["output_viewshed_copc_3d"].with_name("viewshed_3d_viewable_volume.copc.laz"))

    # Remove unwanted files according to config
    delete = config.get("remove", [])
    for key, path in paths.items():
        if key in delete and path.exists():
            path.unlink()

    logger.info(f"Flight height GeoTIFF written to {flight_height_output_path}")


if __name__ == "__main__":
    NAME = "Delft_bouwkunde"
    DATASET = ["AHN5"]  # Options: None (defaults to newest), or list of dataset names (e.g. ["AHN6", "AHN5"])
    CLASSIFICATION_METHOD = "myria3d"  # Options: "myria3d", "rule-based"
    RESOLUTION = 2
    LOS_MODE = "fixed"  # Options: "fixed", "widening_linear"
    LOS_RADIUS = 0.15
    LOS_START_RADIUS = 0.15
    LOS_END_RADIUS = 0.15
    LOS_STEP_LENGTH = 0.15
    PROFILE = "testing"  # Defaults to production when unset
    AOI_SOURCE = Path("data/example_aoi/Delft_bouwkunde.geojson")  # Path("data/example_aoi/Groningen_plein.geojson")
    TARGET_SOURCE = Path("data/Delft_bouwkunde/target_point_Delft.copc.laz")  # Path("data/target_point.copc.laz")
    OVERWRITE = True
    Z_HEIGHT = 50.0

    # base run

    main(
        name=NAME,
        dataset=DATASET,
        classification_method=CLASSIFICATION_METHOD,
        resolution=RESOLUTION,
        los_mode=LOS_MODE,
        los_radius=LOS_RADIUS,
        los_start_radius=LOS_START_RADIUS,
        los_end_radius=LOS_END_RADIUS,
        los_step_length=LOS_STEP_LENGTH,
        profile=PROFILE,
        aoi_source=AOI_SOURCE,
        target_source=TARGET_SOURCE,
        overwrite=OVERWRITE,
        z_height=Z_HEIGHT,
    )

    # test multiple runs with increasing max_radius to see how it affects 2d visibility

    # for LOS_END_RADIUS in [0.10, 0.20, 0.30, 0.40, 0.50, 1, 3, 10]:
    #     main(
    #         name=NAME + f"_maxradius_{LOS_END_RADIUS}",
    #         dataset=DATASET,
    #         classification_method=CLASSIFICATION_METHOD,
    #         resolution=RESOLUTION,
    #         los_mode=LOS_MODE,
    #         los_radius=LOS_RADIUS,
    #         los_start_radius=LOS_START_RADIUS,
    #         los_end_radius=LOS_END_RADIUS,
    #         los_step_length=LOS_STEP_LENGTH,
    #         profile=PROFILE,
    #         aoi_source=AOI_SOURCE,
    #         target_source=TARGET_SOURCE,
    #         overwrite=OVERWRITE,
    #         z_height=Z_HEIGHT,
    #     )
