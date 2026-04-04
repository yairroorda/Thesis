import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import tomllib

from calculate import Point, calculate_viewshed_2d, export_grid_to_copc
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
        "output_viewshed_copc": run_folder / "viewshed.copc.laz",
        "output_viewshed_tif": run_folder / "viewshed.tif",
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
    radius: float = 0.15,
    profile: Optional[str] = None,
    aoi_source: Optional[Path] = None,
    overwrite: bool = False,
    target_source: Optional[Path] = None,
) -> None:
    start_time = time.perf_counter()
    config, active_profile = load_config(profile)
    log_level = config.get("logging_level", "INFO")

    # Prepare isolated run folder under data
    run_name = name or "tmp"
    run_folder = prepare_run_folder("data", run_name, overwrite=overwrite)
    paths = _path_map(run_folder)

    global logger
    logger = get_logger(name="Main", logfile_path=paths["log"], level=log_level)
    logger.info(f"Using config profile: {active_profile}")
    logger.debug(f"Config: {config}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Run folder: {run_folder}")

    # AOI is stored in each run folder and reprojected to EPSG:28992 for processing.
    source_aoi_crs, aoi = _load_run_aoi(run_folder=run_folder, aoi_source=aoi_source)

    output_copc_path = paths["input_copc"]
    get_pointcloud_aoi(aoi, aoi_crs="EPSG:28992", include=dataset, output_path=output_copc_path)

    # Classify vegetation
    output_classified_path = paths["classified_copc"]
    if classification_method == "myria3d":
        logger.info("Delegating vegetation classification to Myria3D Pixi environment")
        result = subprocess.run(f"pixi run -e myria3d python src/segment.py {name} {classification_method}", shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Myria3D classification failed with exit code {result.returncode}.")

    elif classification_method == "rule-based":
        classify_vegetation_rule_based(output_copc_path, output_classified_path)

    else:
        raise ValueError(f"Unknown classification method: {classification_method}")

    # Generate building facades from roof edges
    output_facades_path = paths["facades_copc"]
    generate_facades(output_classified_path, output_facades_path)

    # Generate 2D viewshed
    target, target_source_used = _load_run_target(run_folder=run_folder, target_source=target_source)

    output_path = run_folder / "viewshed"
    _, _, visibility_points = calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        radius=radius,
        resolution=resolution,
        input_path=output_facades_path,
        output_path=output_path,
        z_offset=0.3,
    )

    save_viewshed_as_tif(
        x_coords=visibility_points["X"],
        y_coords=visibility_points["Y"],
        visibility_values=visibility_points["Visibility"],
        aoi=aoi,
        resolution=resolution,
        output_path=paths["output_viewshed_tif"],
    )

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
            "radius": radius,
            "resolution": resolution,
            "grid_points": int(len(visibility_points)),
            "files": {k: str(v) for k, v in paths.items()},
        },
    )

    for key, path in paths.items():
        if key not in config.get("save_files", []) and path.exists():
            path.unlink()


if __name__ == "__main__":
    NAME = None
    DATASET = ["AHN6", "AHN5"]  # Options: None (defaults to newest), or list of dataset names (e.g. ["AHN6", "AHN5"])
    CLASSIFICATION_METHOD = "myria3d"  # Options: "myria3d", "rule-based"
    RESOLUTION = 1
    RADIUS = 0.15
    PROFILE = "testing"  # Defaults to production when unset
    AOI_SOURCE = None  # "data/temp/aoi.geojson"  # Path("data/Groningen_plein.geojson")
    TARGET_SOURCE = None  # Path("data/target_point.copc.laz")
    OVERWRITE = True

    main(
        name=NAME,
        dataset=DATASET,
        classification_method=CLASSIFICATION_METHOD,
        resolution=RESOLUTION,
        radius=RADIUS,
        profile=PROFILE,
        aoi_source=AOI_SOURCE,
        target_source=TARGET_SOURCE,
        overwrite=OVERWRITE,
    )
