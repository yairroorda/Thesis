import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import tomllib

from calculate import (
    Point,
    calculate_flight_height,
    calculate_intervisibility,
    calculate_viewshed_2d,
    calculate_viewshed_for_grid,
    export_grid_to_copc,
    generate_grid,
    only_save_viewable_volume,
    sample_polygon_boundary,
)
from enhance_facades import generate_facades
from models import ProjectConfig, ProjectPaths, RunConfig, RunPaths
from query_copc import AOIPolygon, get_pointcloud_aoi
from segment import classify_vegetation_rule_based
from utils import get_logger, timed
from visualize import save_viewshed_as_tif

logger = get_logger(name="Main")


def load_profile(profile: str | None = None) -> tuple[dict, str]:
    config_path = Path(__file__).parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    active_profile = profile or os.environ.get("THESIS_PROFILE") or "production"
    if active_profile not in config:
        raise ValueError(f"Profile '{active_profile}' not found in config.toml")
    return config[active_profile], active_profile


def _load_or_create_project_aoi(paths: ProjectPaths, aoi_source: Path | None = None) -> tuple[str, AOIPolygon]:
    """Load AOI for the project and return source CRS and AOI in EPSG:28992."""
    if aoi_source:
        if not aoi_source.exists():
            raise FileNotFoundError(f"aoi_source does not exist: {aoi_source}")
        shutil.copy2(aoi_source, paths.aoi)
    elif not paths.aoi.exists():
        aoi_user = AOIPolygon.get_from_user(title=f"Draw AOI for project {paths.name}")
        aoi_user.save_to_file(paths.aoi, crs="EPSG:4326")

    aoi_loaded = AOIPolygon.get_from_file(paths.aoi)
    return aoi_loaded.crs, aoi_loaded.to_crs("EPSG:28992")


def _load_or_create_target(target_path: Path, target_source: Path | None, aoi: AOIPolygon) -> tuple[Point, str]:
    if target_source:
        if not target_source.exists():
            raise FileNotFoundError(f"target_source does not exist: {target_source}")
        if not target_path.exists():
            shutil.copy2(target_source, target_path)
        return Point.get_from_file(target_path), str(target_source)

    if target_path.exists():
        return Point.get_from_file(target_path), str(target_path)

    target = Point.get_from_user("Select target point for viewshed analysis", aoi=aoi)
    export_grid_to_copc([target], output_path=target_path)
    return target, "user"


def _write_metadata(run_cfg: RunConfig, project_paths: ProjectPaths, run_paths: RunPaths, active_profile: str, source_aoi_crs: str, start_time: float) -> None:
    metadata = {
        "project": {
            "name": project_paths.name,
            "folder": str(project_paths.folder),
            "profile": active_profile,
            "aoi_source_crs": source_aoi_crs,
            "processing_crs": "EPSG:28992",
            "classification_method": "cached",
        },
        "run": {
            "name": run_cfg.name,
            "folder": str(run_paths.folder),
            "runtime_seconds": round(time.perf_counter() - start_time, 3),
            "resolution": run_cfg.resolution,
            "z_height": run_cfg.z_height,
            "los": {
                "mode": run_cfg.los_mode,
                "radius": run_cfg.los_radius,
                "min_radius": run_cfg.los_start_radius,
                "max_radius": run_cfg.los_end_radius,
                "step_length": run_cfg.los_step_length,
            },
            "files": {
                "input_copc": str(project_paths.input_copc),
                "rescaled_copc": str(project_paths.rescaled_copc),
                "classified_copc": str(project_paths.classified_copc),
                "facades_copc": str(project_paths.facades_copc),
                "target_point_copc": str(run_paths.target_point_copc),
                "viewshed_2d_copc": str(run_paths.output_viewshed_copc_2d),
                "viewshed_3d_copc": str(run_paths.output_viewshed_copc_3d),
                "flight_height_tif": str(run_paths.output_flight_height_tif),
                "viewable_volume_copc": str(run_paths.viewable_volume_copc),
            },
        },
    }
    run_paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@timed("Prepare project")
def prepare_project(config: ProjectConfig) -> ProjectPaths:
    profile_cfg, active_profile = load_profile(config.profile)
    paths = ProjectPaths(config.name)

    log_level = profile_cfg.get("logging_level", "INFO")
    global logger
    logger = get_logger(name="Main", logfile_path=paths.project_log, level=log_level)

    logger.info(f"Using config profile: {active_profile}")
    logger.debug(f"Config: {profile_cfg}")
    logger.info(f"Project: {config.name}")
    logger.info(f"Project folder: {paths.folder}")

    if config.overwrite:
        for artifact in [paths.input_copc, paths.rescaled_copc, paths.classified_copc, paths.facades_copc]:
            if artifact.exists():
                artifact.unlink()

    source_aoi_crs, aoi = _load_or_create_project_aoi(paths=paths, aoi_source=config.aoi_source)
    logger.info(f"Project AOI ready (source CRS: {source_aoi_crs}, processing CRS: EPSG:28992)")

    if paths.is_prepared:
        logger.info(f"Project {config.name} is already prepared. Reusing cached preprocessing outputs.")
        return paths

    logger.info("Downloading project point cloud for AOI")
    get_pointcloud_aoi(aoi, aoi_crs="EPSG:28992", include=config.dataset, output_path=paths.input_copc)

    logger.info(f"Classifying vegetation using method: {config.classification_method}")
    if config.classification_method == "myria3d":
        result = subprocess.run(
            ["pixi", "run", "-e", "myria3d", "python", "src/segment.py", paths.name, config.classification_method],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Myria3D classification failed with exit code {result.returncode}.")
    elif config.classification_method == "rule-based":
        classify_vegetation_rule_based(paths.input_copc, paths.classified_copc)
    else:
        raise ValueError(f"Unknown classification method: {config.classification_method}")

    logger.info("Generating facade-enhanced point cloud")
    generate_facades(paths.classified_copc, paths.facades_copc)
    if not paths.facades_copc.exists():
        logger.warning("Facade generation produced no output. Falling back to classified cloud.")
        shutil.copy2(paths.classified_copc, paths.facades_copc)

    logger.info("Project preprocessing complete")
    return paths


@timed("Run 2D viewshed")
def calculate_2d_viewshed(project_paths: ProjectPaths, run_cfg: RunConfig, profile: str = "testing") -> RunPaths:
    profile_cfg, active_profile = load_profile(profile)
    run_paths = RunPaths(project_paths, run_cfg.name)

    run_logger = get_logger(name="Main", logfile_path=run_paths.run_log, level=profile_cfg.get("logging_level", "INFO"))

    source_aoi_crs = AOIPolygon.get_from_file(project_paths.aoi).crs
    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs("EPSG:28992")

    run_logger.info(f"Running viewshed '{run_cfg.name}' in project '{project_paths.name}'")
    run_logger.info(f"LoS settings: mode={run_cfg.los_mode} radius={run_cfg.los_radius} min_radius={run_cfg.los_start_radius} max_radius={run_cfg.los_end_radius} step_length={run_cfg.los_step_length}")

    start_time = time.perf_counter()

    target, target_source_used = _load_or_create_target(
        target_path=run_paths.target_point_copc,
        target_source=run_cfg.target_source,
        aoi=aoi,
    )

    viewshed_2d_path = run_paths.output_viewshed_copc_2d
    _, _, visibility_points = calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        radius=run_cfg.los_radius,
        radius_mode=run_cfg.los_mode,
        min_radius=run_cfg.los_start_radius,
        max_radius=run_cfg.los_end_radius,
        step_length=run_cfg.los_step_length,
        resolution=run_cfg.resolution,
        input_path=project_paths.facades_copc,
        output_path=viewshed_2d_path,
        z_offset=0.3,
    )
    viewshed_tif = run_paths.output_viewshed_tif_2d
    save_viewshed_as_tif(
        x_coords=visibility_points["X"],
        y_coords=visibility_points["Y"],
        visibility_values=visibility_points["Visibility"],
        aoi=aoi,
        resolution=run_cfg.resolution,
        output_path=viewshed_tif,
    )

    _write_metadata(run_cfg, project_paths, run_paths, active_profile, source_aoi_crs, start_time)

    run_logger.info("Run completed")
    return run_paths


@timed("Run viewshed")
def calculate_3d_viewshed(project_paths: ProjectPaths, run_cfg: RunConfig, profile: str = "testing") -> RunPaths:
    profile_cfg, active_profile = load_profile(profile)
    run_paths = RunPaths(project_paths, run_cfg.name)

    run_logger = get_logger(name="Main", logfile_path=run_paths.run_log, level=profile_cfg.get("logging_level", "INFO"))

    source_aoi_crs = AOIPolygon.get_from_file(project_paths.aoi).crs
    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs("EPSG:28992")

    run_logger.info(f"Running viewshed '{run_cfg.name}' in project '{project_paths.name}'")
    run_logger.info(f"LoS settings: mode={run_cfg.los_mode} radius={run_cfg.los_radius} min_radius={run_cfg.los_start_radius} max_radius={run_cfg.los_end_radius} step_length={run_cfg.los_step_length}")

    start_time = time.perf_counter()

    target, target_source_used = _load_or_create_target(
        target_path=run_paths.target_point_copc,
        target_source=run_cfg.target_source,
        aoi=aoi,
    )

    top_points = generate_grid(aoi, run_cfg.resolution, z_height=run_cfg.z_height, two_d=True)
    for pt in top_points:
        pt.z = run_cfg.z_height

    boundary_points = sample_polygon_boundary(aoi, sample_distance=run_cfg.resolution, z_height=0.0)
    wall_zs = np.arange(0, run_cfg.z_height + run_cfg.resolution, run_cfg.resolution)
    wall_points = [Point(pt.x, pt.y, z) for pt in boundary_points for z in wall_zs]
    grid_points = top_points + wall_points
    export_grid_to_copc(grid_points, output_path=run_paths.grid_shell_copc)

    calculate_viewshed_for_grid(
        target=target,
        grid_points=grid_points,
        cylinder_radius=run_cfg.los_radius,
        radius_mode=run_cfg.los_mode,
        min_radius=run_cfg.los_start_radius,
        max_radius=run_cfg.los_end_radius,
        step_length=run_cfg.los_step_length,
        input_path=project_paths.facades_copc,
        output_path=run_paths.output_viewshed_copc_3d,
        intervisibility_func=calculate_intervisibility,
        chunk_size=5,
    )

    only_save_viewable_volume(run_paths.output_viewshed_copc_3d, run_paths.viewable_volume_copc)

    _write_metadata(run_cfg, project_paths, run_paths, active_profile, source_aoi_crs, start_time)

    run_logger.info("Run completed")
    return run_paths


def calculate_3d_flight_height(project_paths: ProjectPaths, run_cfg: RunConfig, profile: str = "testing") -> RunPaths:

    profile_cfg, active_profile = load_profile(profile)
    run_paths = RunPaths(project_paths, run_cfg.name)

    run_logger = get_logger(name="Main", logfile_path=run_paths.run_log, level=profile_cfg.get("logging_level", "INFO"))

    source_aoi_crs = AOIPolygon.get_from_file(project_paths.aoi).crs
    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs("EPSG:28992")

    run_logger.info(f"Running viewshed '{run_cfg.name}' in project '{project_paths.name}'")
    run_logger.info(f"LoS settings: mode={run_cfg.los_mode} radius={run_cfg.los_radius} min_radius={run_cfg.los_start_radius} max_radius={run_cfg.los_end_radius} step_length={run_cfg.los_step_length}")

    start_time = time.perf_counter()

    calculate_flight_height(
        aoi=aoi,
        resolution=run_cfg.resolution,
        input_path=run_paths.output_viewshed_copc_3d,
        output_path=run_paths.output_flight_height_tif,
    )

    _write_metadata(run_cfg, project_paths, run_paths, active_profile, source_aoi_crs, start_time)

    run_logger.info("Run completed")
    return run_paths


def remove_intermediate_files(project_paths: ProjectPaths, run_paths: RunPaths, profile: str = "testing") -> None:
    profile_cfg, _ = load_profile(profile)

    files_to_remove = profile_cfg.get("remove", [])
    merged_file_map = {**project_paths.file_map, **run_paths.file_map}
    for file_key in files_to_remove:
        file_path = merged_file_map.get(file_key)
        if file_path and file_path.exists():
            logger.info(f"Removing intermediate file: {file_path}")
            file_path.unlink()


if __name__ == "__main__":
    if not Path("data/test_aoi.geojson").exists():
        aoi = AOIPolygon.get_from_user(title="Draw AOI for testing")
        aoi.save_to_file(Path("data/test_aoi.geojson"))

    project_cfg = ProjectConfig(
        name="refactor_test",
        dataset=["AHN5"],
        profile="testing",
        aoi_source=Path("data/test_aoi.geojson"),
        overwrite=False,
    )

    project_paths = prepare_project(project_cfg)

    run_config = RunConfig(name="refactor_test_run", resolution=1.0, los_mode="fixed", los_radius=0.15, z_height=20.0)

    run_paths = calculate_2d_viewshed(project_paths, run_config, profile=project_cfg.profile)
    run_paths = calculate_3d_viewshed(project_paths, run_config, profile=project_cfg.profile)
    run_paths = calculate_3d_flight_height(project_paths, run_config, profile=project_cfg.profile)
