import json
import logging
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
from rich.console import Console
from shapely import affinity
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from models import AOIPolygon, Point, ProfileConfig, ProjectPaths, RunPaths

LOGGER_LEVEL = logging.DEBUG

console = Console()


def load_profile(profile: str | None = None) -> ProfileConfig:
    import tomllib

    config_path = Path(__file__).parent.parent / "config.toml"

    # load config.toml into memory
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if profile not in config:
        raise ValueError(f"Profile '{profile}' not found in config.toml")

    return ProfileConfig(
        name=profile,
        logging_level=config[profile].get("logging_level", "INFO"),
        remove=config[profile].get("remove", []),
    )


def remove_intermediate_files(project_paths: ProjectPaths, run_paths: RunPaths, profile: str = "testing") -> None:
    profile_cfg = load_profile(profile)

    project_files = {
        "aoi": project_paths.aoi,
        "input_copc": project_paths.input_copc,
        "rescaled_copc": project_paths.rescaled_copc,
        "classified_copc": project_paths.classified_copc,
        "facades_copc": project_paths.facades_copc,
        "project_log": project_paths.project_log,
    }
    run_files = {
        "run_log": run_paths.run_log,
        "metadata": run_paths.metadata,
        "target_point_copc": run_paths.target_point_copc,
        "output_viewshed_copc_2d": run_paths.output_viewshed_copc_2d,
        "output_viewshed_tif_2d": run_paths.output_viewshed_tif_2d,
        "grid_shell_copc": run_paths.grid_shell_copc,
        "output_viewshed_copc_3d": run_paths.output_viewshed_copc_3d,
        "output_viewshed_voxel_grid_3d": run_paths.output_viewshed_voxel_grid_3d,
        "output_flight_height_tif": run_paths.output_flight_height_tif,
        "viewable_volume_copc": run_paths.viewable_volume_copc,
    }

    files_to_remove = profile_cfg.get("remove", [])
    merged_file_map = {**project_files, **run_files}
    for file_key in files_to_remove:
        file_path = merged_file_map.get(file_key)
        if file_path and file_path.exists():
            file_path.unlink()


def write_metadata(run_cfg, project_paths: Path, run_paths: Path, active_profile: str, source_aoi_crs: str, start_time: float) -> None:
    """Write run metadata to JSON file."""
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


@contextmanager
def status_spinner(message="Processing..."):
    """Context manager to show a rich spinner during long tasks."""
    with console.status(message) as status:
        yield status


def timed(label="No label provided"):
    """Decorator that logs elapsed time for a function call."""

    def decorator(func):
        display = label or func.__name__

        @wraps(func)  # Used to preserve the original function's metadata (like name and docstring)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                log = get_logger("Timing")
                log.info(f"{display}: {elapsed:.3f}s")

        return wrapper

    return decorator


def get_logger(name="thesis", logfile_path=None, level=None):
    """Return a configured logger (idempotent). Optionally add a file handler."""
    logger = logging.getLogger(name)
    formatter = logging.Formatter("[%(levelname)s] | %(name)s | %(message)s")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = True
    if logfile_path:
        file_target = str(Path(logfile_path).resolve())
        has_file_handler = any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == file_target for h in logger.handlers)
        if not has_file_handler:
            file_handler = logging.FileHandler(file_target)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(LOGGER_LEVEL)
    return logger


# Configure cloudfetch logging
_cloudfetch_logger = logging.getLogger("cloudfetch")
_cloudfetch_logger.setLevel(logging.INFO)
if not _cloudfetch_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] | %(name)s | %(message)s"))
    _cloudfetch_logger.addHandler(_handler)


def prepare_run_folder(base_dir: str | Path, run_name: str, overwrite: bool = False) -> Path:
    """Create isolated run folder data/<run_name>, with overwrite support."""
    folder = Path(base_dir) / run_name
    if folder.exists():
        if not overwrite:
            raise FileExistsError(f"Run folder '{folder}' already exists. Use overwrite to replace it.")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def compare(
    logger: logging.Logger,
    func_old,
    func_new,
    *args,
    runs: int = 1,
    **kwargs,
) -> dict[str, float]:
    start = time.perf_counter()
    for _ in range(runs):
        result1 = func_old(*args, **kwargs)
    end = time.perf_counter()
    old_time = end - start

    start_2 = time.perf_counter()
    for _ in range(runs):
        result2 = func_new(*args, **kwargs)
    end_2 = time.perf_counter()
    new_time = end_2 - start_2

    speedup = old_time / new_time
    logger.info(f"Old time: {old_time:.3f}s")
    logger.info(f"New time: {new_time:.3f}s")
    logger.info(f"Speedup: {speedup:.2f}x faster")

    if np.array_equal(result1, result2):
        logger.info("Both functions produced the same results.")
    else:
        logger.error("Functions produced different results!")
        logger.debug(f"Length of result 1: {len(result1)}")
        logger.debug(f"Length of result 2: {len(result2)}")
        logger.debug(f"Result 1: {result1}")
        logger.debug(f"Result 2: {result2}")


def generate_benchmark_aois(
    size: float | tuple[float, float],
    area: AOIPolygon,
    seed: int = 42,
) -> Iterator[AOIPolygon]:
    """Yield random square AOIPolygons sampled from the area bounds."""
    rng = np.random.default_rng(seed)

    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError("size must be a number or a 2-item range")
        size_min = float(size[0])
        size_max = float(size[1])
        if size_min <= 0 or size_max <= 0:
            return
        if size_max < size_min:
            raise ValueError("size range must be ordered as (min, max)")
    else:
        size_min = size_max = float(size)
        if size_min <= 0:
            return

    # Build AOIs in projected CRS so base_size is interpreted in meters.
    working_area = area.to_crs("EPSG:28992") if area.crs.upper() in {"EPSG:4326", "CRS84", "OGC:CRS84"} else area

    minx, miny, maxx, maxy = working_area.bounds
    max_attempts = 1000
    while True:
        for attempts in range(1, max_attempts + 1):
            x = float(rng.uniform(minx, maxx))
            y = float(rng.uniform(miny, maxy))

            if not working_area.contains(ShapelyPoint(x, y)):
                continue

            curr_size = float(rng.uniform(size_min, size_max))
            half = curr_size / 2
            angle = float(rng.uniform(0, 360))
            square = Polygon([
                (x - half, y - half),
                (x + half, y - half),
                (x + half, y + half),
                (x - half, y + half),
            ])
            aoi = AOIPolygon(
                affinity.rotate(square, angle, origin="center"),
                crs=working_area.crs,
            )
            if working_area.crs != area.crs:
                aoi = aoi.to_crs(area.crs)

            yield aoi
            break
        else:
            raise RuntimeError(f"Could not place an AOI center inside the area after {max_attempts} attempts. Try reducing size or changing the area.")


def center_target_point_hag(aoi: AOIPolygon, input_path: Path, hag_m: float = 1.7) -> Point:
    """Return an AOI-center target at a fixed height above ground (HAG) in EPSG:28992."""
    from calculate import sample_ground

    aoi_rd = aoi.to_crs("EPSG:28992") if aoi.crs != "EPSG:28992" else aoi
    center = aoi_rd.polygon.centroid
    if not aoi_rd.polygon.covers(center):
        center = aoi_rd.polygon.representative_point()

    ground_z = float(sample_ground(input_path=input_path, points=[Point(float(center.x), float(center.y), 0.0)])[0])
    return Point(float(center.x), float(center.y), ground_z + hag_m)


if __name__ == "__main__":

    @timed("Example function")
    def example():
        time.sleep(1.5)

    example()
