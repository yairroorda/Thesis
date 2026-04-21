import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import numpy as np
import pdal
from shapely import affinity
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

import calculate as calc
from main import calculate_3d_viewshed, prepare_project
from models import AOIPolygon, Point, ProjectConfig, RunConfig
from visualize import save_viewshed_as_voxel_grid


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


def random_target_point(aoi: AOIPolygon, z_range: tuple[float, float], seed: int = 42) -> Point:
    """Return a random point within the AOI in EPSG:28992 coordinates."""
    rng = np.random.default_rng(seed)
    aoi_rd = aoi.to_crs("EPSG:28992") if aoi.crs != "EPSG:28992" else aoi
    minx, miny, maxx, maxy = aoi_rd.bounds
    minz, maxz = z_range
    for _ in range(1000):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        z = float(rng.uniform(minz, maxz))
        point = ShapelyPoint(x, y, z)
        if aoi_rd.contains(point):
            return Point(x, y, z)
    raise RuntimeError("Could not find a target point within the AOI after 1000 attempts. Try changing the AOI or seed.")


def center_target_point_hag(aoi: AOIPolygon, hag_m: float = 1.7) -> Point:
    """Return an AOI-center target at a fixed height above ground (HAG) in EPSG:28992."""
    aoi_rd = aoi.to_crs("EPSG:28992") if aoi.crs != "EPSG:28992" else aoi
    center = aoi_rd.polygon.centroid
    if not aoi_rd.polygon.covers(center):
        center = aoi_rd.polygon.representative_point()

    dtm = calc.download_dtm_raster(aoi_rd)
    ground_z = float(calc.sample_dtm(dtm, [Point(float(center.x), float(center.y), 0.0)])[0])
    return Point(float(center.x), float(center.y), ground_z + hag_m)


def _load_copc(path: Path) -> np.ndarray:
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(path)}]}))
    pipeline.execute()
    return np.concatenate(pipeline.arrays)


def _voxel_metrics(with_path: Path, without_path: Path, tolerance: float = 0.0) -> dict:
    with_points = _load_copc(with_path)
    without_points = _load_copc(without_path)

    with_order = np.lexsort((with_points["Z"], with_points["Y"], with_points["X"]))
    without_order = np.lexsort((without_points["Z"], without_points["Y"], without_points["X"]))
    with_points = with_points[with_order]
    without_points = without_points[without_order]

    delta = np.abs(with_points["Visibility"] - without_points["Visibility"])
    affected = delta > tolerance
    visible = with_points["Visibility"] > 0
    visible_affected = affected & visible

    return {
        "voxel_count": int(delta.size),
        "affected_voxel_percentage": float(100.0 * affected.mean()) if delta.size else 0.0,
        "visible_voxel_percentage_affected": float(100.0 * visible_affected.sum() / visible.sum()) if visible.any() else 0.0,
        "mean_absolute_difference": float(delta.mean()) if delta.size else 0.0,
        "std_absolute_difference": float(delta.std()) if delta.size else 0.0,
        "mean_effected_difference": float(delta[affected].mean()) if affected.any() else 0.0,
        "std_effected_difference": float(delta[affected].std()) if affected.any() else 0.0,
    }


def _vegetation_class_percentage(path: Path) -> float:
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(path)}]}))
    pipeline.execute()

    vegetation_count = 0
    point_count = 0
    for chunk in pipeline.arrays:
        if "Classification" not in chunk.dtype.names:
            continue
        classifications = chunk["Classification"]
        point_count += classifications.size
        vegetation_count += int(np.count_nonzero(classifications == calc.CLASS_VEGETATION))

    return float(100.0 * vegetation_count / point_count) if point_count else 0.0


def summarize_vegetation_influence(output_dir: Path) -> dict:
    run_comparison_files = sorted(output_dir.glob("run_*/comparison.json"))
    run_metrics = []
    for comparison_file in run_comparison_files:
        data = json.loads(comparison_file.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        if isinstance(metrics, dict):
            run_metrics.append(metrics)

    summary = {}
    metric_keys = set().union(*(m.keys() for m in run_metrics)) if run_metrics else set()
    for key in sorted(metric_keys):
        values = [m[key] for m in run_metrics if key in m and isinstance(m[key], (int, float, np.integer, np.floating))]
        if not values:
            continue
        values_arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values_arr)),
            "std_across_aois": float(np.std(values_arr)),
        }

    summary_path = output_dir / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_vegetation_influence(
    config: ProjectConfig,
    num_aois: int,
    output_dir: Path,
    override: bool,
    aoi_generator: Iterator[AOIPolygon] | None = None,
) -> Path:
    if output_dir.exists() and not override:
        raise FileExistsError(f"Output directory {output_dir} already exists. Use override=True to overwrite.")

    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_features: list[dict] = []

    for index in range(1, num_aois + 1):
        aoi = next(aoi_generator)
        run_dir = output_dir / f"run_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        aoi.save_to_file(run_dir / "aoi.geojson")
        aoi_features.append({"type": "Feature", "properties": {"run_index": index}, "geometry": aoi.polygon.__geo_interface__})

        temp_config = ProjectConfig(
            name=f"{config.name}_veg_eval_{index:02d}_{int(time.time())}",
            dataset=config.dataset,
            classification_method=config.classification_method,
            myria3d_vegetation_prob_threshold_pct=config.myria3d_vegetation_prob_threshold_pct,
            profile=config.profile,
            aoi_source=run_dir / "aoi.geojson",
            overwrite=True,
        )
        project_paths = prepare_project(temp_config)
        input_vegetation_percentage = _vegetation_class_percentage(project_paths.facades_copc)

        target = center_target_point_hag(aoi, hag_m=1.7)
        target.save_to_file(run_dir / "target.copc.laz")

        with_cfg = RunConfig(name=f"run_{index:02d}_with", target_source=run_dir / "target.copc.laz")
        with_run = calculate_3d_viewshed(project_paths, with_cfg, profile=config.profile)
        save_viewshed_as_voxel_grid(with_run, with_cfg, project_paths)
        with_voxel = run_dir / "3d_viewshed_voxel_grid_including_vegetation.copc.laz"
        shutil.copy2(with_run.output_viewshed_voxel_grid_3d, with_voxel)

        original_threshold = calc.VEGETATION_DENSITY_THRESHOLD
        calc.VEGETATION_DENSITY_THRESHOLD = float("inf")
        try:
            without_cfg = RunConfig(name=f"run_{index:02d}_without", target_source=run_dir / "target.copc.laz")
            without_run = calculate_3d_viewshed(project_paths, without_cfg, profile=config.profile)
            save_viewshed_as_voxel_grid(without_run, without_cfg, project_paths)
        finally:
            calc.VEGETATION_DENSITY_THRESHOLD = original_threshold

        without_voxel = run_dir / "3d_viewshed_voxel_grid_excluding_vegetation.copc.laz"
        shutil.copy2(without_run.output_viewshed_voxel_grid_3d, without_voxel)

        run_metrics = _voxel_metrics(with_voxel, without_voxel)
        run_metrics["input_vegetation_percentage"] = input_vegetation_percentage

        comparison = {
            "run_index": index,
            "files": {
                "with_vegetation": with_voxel.name,
                "without_vegetation": without_voxel.name,
            },
            "metrics": run_metrics,
        }
        (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        shutil.rmtree(project_paths.folder, ignore_errors=True)

    (output_dir / "aois.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": aoi_features}, indent=2),
        encoding="utf-8",
    )

    summarize_vegetation_influence(output_dir)
    return output_dir


def demo_generate_benchmark_aois():
    from utils import get_logger

    logger = get_logger(name="AOI Generator Test")

    output_folder = Path("data/benchmark_aois")
    output_folder.mkdir(exist_ok=True)
    area_path = Path("data/benchmark_aois/area.geojson")

    # Example usage
    if not area_path.exists():
        area_polygon = AOIPolygon.get_from_user("Select area for AOI generation")
        area_polygon.save_to_file(output_folder / "area.geojson")
    else:
        area_polygon = AOIPolygon.get_from_file(area_path)

    aoi_generator = generate_benchmark_aois(size=(500, 1000), area=area_polygon, seed=42)
    aois = [next(aoi_generator) for i in range(15)]

    gpd.GeoDataFrame(geometry=[aoi.polygon for aoi in aois], crs=aois[0].crs if aois else area_polygon.crs).to_file(
        output_folder / "aois.geojson",
        driver="GeoJSON",
    )
    logger.info(f"Generated {len(aois)} AOIs and saved to {output_folder / 'aois.geojson'}")

    target = center_target_point_hag(aois[0], hag_m=1.7)
    target.save_to_file(output_folder / "target.copc.laz")
    aois[0].save_to_file(output_folder / "aoi_1.geojson")

    # for ix, aoi in enumerate(aois, start=1):
    #     aoi.save_to_file(output_folder / f"aoi_{ix}.geojson")


if __name__ == "__main__":
    area_path = Path("data/vegetation_comp/area.geojson")
    if not area_path.exists():
        area_polygon = AOIPolygon.get_from_user("Select area for AOI generation")
        area_path.parent.mkdir(parents=True, exist_ok=True)
        area_polygon.save_to_file(area_path)

    name = "evaluate_vegetation_influence_bigger"

    evaluate_vegetation_influence(
        ProjectConfig(dataset=["AHN5"], name=name, aoi_source=area_path),
        num_aois=3,
        output_dir=Path(f"data/{name}"),
        override=True,
        aoi_generator=generate_benchmark_aois(size=100, area=AOIPolygon.get_from_file(area_path), seed=42),
    )

    # summarize_vegetation_influence(Path("data/evaluate_vegetation_influence"))
