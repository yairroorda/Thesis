import json
import time

import numpy as np
import pdal
from scipy.spatial import cKDTree
from shapely.geometry import Point as ShapelyPoint
from tqdm import tqdm

from calculate import calculate_intervisibility, export_grid_to_copc, generate_grid
from models import AOIPolygon, Cylinder, ObserverPath, Point, ProjectConfig, ProjectPaths, RunConfig, RunPaths, Segment
from utils import get_logger, load_profile, write_metadata
from visualize import write_to_copc

logger = get_logger(name="Lookout")


def score_observers(
    run_cfg: RunConfig,
    candidates: list[Point],
    trail_targets: list[Point],
    total_targets: int,
    array_points: np.ndarray,
    array_coords: np.ndarray,
    kdtree: cKDTree,
    all_evaluated_points: list[tuple[Point, float]],
    threshold: float = 0,
) -> tuple[float, Point]:
    best_pct = -1.0
    best_pt = None
    for pt in tqdm(candidates, desc="Scoring observers", leave=False):
        visible_count = 0
        for dest in trail_targets:
            segment = Segment(pt, dest)
            cylinder = Cylinder(segment=segment, min_radius=run_cfg.los_radius, max_radius=run_cfg.los_radius, step_length=run_cfg.los_step_length)
            vis, _ = calculate_intervisibility(cylinder, array_points, array_coords, kdtree)
            if vis > threshold:
                visible_count += 1

        percentage = (visible_count / total_targets) * 100.0
        all_evaluated_points.append((pt, percentage))

        if percentage > best_pct:
            best_pct = percentage
            best_pt = pt
    return best_pct, best_pt


def calculate_optimal_lookout(
    project_cfg: ProjectConfig,
    project_paths: ProjectPaths,
    run_cfg: RunConfig,
    profile: str = "testing",
    coarse_res: float = 10.0,
    fine_res: float = 2.0,
    trail_sample_distance: float = 5.0,
    observer_height: float = 1.8,
    overwrite: bool = False,
    threshold: float = 0,
) -> tuple[RunPaths, Point]:
    """Finds the optimal observer location to view a trail using coarse-then-fine grid search.

    Args:
        project_cfg: Project configuration
        project_paths: Project paths
        run_cfg: Run configuration
        profile: Profile name for logging settings
        coarse_res: Coarse grid resolution in meters
        fine_res: Fine grid resolution in meters
        trail_sample_distance: Distance between sampled points on the trail
        observer_height: Observer height above ground (HAG)

    Returns:
        RunPaths with output_copc containing evaluated grid points and the best observer point
    """
    profile_cfg = load_profile(profile)
    run_paths = RunPaths(project_paths, run_cfg.name)
    run_paths.folder.mkdir(parents=True, exist_ok=True)

    run_logger = get_logger(name="Lookout", logfile_path=run_paths.run_log, level=profile_cfg.logging_level)

    source_aoi_crs = AOIPolygon.get_from_file(project_paths.aoi).crs
    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs)

    start_time = time.perf_counter()

    run_logger.info(f"Running optimal lookout '{run_cfg.name}' in project '{project_paths.name}'")
    run_logger.info(f"Coarse res: {coarse_res}m, Fine res: {fine_res}m, Observer height: {observer_height}m")

    # Get trail path from user
    route = ObserverPath.get(
        input_path=run_paths.folder / "trail_path.geojson",
        title="Select path for lookout analysis",
        overwrite=overwrite,
        aoi=aoi,
    )

    # Sample points along the trail
    trail_targets = route.sample_points(step_size=trail_sample_distance, z_height=1.8)
    export_grid_to_copc(trail_targets, run_paths.target_point_copc, crs=project_cfg.crs)
    total_targets = len(trail_targets)
    run_logger.info(f"Loaded {total_targets} trail target points at {trail_sample_distance}m spacing")

    # Load point cloud and build KDTree
    run_logger.info("Loading point cloud and building KDTree...")
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(project_paths.facades_copc)}]}))
    pipeline.execute()
    array_points = np.concatenate(pipeline.arrays)
    array_coords = np.column_stack((array_points["X"], array_points["Y"], array_points["Z"]))
    kdtree = cKDTree(array_coords)

    all_evaluated_points = []  # Stores tuples of (Point, percentage)

    # Coarse Grid Search
    run_logger.info(f"Coarse search ({coarse_res}m resolution)")
    coarse_pts = generate_grid(aoi, coarse_res, z_height=observer_height, two_d=True)

    best_score, best_pt = score_observers(
        run_cfg=run_cfg,
        candidates=coarse_pts,
        trail_targets=trail_targets,
        total_targets=total_targets,
        array_points=array_points,
        array_coords=array_coords,
        kdtree=kdtree,
        all_evaluated_points=all_evaluated_points,
        threshold=threshold,
    )
    run_logger.info(f"Best coarse candidate: {best_score:.1f}% visibility")

    # Fine Grid Search
    run_logger.info(f"Fine search ({fine_res}m resolution) around best candidate")
    fine_pts = []
    for dx in np.arange(-coarse_res / 2, coarse_res / 2 + fine_res, fine_res):
        for dy in np.arange(-coarse_res / 2, coarse_res / 2 + fine_res, fine_res):
            if (dx != 0 or dy != 0) and aoi.covers(ShapelyPoint(best_pt.x + dx, best_pt.y + dy)):
                fine_pts.append(Point(best_pt.x + dx, best_pt.y + dy, observer_height))

    fine_score, fine_pt = score_observers(
        run_cfg=run_cfg,
        candidates=fine_pts,
        trail_targets=trail_targets,
        total_targets=total_targets,
        array_points=array_points,
        array_coords=array_coords,
        kdtree=kdtree,
        all_evaluated_points=all_evaluated_points,
        threshold=threshold,
    )
    if fine_score > best_score:
        best_score = fine_score
        best_pt = fine_pt
    run_logger.info(f"Best fine candidate: {best_score:.1f}% visibility")

    # Export all evaluated points to COPC
    output_path = run_paths.folder / "optimal_lookout_heatmap.copc.laz"
    run_logger.info(f"Exporting {len(all_evaluated_points)} evaluated grid points to {output_path}")
    dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Visibility", "f8")]
    out_array = np.empty(len(all_evaluated_points), dtype=dtype)
    for i, (pt, pct) in enumerate(all_evaluated_points):
        out_array[i]["X"] = pt.x
        out_array[i]["Y"] = pt.y
        out_array[i]["Z"] = pt.z
        out_array[i]["Visibility"] = pct

    write_to_copc(out_array, output_path, crs=project_cfg.crs)

    write_metadata(run_cfg, project_paths, run_paths, project_cfg.profile, source_aoi_crs, start_time)
    run_logger.info(f"Optimal lookout found at ({best_pt.x:.1f}, {best_pt.y:.1f}, {best_pt.z:.1f}) with {best_score:.1f}% visibility")

    return run_paths, best_pt
