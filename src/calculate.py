import json
import shutil
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pdal
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import contains
from shapely import points as shapely_points
from shapely.geometry import Point as ShapelyPoint
from tqdm import tqdm

from models import AOIPolygon, Cylinder, Point, ProjectConfig, ProjectPaths, RunConfig, RunPaths, Segment
from utils import get_logger, load_profile, timed, write_metadata
from visualize import save_viewshed_as_tif, save_viewshed_as_voxel_grid, write_to_copc

logger = get_logger(name="Calculate")

CLASS_TERRAIN = 2
CLASS_BUILDING = 6
CLASS_VEGETATION = 5

# Figuring out what to set these thresholds to is a key part of the research but we can start with this for now.
TERRAIN_DENSITY_THRESHOLD = 14
BUILDING_DENSITY_THRESHOLD = 5
VEGETATION_DENSITY_THRESHOLD = 3
BEER_LAMBERT_COEFFICIENT = 0.05
DEFAULT_CHUNK_SIZE = 3.0

RadiusMode = Literal["fixed", "widening_linear"]

_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def _validate_points_in_aoi(points_xy: list[tuple[float, float]], aoi: AOIPolygon, labels: list[str]) -> None:
    """Raise when any selected point lies outside the AOI polygon."""

    aoi_rd = aoi.to_crs("EPSG:28992") if aoi.crs != "EPSG:28992" else aoi
    for (x, y), label in zip(points_xy, labels):
        if not aoi_rd.covers(ShapelyPoint(x, y)):
            raise ValueError(f"{label} is outside the AOI. Please select a point inside the AOI.")


def sample_ground(input_path: Path, points: list[Point], k: int = 1, ground_tree: cKDTree | None = None, ground_z: np.ndarray | None = None) -> np.ndarray:
    """Interpolate ground Z from nearby Class-2 points using KD-tree (loads on demand if not provided)."""

    if ground_tree is None or ground_z is None:
        ground_tree, ground_z = load_ground_points(input_path)

    query_xy = np.array([(p.x, p.y) for p in points], dtype=np.float64)

    _, indices = ground_tree.query(query_xy, k=k, workers=-1)

    return ground_z[indices]


def load_ground_points(input_path: Path) -> tuple[cKDTree, np.ndarray]:
    """Load all Class-2 ground points from COPC and build KD-tree. Cache results."""

    pipeline = pdal.Pipeline(
        json.dumps({
            "pipeline": [
                {"type": "readers.copc", "filename": str(input_path)},
                {"type": "filters.expression", "expression": f"Classification == {CLASS_TERRAIN}"},
            ]
        })
    )
    pipeline.execute()

    ground = np.concatenate(pipeline.arrays)

    ground_xy = np.column_stack((ground["X"], ground["Y"]))
    ground_z = ground["Z"].astype(np.float64)

    logger.info(f"Loaded {len(ground)} ground points from {input_path}")
    return cKDTree(ground_xy), ground_z


def generate_grid(Area: AOIPolygon, resolution: int, z_height: float = 0.0, two_d: bool = False, hag_base: Path | None = None) -> list[Point]:
    """Generate a grid of points within the given area polygon, with the specified resolution and optional Z height."""
    min_x, min_y, max_x, max_y = Area.bounds

    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    if two_d or z_height <= 0:
        z_coords = np.array([z_height], dtype=np.float64)
    else:
        z_coords = np.arange(0, z_height, resolution)

    # Create full 3D meshgrid in one go
    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords, indexing="xy")
    all_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1)

    # Vectorized polygon containment check (Shapely >= 2.0)
    geom_points = shapely_points(all_points[:, :2])
    mask = contains(Area.polygon, geom_points)

    filtered = all_points[mask]

    # translate to HAG
    if hag_base:
        ground_levels = sample_ground(input_path=hag_base, points=[Point(x, y, 0) for x, y in filtered[:, :2]])
        filtered[:, 2] += ground_levels

    return [Point(x, y, z) for x, y, z in filtered]


def sample_polygon_boundary(polygon: AOIPolygon, sample_distance: float, z_height: float = 0.0) -> list[Point]:
    """Sample points at regular intervals along the exterior boundary of a polygon."""
    boundary = polygon.exterior
    total_length = boundary.length
    num_samples = max(2, int(np.ceil(total_length / sample_distance)))
    distances = np.linspace(0, total_length, num_samples, endpoint=False)
    return [Point(boundary.interpolate(d).x, boundary.interpolate(d).y, z_height) for d in distances]


def export_grid_to_copc(grid_points: list[Point], output_path: Path, project_cfg: ProjectConfig):
    """
    Exports the generated 3D grid to a COPC (.copc.laz) file for CloudCompare.
    """

    dtype = [
        ("X", "f8"),
        ("Y", "f8"),
        ("Z", "f8"),
    ]

    # Create the structured array
    # If you miss the 'dtype=' argument here, point_data["X"] will fail.
    point_data = np.empty(len(grid_points), dtype=dtype)

    point_data["X"] = np.array([pt.x for pt in grid_points])
    point_data["Y"] = np.array([pt.y for pt in grid_points])
    point_data["Z"] = np.array([pt.z for pt in grid_points])

    write_to_copc(point_data, output_path, project_cfg)


def get_distance_mask(point_array: np.ndarray[Point], cylinder: Cylinder) -> tuple[np.ndarray[bool], np.ndarray[float]]:
    segment = cylinder.segment

    p1_array = segment.point1.array_coords

    # Vector from p1 to all points (w)
    w = point_array - p1_array

    # Projection parameter t = (w·v) / |v|^2
    dots = w @ segment.vector  # w·v
    denom = segment.length_squared  # |v|^2
    t = np.clip(dots / denom, 0.0, 1.0)  # Clip t to the range [0, 1] to stay within the segment

    # Calculate the squared distance to the closest point on the segment
    # The formula: distances_squared = |w|^2 + t^2|v|^2 - 2t(w·v)
    w_mag_sq = np.einsum("ij,ij->i", w, w)
    distances_squared = w_mag_sq + (t**2 * denom) - (2 * t * dots)
    radius = cylinder.radius_at_t(t)

    # Return a boolean mask of points within the specified radius, and the projection parameters
    return distances_squared <= radius**2, t


def get_kdtree_candidate_indices(KDtree: cKDTree, cylinder: Cylinder) -> np.ndarray[int]:
    """
    Generate candidate point indices from a KD-tree within a radius of the line segment.
    """
    segment = cylinder.segment
    radius = cylinder.radius
    step = cylinder.step_length
    num_samples = max(2, int(np.ceil(segment.length / step)) + 1)
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    samples = segment.point1.array_coords + t[:, None] * segment.vector

    # calculate query radius knowing R_sphere = sqrt(r_cylinder² + (step²/4))
    query_radius = np.sqrt(radius**2 + (step**2 / 4))
    candidate_lists = KDtree.query_ball_point(samples, r=query_radius, workers=1)  # workers=-1 causes too much overhead for small queries.

    if len(candidate_lists) == 0:
        return np.array([], dtype=np.int64)

    return np.unique(np.concatenate([np.asarray(candidate, dtype=np.int64) for candidate in candidate_lists]))


def get_PDAL_bounds_for_runs(point_pairs: list[Segment], radius: float) -> str:
    """Calculate the bounding box that contains all point pairs, expanded by the radius."""
    all_points = np.array(
        [pair.point1.array_coords for pair in point_pairs] + [pair.point2.array_coords for pair in point_pairs],
        dtype=np.float64,
    )
    minx, miny, minz = np.min(all_points, axis=0) - radius
    maxx, maxy, maxz = np.max(all_points, axis=0) + radius
    # Return string in the format expected by PDAL's filters.crop
    return f"([{minx},{maxx}],[{miny},{maxy}],[{minz},{maxz}])"


@timed("Loading points for runs")
def load_points_for_runs(point_pairs: list[Segment], radius: float, input_path: Path) -> tuple[np.ndarray, np.ndarray, cKDTree]:
    """
    Load points from the input file that fall within a bounding box defined by the point pairs and radius.
    """
    if radius <= 0:
        logger.error("Radius must be greater than zero.")
        raise ValueError("Radius must be greater than zero.")

    bounds = get_PDAL_bounds_for_runs(point_pairs, radius)

    read_pipeline = {
        "pipeline": [
            {
                "type": "readers.copc",
                "filename": str(input_path),
                "requests": 16,
            },
            {
                "type": "filters.crop",
                "bounds": bounds,
            },
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(read_pipeline))
    count = pipeline.execute()
    if count == 0:
        logger.warning("No points found in the bounding box. Check your input data and coordinates.")
        return None, None, None

    arrays = pipeline.arrays
    if not arrays:
        logger.error("No point data returned from PDAL pipeline.")
        raise ValueError("No point data returned from PDAL pipeline.")
    logger.debug(f"Points in bounding box: {arrays[0].size}")

    array_points = np.concatenate(arrays)
    array_coords = np.column_stack((array_points["X"], array_points["Y"], array_points["Z"]))
    KDtree = cKDTree(array_coords)
    return array_points, array_coords, KDtree


# @timed("Intervisibility calculation")
def calculate_intervisibility(
    cylinder: Cylinder,
    array_points: np.ndarray,
    array_coords: np.ndarray,
    KDtree: cKDTree,
    chunk_size: float = DEFAULT_CHUNK_SIZE,
    distance_mask_function: Callable = get_distance_mask,
) -> tuple[float, np.ndarray]:
    """
    Walk the line of sight from source to target in steps and compute a
    visibility value between 0 and 1.

    For each step:
      - If terrain density >= threshold  -> return 0.0 (fully blocked).
      - If building density >= threshold -> return 0.0 (fully blocked).
      - If vegetation density >= threshold -> decrease visibility via:
        visibility = exp(-k * veg_density * step_length)

    Density is defined as  count / (pi * r^2).
    """

    segment = cylinder.segment

    # Define step bins along the segment
    step_length = cylinder.step_length
    num_steps = max(1, int(np.ceil(segment.length / step_length)))

    # Calculate how many steps fit into one chunk
    steps_per_chunk = max(1, int(np.round(chunk_size / step_length)))

    # Pre-compute midpoint positions for each step along the ray
    t_mids = (np.arange(num_steps) + 0.5) / num_steps
    all_step_positions = segment.point1.array_coords + t_mids[:, None] * segment.vector

    # Pre-compute cross-sectional areas for each step based
    step_radii = cylinder.radius_at_t(t_mids)
    step_areas = np.pi * step_radii**2

    # Walk the LoS
    visibility = 1.0
    step_vis = np.zeros(num_steps, dtype=np.float64)

    # Iterate through the LoS in chunks
    for start_step in range(0, num_steps, steps_per_chunk):
        end_step = min(start_step + steps_per_chunk, num_steps)

        # Generate sample points
        t_chunk = np.linspace(start_step / num_steps, end_step / num_steps, (end_step - start_step) + 1)
        chunk_samples = segment.point1.array_coords + t_chunk[:, None] * segment.vector
        chunk_max_radius = np.max(cylinder.radius_at_t(t_chunk))
        query_radius = np.sqrt(chunk_max_radius**2 + (step_length**2 / 4))

        # Batch query the KDTree for the chunk
        candidate_lists = KDtree.query_ball_point(chunk_samples, r=query_radius)

        # Filter out empty lists and flatten unique indices
        valid_candidates = [c for c in candidate_lists if c]
        if not valid_candidates:
            step_vis[start_step:end_step] = visibility
            continue

        flat_indices = np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in valid_candidates]))

        # 3. Filter points to the cylinder and get their T-values
        candidate_coords = array_coords[flat_indices]
        distance_mask, t_all = distance_mask_function(candidate_coords, cylinder)
        filtered_indices = flat_indices[distance_mask]

        if filtered_indices.size == 0:
            step_vis[start_step:end_step] = visibility
            continue

        filtered_points = array_points[filtered_indices]
        t_values = t_all[distance_mask]

        # Pre-fetch classifications
        classifications = filtered_points["Classification"]

        # Masks per class (computed once)
        terrain_mask = classifications == CLASS_TERRAIN
        building_mask = classifications == CLASS_BUILDING
        vegetation_mask = classifications == CLASS_VEGETATION

        # Step through the bins inside this specific chunk
        bin_edges = np.linspace(start_step / num_steps, end_step / num_steps, (end_step - start_step) + 1)

        for j in range(len(bin_edges) - 1):
            t_lo, t_hi = bin_edges[j], bin_edges[j + 1]
            in_bin = (t_values >= t_lo) & (t_values < t_hi)

            # Include the endpoint in the last bin
            current_global_step = start_step + j
            cross_section_area = step_areas[current_global_step]
            if current_global_step == num_steps - 1:
                in_bin |= t_values == t_hi

            # Check threshold for terrain
            terrain_count = int(np.count_nonzero(in_bin & terrain_mask))
            terrain_density = terrain_count / cross_section_area

            if terrain_density >= TERRAIN_DENSITY_THRESHOLD:
                los_data = np.column_stack([all_step_positions, step_vis])
                return 0.0, los_data

            # Check threshold for buildings
            building_count = int(np.count_nonzero(in_bin & building_mask))
            building_density = building_count / cross_section_area

            if building_density >= BUILDING_DENSITY_THRESHOLD:
                los_data = np.column_stack([all_step_positions, step_vis])
                return 0.0, los_data

            # Decrease visibility for vegetation
            vegetation_count = int(np.count_nonzero(in_bin & vegetation_mask))
            vegetation_density = vegetation_count / cross_section_area

            if vegetation_density >= VEGETATION_DENSITY_THRESHOLD:
                actual_step = step_length if current_global_step < num_steps - 1 else segment.length - current_global_step * step_length
                attenuation = np.exp(-BEER_LAMBERT_COEFFICIENT * vegetation_density * actual_step)
                visibility *= attenuation

            step_vis[current_global_step] = visibility

    visibility = np.clip(visibility, 0.0, 1.0)
    # logger.info(f"Final visibility: {visibility:.4f}")
    los_data = np.column_stack([all_step_positions, step_vis])
    return visibility, los_data


def calculate_point_to_point(
    point_pair: Segment,
    radius: float,
    *,
    radius_mode: RadiusMode = "fixed",
    min_radius: float | None = None,
    max_radius: float | None = None,
    step_length: float | None = None,
) -> float:
    if radius_mode == "fixed":
        min_radius = radius
        max_radius = radius

    array_points, array_coords, KDtree = load_points_for_runs([point_pair], max_radius)
    if array_points is None:
        logger.warning("No points loaded for the requested point pair.")
        return 1.0  # Assume full visibility if no points are loaded
    cylinder_of_sight = Cylinder(
        segment=Segment(point_pair.point1, point_pair.point2),
        min_radius=min_radius,
        max_radius=max_radius,
        step_length=step_length,
        radius_mode=radius_mode,
    )
    visibility, _ = calculate_intervisibility(
        cylinder_of_sight,
        array_points,
        array_coords,
        KDtree,
    )
    return visibility


def calculate_viewshed_for_grid(
    target: Point,
    grid_points: list[Point],
    cylinder_radius: float,
    project_cfg: ProjectConfig,
    radius_mode: RadiusMode = "fixed",
    min_radius: float | None = None,
    max_radius: float | None = None,
    step_length: float | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None,
    intervisibility_func: Callable = calculate_intervisibility,
    save_to_disk: bool = True,
    chunk_size: float = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every point in the grid, compute visibility from target to that point
    and write the result as a new 'Visibility' dimension in a COPC file.
    """
    # Load points from AHN with a dummy pair for now
    dummy_pair = Segment(target, target)

    grid_coords = np.array([pt.array_coords for pt in grid_points])
    max_dist = np.max(np.linalg.norm(grid_coords - target.array_coords, axis=1))

    if radius_mode == "fixed":
        min_radius = cylinder_radius
        max_radius = cylinder_radius

    array_points, array_coords, KDtree = load_points_for_runs([dummy_pair], max_dist + max_radius, input_path=input_path)

    logger.info(f"Computing visibility for {len(grid_points)} grid points.")

    visibility_values = np.zeros(len(grid_points), dtype=np.float64)
    all_los_chunks: list[np.ndarray] = []

    for i, dest in enumerate(tqdm(grid_points, desc="Grid Viewshed", unit="LoS")):
        segment = Segment(target, dest)
        cylinder = Cylinder(
            segment=segment,
            min_radius=min_radius,
            max_radius=max_radius,
            step_length=step_length,
            radius_mode=radius_mode,
        )

        # Calculate visibility and collect per-step LoS data
        visibility_values[i], los_data = intervisibility_func(cylinder, array_points, array_coords, KDtree, chunk_size=chunk_size)
        all_los_chunks.append(los_data)

    # Build output array
    dtype = [("X", "<f8"), ("Y", "<f8"), ("Z", "<f8"), ("Visibility", "<f8")]
    out_grid = np.empty(len(grid_points), dtype=dtype)

    out_grid["X"] = grid_coords[:, 0]
    out_grid["Y"] = grid_coords[:, 1]
    out_grid["Z"] = grid_coords[:, 2]
    out_grid["Visibility"] = visibility_values

    if all_los_chunks:
        all_los_arr = np.vstack(all_los_chunks)
        out_los = np.empty(len(all_los_arr), dtype=dtype)
        out_los["X"] = all_los_arr[:, 0]
        out_los["Y"] = all_los_arr[:, 1]
        out_los["Z"] = all_los_arr[:, 2]
        out_los["Visibility"] = all_los_arr[:, 3]
        out_array = np.concatenate([out_grid, out_los])
    else:
        out_array = out_grid

    if save_to_disk:
        write_to_copc(out_array, output_path, project_cfg)

    return visibility_values, out_array


def hag_to_ortho(points: list["Point"], input_path: Path) -> list["Point"]:
    """
    Converts a list of points from Height Above Ground (HAG) to Orthometric Height (ortho).
    """
    ground_levels = sample_ground(input_path=input_path, points=points)
    return [Point(point.x, point.y, point.z + ground) for point, ground in zip(points, ground_levels)]


def ortho_to_hag(points: list["Point"], input_path: Path) -> list["Point"]:
    """
    Converts a list of points from Orthometric Height (ortho) to Height Above Ground (HAG).
    """
    ground_levels = sample_ground(input_path=input_path, points=points)
    return [Point(point.x, point.y, point.z - ground) for point, ground in zip(points, ground_levels)]


def calculate_viewshed_2d(
    target: Point,
    aoi: AOIPolygon,
    project_cfg: ProjectConfig,
    resolution: float = 1.0,
    input_path: Path | None = None,
    output_path: Path | None = None,
    z_offset: float = 0.0,
    radius: float = 0.15,
    *,
    radius_mode: RadiusMode = "fixed",
    min_radius: float | None = None,
    max_radius: float | None = None,
    step_length: float | None = None,
) -> tuple[list[Point], np.ndarray, np.ndarray]:
    # Sample points along the AOI boundary at the given resolution
    boundary_points = sample_polygon_boundary(aoi, sample_distance=resolution, z_height=z_offset)
    grid_points_ortho = hag_to_ortho(boundary_points, input_path=input_path)
    # export_grid_to_copc(grid_points_ortho, output_path=Path("data/grid_points_2d_edges.copc.laz"))

    # For each point, calculate the visibility from the target to that point
    visibility_values, visibility_points = calculate_viewshed_for_grid(
        target=target,
        grid_points=grid_points_ortho,
        project_cfg=project_cfg,
        cylinder_radius=radius,
        radius_mode=radius_mode,
        min_radius=min_radius,
        max_radius=max_radius,
        step_length=step_length,
        input_path=input_path,
        output_path=output_path,
        intervisibility_func=calculate_intervisibility,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )

    return grid_points_ortho, visibility_values, visibility_points


def only_save_viewable_volume(input_path: Path, output_path: Path, project_cfg: ProjectConfig):
    pipeline = {
        "pipeline": [
            {"type": "readers.copc", "filename": str(input_path)},
            {"type": "filters.range", "limits": "Visibility(0.0:)"},  # Only keep points where visibility >= 0.0
        ]
    }
    pl = pdal.Pipeline(json.dumps(pipeline))
    pl.execute()
    arr = pl.arrays[0]
    if arr.size == 0:
        logger.warning("No points with Visibility >= 0.0 found in the input file.")
        return
    write_to_copc(arr, output_path, project_cfg)
    logger.info(f"Saved viewable volume to {output_path}")


@timed("Run 2D viewshed")
def calculate_2d_viewshed(project_cfg: ProjectConfig, project_paths: ProjectPaths, run_cfg: RunConfig, profile: str = "testing") -> RunPaths:
    profile_cfg = load_profile(profile)
    run_paths = RunPaths(project_paths, run_cfg.name)

    run_logger = get_logger(name="Main", logfile_path=run_paths.run_log, level=profile_cfg.logging_level)

    source_aoi_crs = AOIPolygon.get_from_file(project_paths.aoi).crs
    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs)

    run_logger.info(f"Running viewshed '{run_cfg.name}' in project '{project_paths.name}'")
    run_logger.info(f"LoS settings: mode={run_cfg.los_mode} radius={run_cfg.los_radius} min_radius={run_cfg.los_start_radius} max_radius={run_cfg.los_end_radius} step_length={run_cfg.los_step_length}")

    start_time = time.perf_counter()

    target = Point.get_from_file(run_cfg.target_source)

    viewshed_2d_path = run_paths.output_viewshed_copc_2d
    _, _, visibility_points = calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        project_cfg=project_cfg,
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

    write_metadata(run_cfg, project_paths, run_paths, project_cfg.profile, source_aoi_crs, start_time)

    run_logger.info("Run completed")
    return run_paths


@timed("Run viewshed")
def calculate_3d_viewshed(
    project_cfg: ProjectConfig,
    project_paths: ProjectPaths,
    run_cfg: RunConfig,
    profile: str = "testing",
    save_to_disk: bool = True,
) -> tuple[RunPaths, np.ndarray, np.ndarray]:
    profile_cfg = load_profile(profile)
    run_paths = RunPaths(project_paths, run_cfg.name)

    log_path = run_paths.run_log if save_to_disk else None
    run_logger = get_logger(name="Main", logfile_path=log_path, level=profile_cfg.logging_level)

    source_aoi_crs = AOIPolygon.get_from_file(project_paths.aoi).crs
    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs)

    run_logger.info(f"Running viewshed '{run_cfg.name}' in project '{project_paths.name}'")
    run_logger.info(f"LoS settings: mode={run_cfg.los_mode} radius={run_cfg.los_radius} min_radius={run_cfg.los_start_radius} max_radius={run_cfg.los_end_radius} step_length={run_cfg.los_step_length}")

    start_time = time.perf_counter()

    target = Point.get(
        hag_sample_input_path=project_paths.input_copc,
        input_path=run_cfg.target_source,
        title="Select target point for 3D viewshed",
        overwrite=run_cfg.overwrite,
        aoi=AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs),
    )

    top_points = generate_grid(Area=aoi, resolution=run_cfg.resolution, z_height=run_cfg.z_height, two_d=True)
    for pt in top_points:
        pt.z = run_cfg.z_height

    boundary_points = sample_polygon_boundary(aoi, sample_distance=run_cfg.resolution, z_height=0.0)
    wall_zs = np.arange(0, run_cfg.z_height + run_cfg.resolution, run_cfg.resolution)
    wall_points = [Point(pt.x, pt.y, z) for pt in boundary_points for z in wall_zs]
    grid_points = top_points + wall_points

    grid_points_ortho = hag_to_ortho(grid_points, input_path=project_paths.input_copc)

    if save_to_disk:
        export_grid_to_copc(grid_points_ortho, output_path=run_paths.grid_shell_copc, project_cfg=project_cfg)

    vis_vals, out_array = calculate_viewshed_for_grid(
        target=target,
        grid_points=grid_points_ortho,
        cylinder_radius=run_cfg.los_radius,
        project_cfg=project_cfg,
        radius_mode=run_cfg.los_mode,
        min_radius=run_cfg.los_start_radius,
        max_radius=run_cfg.los_end_radius,
        step_length=run_cfg.los_step_length,
        input_path=project_paths.facades_copc,
        output_path=run_paths.output_viewshed_copc_3d,
        intervisibility_func=calculate_intervisibility,
        chunk_size=5,
        save_to_disk=save_to_disk,
    )

    if save_to_disk:
        only_save_viewable_volume(run_paths.output_viewshed_copc_3d, run_paths.viewable_volume_copc, project_cfg=project_cfg)
        write_metadata(run_cfg, project_paths, run_paths, project_cfg.profile, source_aoi_crs, start_time)

    run_logger.info("Run completed")
    return run_paths, vis_vals, out_array


def setup_targets(run_cfg: RunConfig, project_paths: ProjectPaths, aoi_rd: AOIPolygon, number_of_targets: int) -> list[Path]:
    target_sources = []
    merge_dir = project_paths.runs_folder / run_cfg.name / "merge_candidates"
    merge_dir.mkdir(exist_ok=True)
    for i in range(number_of_targets):
        target_path = merge_dir / f"target_run_{i + 1}.copc.laz"
        Point.get(
            hag_sample_input_path=project_paths.input_copc,
            input_path=target_path,
            title=f"Select target point for run {i + 1}",
            overwrite=False,
            aoi=aoi_rd,
        )
        target_sources.append(target_path)

    return target_sources


def save_cumulative_viewshed(
    height: int,
    width: int,
    min_x: float,
    max_y: float,
    min_z: float,
    cumulative_voxels: dict[int, float],
    cumulative_run_paths: RunPaths,
    project_paths: ProjectPaths,
    run_cfg: RunConfig,
    project_cfg: ProjectConfig,
):
    # Reconstruct 3D Point Cloud from Master Grid in batches to keep memory bounded
    logger.info("Reconstructing volumetric cumulative viewshed...")
    save_viewshed_as_voxel_grid(
        run_paths=cumulative_run_paths,
        run_cfg=run_cfg,
        project_paths=project_paths,
        project_cfg=project_cfg,
        file_type="copc",
    )
    # Copy to raw viewshed path for consistency with downstream analysis
    shutil.copy2(cumulative_run_paths.output_viewshed_voxel_grid_3d, cumulative_run_paths.output_viewshed_copc_3d)

    logger.info(f"Cumulative volumetric viewshed saved to run folder: {cumulative_run_paths.folder.name}")


def calculate_cumulative_viewshed(
    number_of_targets: int,
    project_cfg: ProjectConfig,
    project_paths: ProjectPaths,
    run_cfg: RunConfig,
    save_to_disk: bool = False,
) -> RunPaths:

    # Setup AOI
    aoi_rd = AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs)

    # Establish global grid
    min_x, min_y, max_x, max_y = aoi_rd.bounds
    min_z = 0.0  # Assuming orthometric ground floor (ortho)
    max_z = run_cfg.z_height

    width = int(np.floor((max_x - min_x) / run_cfg.resolution)) + 1
    height = int(np.floor((max_y - min_y) / run_cfg.resolution)) + 1
    depth = int(np.floor((max_z - min_z) / run_cfg.resolution)) + 1

    # Dictionary to store the accumulated visibility (Sparse 3D matrix)
    cumulative_voxels: dict[int, float] = {}

    # Cache the original run name to restore later (since we'll mutate it in the loop)
    original_run_name = run_cfg.name
    cumulative_run_paths = RunPaths(project_paths, original_run_name)
    cumulative_run_paths.folder.mkdir(parents=True, exist_ok=True)

    target_sources = setup_targets(run_cfg, project_paths, aoi_rd, number_of_targets)
    for idx, target_source in enumerate(target_sources):
        # Mutate config for the worker
        run_cfg.name = f"{project_cfg.name}_run_{idx + 1}"
        # make sure directory for log file exists
        if save_to_disk:
            (project_paths.runs_folder / run_cfg.name).mkdir(parents=True, exist_ok=True)
        run_cfg.target_source = target_source

        logger.info(f"Processing run {idx + 1}/{number_of_targets}...")

        # Compute individual viewshed (Worker returns raw point array with LoS rays)
        run_paths, _, out_array = calculate_3d_viewshed(project_cfg=project_cfg, project_paths=project_paths, run_cfg=run_cfg, profile=project_cfg.profile, save_to_disk=save_to_disk)
        if save_to_disk:
            save_viewshed_as_voxel_grid(run_paths, run_cfg=run_cfg, project_paths=project_paths, project_cfg=project_cfg)

        # Voxelize this observer's rays into the Global Grid
        cols = np.floor((out_array["X"] - min_x) / run_cfg.resolution).astype(np.int64)
        rows = np.floor((max_y - out_array["Y"]) / run_cfg.resolution).astype(np.int64)
        depths = np.floor((out_array["Z"] - min_z) / run_cfg.resolution).astype(np.int64)
        vis = out_array["Visibility"]

        # Filter out points that fall outside our bounded grid
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width) & (depths >= 0) & (depths < depth)

        # Flatten 3D indices to 1D keys
        flat_indices = (depths[valid] * height * width) + (rows[valid] * width) + cols[valid]
        valid_vis = vis[valid]

        # Fast NumPy trick to find the MAX visibility per voxel for THIS observer
        order = np.argsort(flat_indices, kind="mergesort")
        flat_sorted = flat_indices[order]
        vis_sorted = valid_vis[order]
        unique_flat, first_idx = np.unique(flat_sorted, return_index=True)
        chunk_max = np.maximum.reduceat(vis_sorted, first_idx)

        # Accumulate the max visibility
        for flat, value in zip(unique_flat.tolist(), chunk_max.tolist()):
            existing = cumulative_voxels.get(flat, -1.0)
            cumulative_voxels[flat] = max(existing, float(value))

    # Restore original run name in config
    run_cfg.name = original_run_name
    save_cumulative_viewshed(
        height=height,
        width=width,
        min_x=min_x,
        max_y=max_y,
        min_z=min_z,
        cumulative_voxels=cumulative_voxels,
        cumulative_run_paths=cumulative_run_paths,
        run_cfg=run_cfg,
        project_paths=project_paths,
        project_cfg=project_cfg,
    )

    return cumulative_run_paths


if __name__ == "__main__":
    pass
