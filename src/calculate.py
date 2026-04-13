import json
import sys
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pdal
from nlmod.read import ahn
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import contains
from shapely import points as shapely_points
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

from gui import make_map
from query_copc import AOIPolygon
from utils import get_logger, timed
from visualize import save_viewshed_as_tif

logger = get_logger(name="Calculate")

DEFAULT_INPUT = Path(r"data/Groningen_plein_AHN4/facades.copc.laz")
DEFAULT_OUTPUT = Path(r"data/output_viewshed_shell.copc.laz")

WRITE_TO_FILE = True  # Control whether to write query outputs to a file.

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


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.array_coords = np.array([x, y, z], dtype=np.float64)
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def get_from_file(cls, path: Path) -> "Point":
        """Read first point from COPC/LAZ and return it as a Point."""
        reader_type = "readers.copc" if ".copc" in path.name.lower() else "readers.las"
        pipeline = pdal.Pipeline(
            json.dumps(
                {
                    "pipeline": [
                        {"type": reader_type, "filename": str(path)},
                    ]
                }
            )
        )
        count = pipeline.execute()
        if count == 0 or not pipeline.arrays:
            raise ValueError(f"No points found in target source file: {path}")
        first = pipeline.arrays[0][0]
        return cls(first["X"], first["Y"], first["Z"])

    @classmethod
    def get_from_user(cls, title: str = "Set point", aoi: AOIPolygon | None = None) -> "Point":
        """Let the user pick one point on the map. Returns (x, y, z) in RD."""
        import tkinter as tk

        root, map_widget, controls = make_map(title, aoi=aoi)

        p_xy = {"v": None}
        marker = {"p": None}

        def on_click(coords):
            lat_c, lon_c = coords[0], coords[1]
            x, y = _TO_RD.transform(lon_c, lat_c)

            if marker["p"] is not None:
                marker["p"].delete()
            marker["p"] = map_widget.set_marker(lat_c, lon_c, text="P1")
            p_xy["v"] = (x, y)

        tk.Label(controls, text="Point P1").pack(anchor="w")

        tk.Label(controls, text="P1 Z").pack(anchor="w", pady=(8, 0))
        pz = tk.Entry(controls)
        pz.insert(0, "8.0")
        pz.pack(fill=tk.X)

        is_hag = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Z is HAG (vs NAP)", variable=is_hag).pack(anchor="w", pady=(5, 0))

        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(10, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()

        if aoi is not None:
            _validate_points_in_aoi([p_xy["v"]], aoi, labels=["Selected point"])

        p = (*p_xy["v"], float(pz.get()))
        root.destroy()

        pt = cls(*p)
        return hag_to_nap([pt])[0] if is_hag.get() else pt

    def save_to_file(self, path: Path) -> None:
        """Save this point to a COPC/LAZ file for later retrieval."""
        dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
        point_data = np.array([(self.x, self.y, self.z)], dtype=dtype)
        write_to_copc(point_data, path)


class Segment:
    def __init__(self, point1: Point, point2: Point):
        self.point1 = point1
        self.point2 = point2
        self.vector = np.array(
            [point2.x - point1.x, point2.y - point1.y, point2.z - point1.z],
            dtype=np.float64,
        )
        self.length_squared = np.dot(self.vector, self.vector)
        self.length = np.sqrt(self.length_squared)

    @classmethod
    def get_from_user(cls, title: str = "Set P1/P2", aoi: AOIPolygon | None = None) -> "Segment":
        """Let the user pick two points on the map. Returns (p1, p2) with p1/p2 as (x, y, z) in RD."""
        import tkinter as tk

        root, map_widget, controls = make_map(title, aoi=aoi)

        mode = tk.StringVar(value="p1")
        p1_xy = {"v": None}
        p2_xy = {"v": None}
        markers = {"p1": None, "p2": None}

        def on_click(coords):
            lat_c, lon_c = coords[0], coords[1]
            x, y = _TO_RD.transform(lon_c, lat_c)
            key = mode.get()

            if markers[key] is not None:
                markers[key].delete()
            markers[key] = map_widget.set_marker(lat_c, lon_c, text=key.upper())
            (p1_xy if key == "p1" else p2_xy)["v"] = (x, y)

            if key == "p1":
                mode.set("p2")

        tk.Label(controls, text="Mode").pack(anchor="w")
        tk.Radiobutton(controls, text="Point P1", variable=mode, value="p1").pack(anchor="w")
        tk.Radiobutton(controls, text="Point P2", variable=mode, value="p2").pack(anchor="w")

        tk.Label(controls, text="P1 Z").pack(anchor="w", pady=(8, 0))
        p1z = tk.Entry(controls)
        p1z.insert(0, "8.0")
        p1z.pack(fill=tk.X)

        tk.Label(controls, text="P2 Z").pack(anchor="w", pady=(8, 0))
        p2z = tk.Entry(controls)
        p2z.insert(0, "10.0")
        p2z.pack(fill=tk.X)

        is_hag = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Z is HAG (vs NAP)", variable=is_hag).pack(anchor="w", pady=(5, 0))

        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(10, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()

        if aoi is not None:
            _validate_points_in_aoi([p1_xy["v"], p2_xy["v"]], aoi, labels=["P1", "P2"])

        p1 = (*p1_xy["v"], float(p1z.get()))
        p2 = (*p2_xy["v"], float(p2z.get()))
        root.destroy()

        pts = [Point(*p1), Point(*p2)]
        if is_hag.get():
            pts = hag_to_nap(pts)
        return cls(pts[0], pts[1])


class Cylinder:
    def __init__(
        self,
        segment: Segment,
        min_radius: float,
        max_radius: float,
        step_length: float,
        radius_mode: RadiusMode = "fixed",
    ):
        self.segment = segment
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.step_length = step_length
        self.radius_mode = radius_mode

    @property
    def radius(self) -> float:
        return self.max_radius

    def radius_at_t(self, t: np.ndarray | float) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.float64)
        t_clipped = np.clip(t_arr, 0.0, 1.0)

        if self.radius_mode == "fixed" or self.min_radius == self.max_radius:
            return np.full_like(t_clipped, self.max_radius, dtype=np.float64)

        return self.min_radius + t_clipped * (self.max_radius - self.min_radius)


def download_dtm_raster(aoi: AOIPolygon, buffer: float = 2.0) -> ahn.AhnRaster:
    """
    Download the AHN DTM raster for a given polygon area.
    """
    logger = get_logger("Utils")
    min_x, min_y, max_x, max_y = aoi.bounds
    buffered_extent = [
        min_x - buffer,
        max_x + buffer,
        min_y - buffer,
        max_y + buffer,
    ]
    logger.debug(f"Downloading AHN DTM for extent {buffered_extent}")
    try:
        return ahn.download_latest_ahn_from_wcs(buffered_extent, identifier="dtm_05m")
    except Exception as e:
        raise RuntimeError(f"Failed to download AHN DTM: {e}")


def sample_dtm(dtm_raster, points: list[Point]) -> np.ndarray:
    """
    Sample ground elevations from a DTM raster at the given (x, y) coordinates.
    """
    mean_val = dtm_raster.mean(skipna=True).values
    ground = np.full(len(points), mean_val, dtype=np.float64)

    for i, point in enumerate(points):
        try:
            val = dtm_raster.sel(x=point.x, y=point.y, method="nearest").values
        except Exception:
            val = np.nan

        if np.isnan(val):
            # fill with previous valid value (or mean if first)
            ground[i] = ground[i - 1] if i > 0 else mean_val
        else:
            ground[i] = val

    return ground


def generate_grid(Area: AOIPolygon, resolution: int, z_height: float = 0.0, two_d: bool = False) -> list[Point]:
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
    return [Point(x, y, z) for x, y, z in filtered]


def sample_polygon_boundary(polygon: AOIPolygon, sample_distance: float, z_height: float = 0.0) -> list[Point]:
    """Sample points at regular intervals along the exterior boundary of a polygon."""
    boundary = polygon.exterior
    total_length = boundary.length
    num_samples = max(2, int(np.ceil(total_length / sample_distance)))
    distances = np.linspace(0, total_length, num_samples, endpoint=False)
    return [Point(boundary.interpolate(d).x, boundary.interpolate(d).y, z_height) for d in distances]


def export_grid_to_copc(grid_points: list[Point], output_path: Path):
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

    write_to_copc(point_data, output_path)


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


def write_to_copc(points_in_cylinder: np.ndarray, output_path: Path):
    write_pipeline = {
        "pipeline": [
            {
                "type": "writers.copc",
                "filename": str(output_path),
                "forward": "all",
                "extra_dims": "all",
            }
        ]
    }

    writer = pdal.Pipeline(json.dumps(write_pipeline), arrays=[points_in_cylinder])
    writer.execute()
    logger.info(f"Wrote {points_in_cylinder.size} points to {output_path}")


@timed("Loading points for runs")
def load_points_for_runs(point_pairs: list[Segment], radius: float, input_path: Path = DEFAULT_INPUT) -> tuple[np.ndarray, np.ndarray, cKDTree]:
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
    radius_mode: RadiusMode = "fixed",
    min_radius: float | None = None,
    max_radius: float | None = None,
    step_length: float | None = None,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    intervisibility_func: Callable = calculate_intervisibility,
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

    if WRITE_TO_FILE:
        write_to_copc(out_array, output_path)

    return visibility_values, out_array


@timed("Viewshed calculation")
def calculate_viewshed(
    target: Point,
    search_radius: float,
    cylinder_radius: float,
    radius_mode: RadiusMode = "fixed",
    min_radius: float | None = None,
    max_radius: float | None = None,
    step_length: float | None = None,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    thinning_factor: int = 1,
    intervisibility_func: Callable = calculate_intervisibility,
    chunk_size: float = DEFAULT_CHUNK_SIZE,
) -> None:
    """
    For every point within search_radius of target, compute the intervisibility
    from target to that point (using a cylinder of cylinder_radius around each
    line of sight) and write the result as a new 'Visibility' dimension.

    Parameters
    ----------
    target : Point
        The observer / source point.
    search_radius : float
        3-D radius around target to select candidate points.
    cylinder_radius : float
        Radius of the LoS cylinder used for each intervisibility calculation.
    input_path : Path
        COPC input file.
    output_path : Path
        Output COPC file with the added Visibility dimension.
    thinning_factor : int
        Optional factor to thin the number of points for testing (e.g., 10 means use every 10th point).
    """
    # Load all points in the bounding box around the target
    # Create a dummy pair so we can reuse load_points_for_runs with search_radius
    # Code is **** ugly but it will work for now
    dummy_pair = Segment(target, target)
    if radius_mode == "fixed":
        min_radius = cylinder_radius
        max_radius = cylinder_radius

    array_points, array_coords, KDtree = load_points_for_runs([dummy_pair], search_radius + max_radius, input_path=input_path)

    # Select points within search_radius of the target
    target_coords = target.array_coords
    distances = np.linalg.norm(array_coords - target_coords, axis=1)
    sphere_mask = distances <= search_radius
    sphere_indices = np.where(sphere_mask)[0]

    if sphere_indices.size == 0:
        logger.error("No points within search radius of target.")
        sys.exit(1)

    logger.info(f"Computing visibility for {sphere_indices.size} points with {thinning_factor}x thinning within a {search_radius}m radius.")

    # Compute intervisibility for each thinned point
    thinned_sphere_indices = sphere_indices[::thinning_factor]
    visibility_values = np.ones(thinned_sphere_indices.size, dtype=np.float64)

    for i, idx in enumerate(tqdm(thinned_sphere_indices, desc="Viewshed", unit="pts")):
        pt_coords = array_coords[idx]
        dest = Point(pt_coords[0], pt_coords[1], pt_coords[2])

        segment = Segment(target, dest)
        cylinder = Cylinder(
            segment=segment,
            min_radius=min_radius,
            max_radius=max_radius,
            step_length=step_length,
            radius_mode=radius_mode,
        )
        visibility_values[i], _ = intervisibility_func(cylinder, array_points, array_coords, KDtree, chunk_size=chunk_size)

    # Build output array with Visibility dimension (only thinned points)
    thinned_points = array_points[thinned_sphere_indices]

    # Create a new dtype that includes Visibility
    new_dtype = np.dtype(thinned_points.dtype.descr + [("Visibility", "<f8")])
    out_array = np.empty(thinned_points.size, dtype=new_dtype)

    # Copy existing fields
    for name in thinned_points.dtype.names:
        out_array[name] = thinned_points[name]
    out_array["Visibility"] = visibility_values

    if WRITE_TO_FILE:
        write_to_copc(out_array, output_path)


def hag_to_nap(points: list["Point"], buffer: float = 2.0) -> list["Point"]:
    """
    Converts a list of points from Height Above Ground (HAG) to NAP.
    """
    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    aoi = AOIPolygon(ShapelyPolygon.from_bounds(xs.min(), ys.min(), xs.max(), ys.max()))
    dtm = download_dtm_raster(aoi=aoi, buffer=buffer)
    ground_levels = sample_dtm(dtm, points).tolist()
    translated_points = []

    for point, ground in zip(points, ground_levels):
        # NAP = HAG + Ground
        z_nap = point.z + ground
        translated_points.append(Point(point.x, point.y, z_nap))

    return translated_points


def nap_to_hag(points: list["Point"], buffer: float = 2.0) -> list["Point"]:
    """
    Converts a list of points from NAP to Height Above Ground (HAG).
    """
    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    aoi = AOIPolygon(ShapelyPolygon.from_bounds(xs.min(), ys.min(), xs.max(), ys.max()))
    dtm = download_dtm_raster(aoi=aoi, buffer=buffer)
    ground_levels = sample_dtm(dtm, points).tolist()
    translated_points = []

    for point, ground in zip(points, ground_levels):
        # HAG = NAP - Ground
        z_hag = point.z - ground
        translated_points.append(Point(point.x, point.y, z_hag))

    return translated_points


def calculate_viewshed_2d(
    target: Point,
    aoi: AOIPolygon,
    resolution: float = 1.0,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
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
    grid_points_nap = hag_to_nap(boundary_points)
    # export_grid_to_copc(grid_points_nap, output_path=Path("data/grid_points_2d_edges.copc.laz"))

    # For each point, calculate the visibility from the target to that point
    visibility_values, visibility_points = calculate_viewshed_for_grid(
        target=target,
        grid_points=grid_points_nap,
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

    return grid_points_nap, visibility_values, visibility_points


def calculate_flight_height(
    aoi: AOIPolygon,
    resolution: float = 1.0,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    visibility_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute HAG flight height grid from visible 3D viewshed points."""

    # Get points from the 3d vieshed below the visibility threshold
    minx, miny, maxx, maxy = aoi.bounds
    pipeline = {
        "pipeline": [
            {"type": "readers.copc", "filename": str(input_path)},
            {"type": "filters.range", "limits": f"Visibility({visibility_threshold}:)"},
        ]
    }
    pl = pdal.Pipeline(json.dumps(pipeline))
    pl.execute()
    arr = pl.arrays[0]
    X = arr["X"]
    Y = arr["Y"]
    Z = arr["Z"]

    # Create a grid at the specified resolution and find the minimum Z value for points that fall into each grid cell
    x_coords = np.arange(minx, maxx + resolution, resolution)
    y_coords = np.arange(miny, maxy + resolution, resolution)
    grid_xx, grid_yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    grid_min_z = np.full(grid_xx.shape, np.nan, dtype=float)

    xi = np.floor((X - minx) / resolution).astype(int)
    yi = np.floor((Y - miny) / resolution).astype(int)
    for x, y, z in zip(xi, yi, Z):
        current = grid_min_z[y, x]
        if np.isnan(current) or z < current:
            grid_min_z[y, x] = z

    # Fill empty cells with the maximum Z value from the input points (or 0 if no points)
    max_z = np.nanmax(Z) if Z.size > 0 else 0.0
    grid_min_z[np.isnan(grid_min_z)] = max_z

    # Flatten the grid arrays
    flat_x = grid_xx.ravel()
    flat_y = grid_yy.ravel()
    flat_z = grid_min_z.ravel()

    # Calculate HAG
    grid_points = [Point(x, y, 0.0) for x, y in zip(flat_x, flat_y)]
    try:
        ground_elev = sample_dtm(download_dtm_raster(aoi, buffer=2.0), grid_points)
    except Exception as e:
        logger.warning(f"Could not sample DTM for HAG conversion: {e}")
        ground_elev = np.zeros_like(flat_z)
    flat_hag = flat_z - ground_elev

    save_viewshed_as_tif(
        x_coords=flat_x,
        y_coords=flat_y,
        visibility_values=flat_hag,
        aoi=aoi,
        resolution=resolution,
        output_path=output_path,
    )

    return flat_x, flat_y, flat_hag


def demo_flight_height():
    """
    Interactive demo: select target and AOI, generate 3D grid, compute 3D viewshed, then compute flight height ceiling and output GeoTIFF.
    """

    # Select target and AOI interactively
    target_NAP = Point(233851.5, 581986.8, 1.7)
    target = hag_to_nap([target_NAP])[0]
    export_grid_to_copc([target], output_path=Path("data/target_point.copc.laz"))
    aoi = AOIPolygon.get_from_file(Path("data/example_aoi/Groningen_plein.geojson"))
    resolution = 2.0
    radius = 0.15
    z_height = 100.0  # max Z for grid
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
    export_grid_to_copc(grid_points, output_path=Path("data/grid_points_3d_shell.copc.laz"))

    # Compute 3D viewshed for shell grid points
    radius = 0.15
    chunk_size = 5
    viewshed_output_path = Path("data/viewshed_3d_output.copc.laz")
    calculate_viewshed_for_grid(
        target=target,
        grid_points=grid_points,
        cylinder_radius=radius,
        input_path=DEFAULT_INPUT,
        output_path=viewshed_output_path,
        intervisibility_func=calculate_intervisibility,
        chunk_size=chunk_size,
    )

    # Compute flight height ceiling from the 3D viewshed COPC
    flight_height_output_path = Path("data/flight_height_output.tif")
    calculate_flight_height(
        aoi=aoi,
        resolution=resolution,
        input_path=viewshed_output_path,
        output_path=flight_height_output_path,
        visibility_threshold=0.5,
    )

    print(f"Flight height GeoTIFF written to {flight_height_output_path}")


def demo_viewshed_2d():
    target_NAP = Point(233851.5, 581986.8, 1.7)
    target = hag_to_nap([target_NAP])[0]
    export_grid_to_copc([target], output_path=Path("data/target_point.copc.laz"))
    aoi = AOIPolygon.get_from_file(Path("data/example_aoi/Groningen_plein.geojson"))
    resolution = 1.0
    radius = 0.15
    output_path = Path("data/viewshed_2d_output")
    calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        radius=radius,
        resolution=resolution,
        input_path=DEFAULT_INPUT,
        output_path=output_path,
        z_offset=0.3,
    )


def demo_viewshed_from_grid():
    target = Point.get_from_user("Select target point for viewshed calculation")
    area = AOIPolygon.get_from_user("Select area for grid generation")
    resolution = 1.0
    grid_points = generate_grid(area, resolution, z_height=40.0)
    radius = 0.15
    chunk_size = 5
    # export_grid_to_copc(grid_points, output_path="data/grid_points.copc.laz")
    calculate_viewshed_for_grid(
        target=target,
        grid_points=grid_points,
        cylinder_radius=radius,
        input_path=DEFAULT_INPUT,
        output_path=DEFAULT_OUTPUT,
        intervisibility_func=calculate_intervisibility,
        chunk_size=chunk_size,
    )


def demo_viewshed_from_cloud():
    target = Point(233974.5, 582114.2, 5.0)
    search_radius = 100
    radius = 0.15
    thinning_factor = 10
    chunk_size = 5
    calculate_viewshed(
        target=target,
        search_radius=search_radius,
        cylinder_radius=radius,
        input_path=DEFAULT_INPUT,
        output_path=DEFAULT_OUTPUT,
        thinning_factor=thinning_factor,
        intervisibility_func=calculate_intervisibility,
        chunk_size=chunk_size,
    )


def demo_point_to_point():
    pair = Segment.get_from_user("Select points for intervisibility")
    radius = 0.15
    visibility = calculate_point_to_point(pair, radius)
    logger.info(f"Calculated visibility: {visibility:.4f}")


def demo_hag_grid():
    area = AOIPolygon.get_from_user("Select area for HAG grid generation")
    grid_points = generate_grid(area, resolution=1.0, z_height=0.0)

    nap_points = hag_to_nap(grid_points)

    output_path = Path("data/hag_grid.copc.laz")
    export_grid_to_copc(nap_points, output_path=output_path)


def only_save_viewable_volume(input_path: Path, output_path: Path):
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
    write_to_copc(arr, output_path)
    logger.info(f"Saved viewable volume to {output_path}")


if __name__ == "__main__":
    # city_block = Segment(Point(233609.0, 581598.0, 0.0), Point(233957.0, 581946.0, 20.0))
    # park = Segment(Point(233974.5, 582114.2, 5.0), Point(233912.2, 582187.5, 10.0))

    # demo_point_to_point()
    # demo_viewshed_from_cloud()
    # demo_viewshed_from_grid()
    # demo_hag_grid()
    # demo_viewshed_2d()
    demo_flight_height()
    only_save_viewable_volume(input_path=Path("data/viewshed_3d_output.copc.laz"), output_path=Path("data/viewable_volume.copc.laz"))
