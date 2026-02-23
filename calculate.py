import json
import numpy as np
import pdal
import sys
from scipy.spatial import cKDTree
from tqdm import tqdm
from pathlib import Path

from utils import timed, get_logger, compare
from gui import make_map, _TO_RD


logger = get_logger(name="Calculate")

DEFAULT_INPUT = Path(r"data/output_classified.copc.laz")
DEFAULT_OUTPUT = Path(r"data/output_viewshed.copc.laz")

WRITE_TO_FILE = True  # Control whether to write query outputs to a file.

CLASS_TERRAIN = 2
CLASS_BUILDING = 6
CLASS_VEGETATION = {3, 4, 5}  # low, medium, high vegetation classes

# Figuring out what to set these thresholds to is a key part of the research but we can start with this for now.
TERRAIN_DENSITY_THRESHOLD = 14
BUILDING_DENSITY_THRESHOLD = 14
VEGETATION_DENSITY_THRESHOLD = 1
BEER_LAMBERT_COEFFICIENT = 0.05


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.array_coords = np.array([x, y, z], dtype=np.float64)
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def get_from_user(cls, title: str = "Set point") -> "Point":
        """Let the user pick one point on the map. Returns (x, y, z) in RD."""
        import tkinter as tk

        root, map_widget, controls = make_map(title)

        p_xy = {"v": None}
        marker = {"p": None}

        def on_click(coords):
            lat_c, lon_c = float(coords[0]), float(coords[1])
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

        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(10, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()

        p = (*p_xy["v"], float(pz.get()))
        root.destroy()
        return cls(*p)


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
    def get_from_user(cls, title: str = "Set P1/P2") -> "Segment":
        """Let the user pick two points on the map. Returns (p1, p2) with p1/p2 as (x, y, z) in RD."""
        import tkinter as tk

        root, map_widget, controls = make_map(title)

        mode = tk.StringVar(value="p1")
        p1_xy = {"v": None}
        p2_xy = {"v": None}
        markers = {"p1": None, "p2": None}

        def on_click(coords):
            lat_c, lon_c = float(coords[0]), float(coords[1])
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

        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(10, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()

        p1 = (*p1_xy["v"], float(p1z.get()))
        p2 = (*p2_xy["v"], float(p2z.get()))
        root.destroy()
        return cls(Point(*p1), Point(*p2))


class Cylinder:
    def __init__(self, segment: Segment, radius: float):
        self.segment = segment
        self.radius = radius


def get_distance_mask(point_array: np.ndarray[Point], cylinder: Cylinder) -> tuple[np.ndarray[bool], np.ndarray[float]]:
    segment = cylinder.segment
    radius = float(cylinder.radius)

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

    # Return a boolean mask of points within the specified radius, and the projection parameters
    return distances_squared <= radius**2, t


def get_kdtree_candidate_indices(KDtree: cKDTree, cylinder: Cylinder) -> np.ndarray[int]:
    """
    Generate candidate point indices from a KD-tree within a radius of the line segment.
    """
    segment = cylinder.segment
    radius = float(cylinder.radius)

    step = radius
    num_samples = max(2, int(np.ceil(segment.length / step)) + 1)
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    samples = segment.point1.array_coords + t[:, None] * segment.vector

    # calculate query radius knowing R_sphere = sqrt(r_cylinder² + (step²/4))
    query_radius = np.sqrt(radius**2 + (step**2 / 4))
    candidate_lists = KDtree.query_ball_point(samples, r=query_radius, workers=1)  # workers=-1 causes to much overhead for small queries.

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


def write_to_copc(points_in_cylindar: np.ndarray, output_path: Path):
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

    writer = pdal.Pipeline(json.dumps(write_pipeline), arrays=[points_in_cylindar])
    writer.execute()
    logger.info(f"Wrote {points_in_cylindar.size} points to {output_path}")


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


@timed("Line of sight calculation")
def calculate_number_of_points_in_cylinder(
    cylinder: Cylinder,
    array_points: np.ndarray,
    array_coords: np.ndarray,
    KDtree: cKDTree,
    output_path: Path = DEFAULT_OUTPUT,
) -> int:
    """Filter points within a radius of the line connecting point1 and point2."""

    if array_points is None or array_coords is None or KDtree is None:
        logger.error("Points, coordinates, and tree must be provided for line of sight calculation.")
        raise ValueError("Points, coordinates, and tree must be provided for line of sight calculation.")

    if cylinder.segment.length == 0.0:
        logger.error("Point1 and Point2 cannot be the same for line of sight calculation.")
        raise ValueError("Point1 and Point2 cannot be the same for line of sight calculation.")

    array_candidate_indices = get_kdtree_candidate_indices(KDtree, cylinder)

    if array_candidate_indices.size == 0:
        logger.warning("No candidate points found near the line of sight.")
        return 0

    candidate_coords = array_coords[array_candidate_indices]
    distance_mask, _ = get_distance_mask(candidate_coords, cylinder)
    filtered = array_points[array_candidate_indices[distance_mask]]
    logger.debug(f"Filtered points count: {filtered.size}")

    if filtered.size == 0:
        logger.warning("No points found within the specified radius of the line of sight.")
        return 0

    if "Classification" in filtered.dtype.names:
        classes, counts = np.unique(filtered["Classification"], return_counts=True)
        summary = ", ".join(f"{point_class}:{point_count}" for point_class, point_count in zip(classes, counts))
        logger.info(f"Class counts: {summary}")
    else:
        logger.warning("Classification dimension not found")

    if WRITE_TO_FILE:
        write_to_copc(filtered, output_path)

    return filtered.size


# @timed("Intervisibility calculation")
def calculate_intervisibility(
    cylinder: Cylinder,
    array_points: np.ndarray,
    array_coords: np.ndarray,
    KDtree: cKDTree,
) -> float:
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
    radius = float(cylinder.radius)
    cross_section_area = np.pi * radius**2

    # Get all points inside the cylinder
    candidate_indices = get_kdtree_candidate_indices(KDtree, cylinder)

    candidate_coords = array_coords[candidate_indices]
    distance_mask, t_all = get_distance_mask(candidate_coords, cylinder)
    filtered_indices = candidate_indices[distance_mask]

    if filtered_indices.size == 0:
        # logger.info("No points in cylinder – full visibility.")
        return 1.0

    filtered_points = array_points[filtered_indices]
    t_values = t_all[distance_mask]  # reuse projection parameters from get_distance_mask

    # if WRITE_TO_FILE:
    #     write_to_copc(filtered_points, DEFAULT_OUTPUT)

    # Define step bins along the segment
    step_length = radius  # same step size used by the KD-tree sampler
    num_steps = max(1, int(np.ceil(segment.length / step_length)))
    bin_edges = np.linspace(0.0, 1.0, num_steps + 1)

    # Pre-fetch classifications
    classifications = filtered_points["Classification"]

    # Masks per class (computed once)
    terrain_mask = classifications == CLASS_TERRAIN
    building_mask = classifications == CLASS_BUILDING
    vegetation_mask = np.isin(classifications, list(CLASS_VEGETATION))

    # Walk the LoS
    visibility = 1.0

    for i in range(num_steps):
        t_lo, t_hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (t_values >= t_lo) & (t_values < t_hi)
        # Include the endpoint in the last bin
        if i == num_steps - 1:
            in_bin |= t_values == t_hi

        # Check threshold for terrain
        terrain_count = int(np.count_nonzero(in_bin & terrain_mask))
        terrain_density = terrain_count / cross_section_area

        if terrain_density >= TERRAIN_DENSITY_THRESHOLD:
            # logger.info(f"Step {i + 1}/{num_steps}: terrain density {terrain_density:.2f} >= threshold {TERRAIN_DENSITY_THRESHOLD} – blocked.")
            return 0.0

        # Check threshold for buildings
        building_count = int(np.count_nonzero(in_bin & building_mask))
        building_density = building_count / cross_section_area

        if building_density >= BUILDING_DENSITY_THRESHOLD:
            # logger.info(f"Step {i + 1}/{num_steps}: building density {building_density:.2f} >= threshold {BUILDING_DENSITY_THRESHOLD} – blocked.")
            return 0.0

        # Decrease visibility for vegetation
        vegetation_count = int(np.count_nonzero(in_bin & vegetation_mask))
        vegetation_density = vegetation_count / cross_section_area

        if vegetation_density >= VEGETATION_DENSITY_THRESHOLD:
            actual_step = step_length if i < num_steps - 1 else segment.length - i * step_length
            attenuation = np.exp(-BEER_LAMBERT_COEFFICIENT * vegetation_density * actual_step)
            visibility *= attenuation
            # logger.debug(f"Step {i + 1}/{num_steps}: veg density {vegetation_density:.2f}, attenuation {attenuation:.4f}, visibility now {visibility:.4f}")

    visibility = float(np.clip(visibility, 0.0, 1.0))
    # logger.info(f"Final visibility: {visibility:.4f}")
    return visibility


def generate_example_points(base_p1: Point, base_p2: Point, num_pairs: int) -> list[Segment]:
    point_pairs = []
    for i in range(num_pairs):
        p1 = Point(base_p1.x + i, base_p1.y, base_p1.z)
        p2 = Point(base_p2.x + i, base_p2.y, base_p2.z)
        point_pairs.append(Segment(p1, p2))
    return point_pairs


def calculate_point_to_multiple_points(
    base_pair: Segment,
    radius: float,
    runs: int,
) -> None:
    point_pairs = generate_example_points(base_pair.point1, base_pair.point2, runs)
    array_points, array_coords, KDtree = load_points_for_runs(point_pairs, radius)
    if array_points is None:
        logger.warning("No points loaded for the requested runs.")
    else:
        LoS_counts: dict[int, int] = {}
        for Los_index, point_pair in enumerate(point_pairs, start=1):
            output_path = DEFAULT_OUTPUT.parent / f"output_{Los_index}.copc.laz"
            segment = Segment(point_pair.point1, point_pair.point2)
            cylinder = Cylinder(segment, radius)
            number_of_points = calculate_number_of_points_in_cylinder(
                cylinder,
                array_points,
                array_coords,
                KDtree,
                output_path=output_path,
            )
            logger.debug(f"Run {Los_index}/{runs}: processed {number_of_points} points")
            LoS_counts[Los_index] = number_of_points
    return LoS_counts


def calculate_point_to_point(point_pair: Segment, radius: float) -> float:
    array_points, array_coords, KDtree = load_points_for_runs([point_pair], radius)
    if array_points is None:
        logger.warning("No points loaded for the requested point pair.")
        return 1.0  # Assume full visibility if no points are loaded
    cylinder_of_sight = Cylinder(Segment(point_pair.point1, point_pair.point2), radius)
    visibility = calculate_intervisibility(
        cylinder_of_sight,
        array_points,
        array_coords,
        KDtree,
    )
    return visibility


@timed("Viewshed calculation")
def calculate_viewshed(
    target: Point,
    search_radius: float,
    cylinder_radius: float,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    thinning_factor: int = 1,
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
    array_points, array_coords, KDtree = load_points_for_runs([dummy_pair], search_radius, input_path=input_path)

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
        cylinder = Cylinder(segment, cylinder_radius)
        visibility_values[i] = calculate_intervisibility(cylinder, array_points, array_coords, KDtree)

    # Build output array with Visibility dimension (only thinned points)
    thinned_points = array_points[thinned_sphere_indices]

    # Create a new dtype that includes Visibility
    new_dtype = np.dtype(thinned_points.dtype.descr + [("Visibility", "<f8")])
    out_array = np.empty(thinned_points.size, dtype=new_dtype)

    # Copy existing fields
    for name in thinned_points.dtype.names:
        out_array[name] = thinned_points[name]
    out_array["Visibility"] = visibility_values

    write_to_copc(out_array, output_path)


if __name__ == "__main__":
    city_block = Segment(Point(233609.0, 581598.0, 0.0), Point(233957.0, 581946.0, 20.0))
    park = Segment(Point(233974.5, 582114.2, 5.0), Point(233912.2, 582187.5, 10.0))
    point_pair = park
    radius = 0.15

    # # Viewshed
    # target = Point(233974.5, 582114.2, 5.0)
    # search_radius = 50
    # thinning_factor = 10
    # calculate_viewshed(target, search_radius, radius, thinning_factor=thinning_factor)

    pair = Segment.get_from_user("Select points for intervisibility")
    radius = 3.0
    visibility = calculate_point_to_point(pair, radius)
    logger.info(f"Calculated visibility: {visibility:.4f}")
