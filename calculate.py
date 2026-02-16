import json
import numpy as np
import pdal
from scipy.spatial import cKDTree

from utils import timed, get_logger


logger = get_logger(name="Calculate")

DEFAULT_INPUT = r"data/output_classified.copc.laz"
DEFAULT_OUTPUT = r"data/output_line_of_sight.copc.laz"

WRITE_TO_FILE = False  # Control whether to write query outputs to a file.


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


class PointPair:
    def __init__(self, point1: Point, point2: Point):
        self.point1 = point1
        self.point2 = point2


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


class Cylinder:
    def __init__(self, segment: Segment, radius: float):
        self.segment = segment
        self.radius = radius


def get_distance_mask(point_array: np.ndarray[Point], cylinder: Cylinder) -> np.ndarray[bool]:
    segment = cylinder.segment
    denom = segment.length_squared
    radius = float(cylinder.radius)

    p1 = segment.point1.to_array()

    if denom == 0.0:
        distances = np.linalg.norm(point_array - p1, axis=1)
        return distances <= radius

    # Compute the projection of each point onto the line defined by p1 and p2
    # Clip t to the range [0, 1] to restrict to the LoS segment
    t = np.clip(((point_array - p1) @ segment.vector) / denom, 0.0, 1.0)
    # Get the coordinates of the closest points on the segment for each point in the input
    proj = p1 + t[:, None] * segment.vector
    # Compute the distance from each point to its projection on the line
    distances = np.linalg.norm(point_array - proj, axis=1)
    # Return a boolean mask of points within the specified radius
    return distances <= radius


def get_kdtree_candidate_indices(KDtree: cKDTree, cylinder: Cylinder) -> np.ndarray[int]:
    """
    Generate candidate point indices from a KD-tree within a radius of the line segment.
    """
    segment = cylinder.segment
    radius = float(cylinder.radius)

    step = radius
    num_samples = max(2, int(np.ceil(segment.length / step)) + 1)
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    samples = segment.point1.to_array() + t[:, None] * segment.vector

    # calculate query radius knowing R_sphere = sqrt(r_cylinder² + (step²/4))
    query_radius = np.sqrt(radius**2 + (step**2 / 4))
    candidate_lists = KDtree.query_ball_point(samples, r=query_radius)

    if len(candidate_lists) == 0:
        return np.array([], dtype=np.int64)

    return np.unique(np.concatenate([np.asarray(candidate, dtype=np.int64) for candidate in candidate_lists]))


def get_PDAL_bounds_for_runs(point_pairs: list[PointPair], radius: float) -> str:
    """Calculate the bounding box that contains all point pairs, expanded by the radius."""
    all_points = np.array(
        [pair.point1.to_array() for pair in point_pairs] + [pair.point2.to_array() for pair in point_pairs],
        dtype=np.float64,
    )
    minx, miny, minz = np.min(all_points, axis=0) - radius
    maxx, maxy, maxz = np.max(all_points, axis=0) + radius
    # Return string in the format expected by PDAL's filters.crop
    return f"([{minx},{maxx}],[{miny},{maxy}],[{minz},{maxz}])"


def write_to_copc(points_in_cylindar: np.ndarray, output_path: str):
    write_pipeline = {
        "pipeline": [
            {
                "type": "writers.copc",
                "filename": output_path,
                "forward": "all",
            }
        ]
    }

    writer = pdal.Pipeline(json.dumps(write_pipeline), arrays=[points_in_cylindar])
    writer.execute()
    logger.info(f"Wrote {points_in_cylindar.size} points to {output_path}")


@timed("Loading points for runs")
def load_points_for_runs(point_pairs: list[PointPair], radius: float, input_path: str = DEFAULT_INPUT) -> tuple[np.ndarray, np.ndarray, cKDTree]:
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
                "filename": input_path,
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
    output_path: str = DEFAULT_OUTPUT,
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
    distance_mask = get_distance_mask(candidate_coords, cylinder)
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


def generate_example_points(base_p1: Point, base_p2: Point, num_pairs: int) -> list[PointPair]:
    point_pairs = []
    for i in range(num_pairs):
        p1 = Point(base_p1.x + i, base_p1.y, base_p1.z)
        p2 = Point(base_p2.x + i, base_p2.y, base_p2.z)
        point_pairs.append(PointPair(p1, p2))
    return point_pairs


def calculate_point_to_multiple_points(
    base_pair: PointPair,
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
            output_path = f"output_{Los_index}.copc.laz"
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


def calculate_point_to_point(point_pair: PointPair, radius: float) -> None:
    array_points, array_coords, KDtree = load_points_for_runs([point_pair], radius)
    if array_points is None:
        logger.warning("No points loaded for the requested point pair.")
        return 0
    cylinder_of_sight = Cylinder(Segment(point_pair.point1, point_pair.point2), radius)
    number_of_points = calculate_number_of_points_in_cylinder(
        cylinder_of_sight,
        array_points,
        array_coords,
        KDtree,
        output_path=DEFAULT_OUTPUT,
    )
    return number_of_points

if __name__ == "__main__":
    city_block = PointPair(Point(233609.0, 581598.0, 0.0), Point(233957.0, 581946.0, 20.0))
    park = PointPair(Point(233974.5, 582114.2, 5.0), Point(233912.2, 582187.5, 10.0))
    point_pair = park
    radius = 3
    runs = 10
    LoS_count = calculate_point_to_point(point_pair, radius)
    print(f"Line of sight count for single pair: {LoS_count}")
    LOS_counts = calculate_point_to_multiple_points(point_pair, radius, runs)
    print("Line of sight counts for multiple runs:", LOS_counts)
