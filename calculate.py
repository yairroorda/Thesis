import json
import numpy as np
import pdal

from utils import timed, get_logger


logger = get_logger(name = "Calculate")

DEFAULT_INPUT = r"data/output_classified.copc.laz"
DEFAULT_OUTPUT = r"data/output_line_of_sight.copc.laz"


def build_bounds(point1, point2, radius):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    minx = min(x1, x2) - radius
    maxx = max(x1, x2) + radius
    miny = min(y1, y2) - radius
    maxy = max(y1, y2) + radius
    minz = min(z1, z2) - radius
    maxz = max(z1, z2) + radius

    # return as a string for the PDAL crop filter
    return f"([{minx},{maxx}],[{miny},{maxy}],[{minz},{maxz}])"


def distance_mask(points, point1, point2, radius):

    # Convert inputs to numpy arrays for vectorized operations
    p1 = np.asarray(point1, dtype=np.float64)
    p2 = np.asarray(point2, dtype=np.float64)
    radius = float(radius)

    # Compute the vector from p1 to p2 and its denominator for projection
    segment = p2 - p1
    denom = np.dot(segment, segment)
    # Compute the projection of each point onto the line defined by p1 and p2
    # Clip t to the range [0, 1] to restrict to the LoS segment
    t = np.clip(((points - p1) @ segment) / denom, 0.0, 1.0)
    # Get the coordinates of the closest points on the segment for each point in the input
    proj = p1 + t[:, None] * segment
    # Compute the distance from each point to its projection on the line
    distances = np.linalg.norm(points - proj, axis=1)

    # Return a boolean mask of points within the specified radius
    return distances <= radius

@timed("Line of sight calculation")
def calculate_line_of_sight(point1, point2, r, input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    """Filter points within radius r of the LoS between point1 and point2.

    Writes the filtered points to a COPC file.
    """

    bounds = build_bounds(point1, point2, r)

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
        return 0

    arrays = pipeline.arrays
    if not arrays:
        logger.warning("No point data returned from PDAL pipeline.")
        return 0
    logger.debug(f"Points in bounding box: {arrays[0].size}")

    points = np.concatenate(arrays)
    coords = np.column_stack((points["X"], points["Y"], points["Z"]))
    mask = distance_mask(coords, point1, point2, r)
    filtered = points[mask]
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

    write_pipeline = {
        "pipeline": [
            {
                "type": "writers.copc",
                "filename": output_path,
                "forward": "all",
            }
        ]
    }

    writer = pdal.Pipeline(json.dumps(write_pipeline), arrays=[filtered])
    writer.execute()

    logger.info(f"Wrote {filtered.size} points to {output_path}")

    return filtered.size


if __name__ == "__main__":
    p1 = (233974.5, 582114.2, 8.0)
    p2 = (233912.2, 582187.5, 10.0)
    radius = 3
    processed = calculate_line_of_sight(p1, p2, radius)

# city block
    # p1 = (233609.0, 581598.0, 0.0)
    # p2 = (233957.0, 581946.0, 20.0)

#park
    # p1 = (233974.5, 582114.2, 5.0)
    # p2 = (233912.2, 582187.5, 10.0)