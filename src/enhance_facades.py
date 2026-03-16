"""
Preprocessing step: generate building-facade points from classified roof points.

After classification (segment.py), this module:
- Reads building points from the classified COPC.
- Computes surface normals to identify roof surfaces (non-vertical).
- Clusters roof points per building using DBSCAN.
- Extracts concave-hull boundaries for each building footprint.
- Generates dense vertical facade (wall) points along the boundary edges from the roof height down to the ground.
- Merges the facade points back into the full point cloud and writes it out.
"""

import json
from pathlib import Path

import numpy as np
import pdal
from scipy.spatial import cKDTree
from shapely import concave_hull
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from calculate import download_dtm_raster
from query_copc import Polygon
from utils import get_logger, timed

logger = get_logger(name="EnhanceFacades")


def read_building_points(input_path: Path) -> np.ndarray:
    """
    Read building-class points (Classification == 6) and compute surface normals.
    """
    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.copc",
                "filename": str(input_path),
            },
            {"type": "filters.expression", "expression": "Classification == 6"},
            {
                "type": "filters.normal",
                "knn": 15,
            },
        ]
    }
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    arrays = pipeline.arrays
    if not arrays or arrays[0].size == 0:
        logger.info(f"No building points found in {input_path}")
        return np.empty(0)
    return arrays[0]


def filter_roof_points(building_pts: np.ndarray, normal_z_threshold: float) -> np.ndarray:
    """
    Keep only points whose surface normal is not (nearly) vertical.
    """
    normal_z = np.abs(building_pts["NormalZ"])
    mask = normal_z > normal_z_threshold
    logger.info(f"Roof filter: keeping {{{mask.sum()}}} / {len(building_pts)} building points (|NormalZ| > {normal_z_threshold:.2f})")
    return building_pts[mask]


def cluster_buildings(roof_pts: np.ndarray, eps: float, min_samples: int = 5) -> np.ndarray:
    """
    Cluster roof points into individual buildings using DBSCAN.
    """
    coords_2d = np.column_stack((roof_pts["X"], roof_pts["Y"], roof_pts["Z"]))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords_2d)
    n_clusters = len(set(labels)) - 1  # exclude noise label (-1)
    logger.info(f"DBSCAN found {n_clusters} roofs (eps={eps})")
    return labels


def boundary_from_cluster(
    cluster_pts: np.ndarray,
    hull_ratio: float,
) -> np.ndarray | None:
    """
    Compute a concave-hull boundary for a given building cluster.
    """
    xy = np.column_stack((cluster_pts["X"], cluster_pts["Y"]))
    z = cluster_pts["Z"]

    # `shapely.concave_hull` requires at least 4 points to form a polygon
    if len(xy) < 4:
        return None

    # Concave hull on the 2-D projection
    mp = MultiPoint(xy)
    hull = concave_hull(mp, ratio=hull_ratio)

    if hull.is_empty:
        return None

    # Extract exterior ring coordinates (last vertex == first; drop duplicate)
    ring = np.array(hull.exterior.coords)[:-1]

    # Assign roof Z to each boundary vertex via nearest roof point
    tree_2d = cKDTree(xy)
    _, idx = tree_2d.query(ring)
    ring_z = z[idx]

    return np.column_stack((ring[:, 0], ring[:, 1], ring_z))


def generate_facade_points(
    boundary_verts: np.ndarray,
    dtm_raster,
    point_spacing: float,
) -> np.ndarray:
    """
    Generate dense vertical facade points along the boundary edges of one building.

    For each consecutive pair of boundary vertices the function:
    - walks along the edge at *point_spacing* intervals,
    - interpolates the roof Z between the two endpoints,
    - samples the DTM to get the ground Z,
    - creates a vertical column of points from roof Z down to ground Z.
    """
    all_points: list[np.ndarray] = []
    mean_ground = float(dtm_raster.mean(skipna=True).values)
    n_verts = len(boundary_verts)

    for i in range(n_verts):
        p1 = boundary_verts[i]
        p2 = boundary_verts[(i + 1) % n_verts]

        edge_vec = p2[:2] - p1[:2]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-6:
            continue

        n_steps = max(int(np.ceil(edge_len / point_spacing)), 1)
        t_vals = np.linspace(0.0, 1.0, n_steps, endpoint=False)

        for t in t_vals:
            x = p1[0] + t * edge_vec[0]
            y = p1[1] + t * edge_vec[1]
            z_roof = p1[2] + t * (p2[2] - p1[2])

            if not np.isfinite(z_roof):
                continue

            # Ground elevation from DTM
            try:
                z_ground = float(dtm_raster.sel(x=x, y=y, method="nearest").values)
            except Exception:
                z_ground = mean_ground

            if not np.isfinite(z_ground):
                z_ground = mean_ground

            if not np.isfinite(z_ground) or z_roof <= z_ground:
                continue

            n_vert = max(int(np.ceil((z_roof - z_ground) / point_spacing)), 1)
            z_vals = np.linspace(z_ground, z_roof, n_vert, endpoint=True)

            col = np.column_stack(
                (
                    np.full(n_vert, x),
                    np.full(n_vert, y),
                    z_vals,
                )
            )
            all_points.append(col)

    if not all_points:
        return np.empty((0, 3))
    return np.vstack(all_points)


def build_structured_array(coords: np.ndarray, classification: int = 6) -> np.ndarray:
    """
    Build a PDAL-compatible structured numpy array from (M, 3) XYZ coordinates.
    """
    n = len(coords)
    dtype = np.dtype(
        [
            ("X", np.float64),
            ("Y", np.float64),
            ("Z", np.float64),
            ("Classification", np.uint8),
        ]
    )
    array = np.zeros(n, dtype=dtype)
    array["X"] = coords[:, 0]
    array["Y"] = coords[:, 1]
    array["Z"] = coords[:, 2]
    array["Classification"] = classification
    return array


def merge_and_write(input_path: Path, facade_array: np.ndarray, output_path: Path) -> None:
    """
    Merge the original point cloud with generated facade points and write to COPC.
    """
    pipeline_spec = {
        "pipeline": [
            {
                "type": "readers.copc",
                "filename": str(input_path),
            },
            {
                "type": "filters.merge",
            },
            {
                "type": "writers.copc",
                "filename": str(output_path),
                "forward": "all",
                "extra_dims": "all",
            },
        ]
    }
    pipeline = pdal.Pipeline(
        json.dumps(pipeline_spec),
        arrays=[facade_array],
    )
    count = pipeline.execute()
    logger.info(f"Wrote {count} points (incl. {len(facade_array)} facade) to {output_path}")


@timed("Generate facades")
def generate_facades(
    input_path: Path,
    output_path: Path,
    point_spacing: float = 0.2,
    normal_z_threshold: float = 0.3,
    cluster_eps: float = 1.0,
    hull_ratio: float = 0.3,
) -> None:
    """
    Add point on building facades derived from classified roof surfaces.
    """
    logger.info(f"starting facade enhancement: input={input_path}, output={output_path}")

    # Read building points and compute normals
    building_pts = read_building_points(input_path)
    if building_pts.size == 0:
        logger.warning("No building points, skipping facade generation.")
        return

    # Keep only roof surfaces
    roof_pts = filter_roof_points(building_pts, normal_z_threshold)
    if roof_pts.size == 0:
        logger.warning("No roof points after filtering, skipping facade generation.")
        return

    # Cluster into individual buildings
    labels = cluster_buildings(roof_pts, eps=cluster_eps)
    unique_labels = sorted(set(labels) - {-1})  # exclude noise label (-1)
    if not unique_labels:
        logger.warning("No building clusters found, skipping facade generation.")
        return

    # Download DTM raster once for the full aoi extent
    aoi = Polygon.from_bounds(
        roof_pts["X"].min(),
        roof_pts["Y"].min(),
        roof_pts["X"].max(),
        roof_pts["Y"].max(),
    )
    dtm_raster = download_dtm_raster(aoi=aoi, buffer=5.0)

    # Calculate facade points for each building cluster and accumulate
    all_facade_coords: list[np.ndarray] = []
    for label in tqdm(unique_labels, desc="Generating facades"):
        cluster = roof_pts[labels == label]
        boundary = boundary_from_cluster(cluster, hull_ratio)
        if boundary is None:
            logger.debug(f"Cluster {label}: too small for boundary (n={len(cluster)}), skipping")
            continue
        facade_coords = generate_facade_points(boundary, dtm_raster, point_spacing)
        if facade_coords.size > 0:
            all_facade_coords.append(facade_coords)

    if not all_facade_coords:
        logger.warning("No facade points generated — skipping merge.")
        return

    facade_xyz = np.vstack(all_facade_coords)

    # Build structured array and merge into full cloud
    facade_array = build_structured_array(facade_xyz, classification=6)
    merge_and_write(input_path, facade_array, output_path)


def demo_generate_facades():
    """Run the full facade generation on a small cropped area for quick testing."""
    input_path = Path("data/output_classified.copc.laz")
    output_path = Path("data/output_with_facades.copc.laz")

    generate_facades(input_path=input_path, output_path=output_path, point_spacing=0.5, normal_z_threshold=0.3, cluster_eps=1.0, hull_ratio=0.2)


if __name__ == "__main__":
    demo_generate_facades()
