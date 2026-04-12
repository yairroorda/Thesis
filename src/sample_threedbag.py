import json
from pathlib import Path

import numpy as np
import pdal

from utils import get_logger, timed

logger = get_logger(name="Sample3DBAG")


def _read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    triangles = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                face = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                for i in range(1, len(face) - 1):
                    triangles.append((face[0], face[i], face[i + 1]))

    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def _sample_points_on_triangles(vertices: np.ndarray, triangles: np.ndarray, density: float) -> tuple[np.ndarray, int]:
    tri_verts = vertices[triangles]
    areas = 0.5 * np.linalg.norm(np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0]), axis=1)
    degenerate_triangles = int(np.count_nonzero(areas == 0.0))
    valid = areas > 0.0
    tri_verts = tri_verts[valid]
    areas = areas[valid]

    rng = np.random.default_rng()
    counts = rng.poisson(areas * density)

    if not counts.any():
        return np.empty((0, 3), dtype=np.float64), degenerate_triangles

    sampled_tris = tri_verts[np.repeat(np.arange(len(tri_verts)), counts)]
    r1 = rng.random(len(sampled_tris))
    r2 = rng.random(len(sampled_tris))
    s = np.sqrt(r1)
    points = (1.0 - s)[:, None] * sampled_tris[:, 0] + (s * (1.0 - r2))[:, None] * sampled_tris[:, 1] + (s * r2)[:, None] * sampled_tris[:, 2]
    return points, degenerate_triangles


def _write_points(points_xyz, output_path):
    arr = np.zeros(len(points_xyz), dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1")])
    arr["X"] = points_xyz[:, 0]
    arr["Y"] = points_xyz[:, 1]
    arr["Z"] = points_xyz[:, 2]
    arr["Classification"] = 6

    pdal.Pipeline(json.dumps({"pipeline": [{"type": "writers.copc", "filename": str(output_path), "forward": "all", "extra_dims": "all"}]}), arrays=[arr]).execute()
    logger.info(f"Wrote {len(points_xyz)} sampled points to {output_path}")


@timed("Sample points on 3DBAG mesh")
def sample_on_mesh(input_path: Path, output_path: Path, density) -> Path:
    vertices, triangles = _read_obj_mesh(input_path)
    sampled_xyz, degenerate_triangles = _sample_points_on_triangles(vertices, triangles, density)
    _write_points(sampled_xyz, output_path)
    logger.info(f"Sampling complete. density={density:.6f}, points={len(sampled_xyz)}, degenerate_triangles={degenerate_triangles}")
    return output_path


if __name__ == "__main__":
    sample_on_mesh(
        input_path=Path("data/3dbag_lod22_merged.obj"),
        output_path=Path("data/3dbag_sampled_test.copc.laz"),
        density=10.0,
    )
