import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
import seaborn as sns
from shapely.geometry import box

from calculate import calculate_3d_viewshed
from enhance_facades import generate_facades
from models import AOICircle, AOIPolygon, ProjectConfig, ProjectPaths, RunConfig, RunPaths
from utils import get_logger
from visualize import save_viewshed_as_voxel_grid

logger = get_logger(name="TuneBuilding")


def _obj_bbox_intersects_aoi(used_indices: set[int], vertices: list[tuple[float, float, float]], aoi_polygon) -> bool:
    if not used_indices:
        return False
    xs = [vertices[i - 1][0] for i in used_indices]
    ys = [vertices[i - 1][1] for i in used_indices]
    return box(min(xs), min(ys), max(xs), max(ys)).intersects(aoi_polygon)


def _append_filtered_lod22_obj(src_obj: Path, out_file, aoi_polygon, object_prefix: str, vertex_offset: int) -> tuple[int, int]:
    vertices: list[tuple[float, float, float]] = []
    objects: list[tuple[str, list[list[int]], set[int]]] = []
    current_name = "unnamed"
    current_faces: list[list[int]] = []
    current_used: set[int] = set()

    def flush_current():
        nonlocal current_name, current_faces, current_used
        if current_faces:
            objects.append((current_name, current_faces, current_used))
        current_faces = []
        current_used = set()

    with open(src_obj) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("o ") or line.startswith("g "):
                flush_current()
                current_name = line.strip().split(maxsplit=1)[1]
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                face = [int(token.split("/")[0]) for token in tokens]
                current_faces.append(face)
                current_used.update(face)
    flush_current()

    kept_count = 0
    current_offset = vertex_offset
    for name, faces, used in objects:
        if not _obj_bbox_intersects_aoi(used, vertices, aoi_polygon):
            continue
        kept_count += 1
        ordered = sorted(used)
        remap = {old: i + 1 for i, old in enumerate(ordered)}
        out_file.write(f"o {object_prefix}_{name}\n")
        for idx in ordered:
            v = vertices[idx - 1]
            out_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            mapped = [str(current_offset + remap[idx]) for idx in face]
            out_file.write("f " + " ".join(mapped) + "\n")
        current_offset += len(ordered)

    return kept_count, current_offset


class ThreeDBAG:
    """
    3DBAG LoD22 OBJ downloader and filter by AOI
    based on 3d tiles and the script(https://github.com/3DBAG/3dbag-scripts/blob/main/tile_download.py#L31) by Ravi Peters (https://github.com/Ylannl)
    """

    name = "3DBAG"
    crs = "EPSG:28992"

    @classmethod
    def fetch(
        cls,
        aoi: "AOIPolygon",
        output_path: Path | str = "data/3dbag_lod22_merged.obj",
        keep_tiles: bool = False,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        aoi_rd = gpd.GeoDataFrame(geometry=[aoi.polygon], crs=aoi.crs).to_crs(cls.crs)
        aoi_poly = aoi_rd.geometry.iloc[0]
        bbox = tuple(aoi_rd.total_bounds)

        tiles = gpd.read_file("https://data.3dbag.nl/latest/tile_index.fgb", bbox=bbox)
        if tiles.crs != aoi_rd.crs:
            tiles = tiles.to_crs(aoi_rd.crs)
        tiles = tiles[tiles.intersects(aoi_poly)]

        if tiles.empty:
            raise RuntimeError("No 3DBAG tiles intersect AOI")

        tmp_dir = output_path.parent / "_3dbag_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tile_objs: list[Path] = []
        for _, tile in tiles.iterrows():
            tile_id = tile["tile_id"].replace("/", "-")
            zip_path = tmp_dir / f"{tile_id}.zip"
            urlretrieve(tile["obj_download"], zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                lod22_name = next(n for n in zf.namelist() if "LoD22" in n and n.endswith(".obj"))
                extracted = tmp_dir / f"{tile_id}-LoD22.obj"
                with zf.open(lod22_name) as src, open(extracted, "wb") as dst:
                    dst.write(src.read())
            tile_objs.append(extracted)

        total_kept = 0
        with open(output_path, "w") as merged:
            merged.write("# 3DBAG LoD22 merged OBJ\n")
            vertex_offset = 0
            for obj_path in tile_objs:
                kept, vertex_offset = _append_filtered_lod22_obj(
                    obj_path,
                    merged,
                    aoi_poly,
                    obj_path.stem,
                    vertex_offset,
                )
                total_kept += kept

        if not keep_tiles:
            for p in tmp_dir.glob("*"):
                if p.is_file():
                    p.unlink()
            tmp_dir.rmdir()

        return output_path


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


def sample_on_mesh(input_path: Path, output_path: Path, density) -> Path:
    vertices, triangles = _read_obj_mesh(input_path)
    sampled_xyz, degenerate_triangles = _sample_points_on_triangles(vertices, triangles, density)
    _write_points(sampled_xyz, output_path)
    logger.info(f"Sampling complete. density={density:.6f}, points={len(sampled_xyz)}, degenerate_triangles={degenerate_triangles}")
    return output_path


def get_sampled_threedbag(aoi: AOIPolygon, obj_path: Path, sampled_path: Path, filtered_path: Path, density: int = 150) -> Path:

    # get "perfect" point cloud by sampling the 3DBAG LoD22 mesh at a high density
    ThreeDBAG.fetch(aoi, output_path=obj_path)
    sample_on_mesh(
        input_path=obj_path,
        output_path=sampled_path,
        density=density,
    )
    filter_to_aoi(
        input_path=sampled_path,
        output_path=filtered_path,
        aoi=aoi.to_crs("EPSG:28992"),
    )

    return filtered_path


def filter_to_aoi(input_path: Path, output_path: Path, aoi: AOIPolygon) -> Path:
    """Filter COPC file to AOI polygon boundary."""
    pipeline = {
        "pipeline": [
            {"type": "readers.copc", "filename": str(input_path)},
            {"type": "filters.crop", "polygon": aoi.wkt},
            {"type": "writers.copc", "filename": str(output_path)},
        ]
    }
    count = pdal.Pipeline(json.dumps(pipeline)).execute()
    logger.info(f"Wrote {count} filtered points to {output_path}")
    return output_path


def filter_buildings(input_path: Path, output_path: Path) -> Path:
    """Filter point cloud to building points only, using the "Classification" dimension."""
    pipeline = {
        "pipeline": [
            {"type": "readers.copc", "filename": str(input_path)},
            {"type": "filters.range", "limits": "Classification[6:6]"},
            {"type": "writers.copc", "filename": str(output_path)},
        ]
    }
    count = pdal.Pipeline(json.dumps(pipeline)).execute()
    logger.info(f"Wrote {count} building points to {output_path}")
    return output_path


def _load_copc(path: Path) -> np.ndarray:
    """Helper to load a point cloud array using PDAL."""
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(path)}]}))
    pipeline.execute()
    return pipeline.arrays[0]


def compare_voxel_grid(path_my_method: Path, path_3dbag: Path, visibility_threshold: float = 0.0) -> dict[str, float]:
    logger.info(f"Comparing {path_my_method.name} vs {path_3dbag.name}")

    arr_my = _load_copc(path_my_method)
    arr_3dbag = _load_copc(path_3dbag)

    if len(arr_my) != len(arr_3dbag):
        raise ValueError(f"Grid size mismatch! My method: {len(arr_my)}, 3DBAG: {len(arr_3dbag)}.")

    order_my = np.lexsort((arr_my["X"], arr_my["Y"], arr_my["Z"]))
    arr_my = arr_my[order_my]

    order_3d = np.lexsort((arr_3dbag["X"], arr_3dbag["Y"], arr_3dbag["Z"]))
    arr_3dbag = arr_3dbag[order_3d]

    pred_visible = arr_my["Visibility"] > visibility_threshold
    truth_visible = arr_3dbag["Visibility"] > visibility_threshold

    total = len(pred_visible)
    tp = np.sum(pred_visible & truth_visible)
    tn = np.sum((~pred_visible) & (~truth_visible))
    fp = np.sum(pred_visible & (~truth_visible))
    fn = np.sum((~pred_visible) & truth_visible)

    # Calculate advanced metrics safely
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {"total_voxels": int(total), "TP_pct": float(tp / total) * 100, "TN_pct": float(tn / total) * 100, "FP_pct": float(fp / total) * 100, "FN_pct": float(fn / total) * 100, "Accuracy_pct": float((tp + tn) / total) * 100, "Precision": float(precision), "Recall": float(recall), "F1_Score": float(f1_score)}

    logger.info(f"F1: {f1_score:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
    return metrics


def run_comprehensive_sweep(z_height, resolution, project_paths, project_config, target_path, truth_voxel_path):
    """
    Nested Sweep: Tests multiple LoS Radii across multiple Facade Densities.
    """
    # 1.0m = sparse, 0.5m = medium, 0.3m = dense (roof match), 0.0 = Raw AHN (Baseline)
    spacings = [0.0, 2.0, 1.0, 0.75, 0.5, 0.3]
    radii = np.linspace(0.1, 0.75, 10)  # 0.1m to 0.75m in 10 steps

    results = []

    for spacing in spacings:
        print("\n======================================")
        print(f" STARTING SWEEP: Facade Spacing {spacing}m")
        print("======================================")

        # Handle the Baseline (Raw AHN) vs Enhanced Facades
        if spacing == 0.0:
            current_facades = project_paths.folder / "baseline_buildings_only.copc.laz"
            if not current_facades.exists():
                filter_buildings(project_paths.input_copc, current_facades)
            label = "Raw AHN (Baseline)"
        else:
            current_facades = project_paths.folder / f"facades_{spacing}m.copc.laz"
            if not current_facades.exists():
                temp_facades = project_paths.folder / f"facades_{spacing}m_temp.copc.laz"

                # 1. Generate to temp file
                generate_facades(project_paths.input_copc, temp_facades, point_spacing=spacing)

                # 2. Filter from temp to final
                filter_buildings(temp_facades, current_facades)

                # 3. Cleanup
                temp_facades.unlink()
            label = f"{spacing}m Spacing"

        project_paths.facades_copc = current_facades

        for r in radii:
            print(f"  -> Testing LoS Radius: {r}m")

            run_name = f"run_space_{spacing}_rad_{r}"
            run_cfg = RunConfig(
                name=run_name,
                z_height=z_height,
                resolution=resolution,
                target_source=target_path,
                los_radius=r,
            )

            run_paths = RunPaths(project_paths, run_name)
            run_paths.folder.mkdir(parents=True, exist_ok=True)

            # Compute and Voxelize
            calculate_3d_viewshed(project_config, project_paths, run_cfg, project_config.profile)
            save_viewshed_as_voxel_grid(run_paths, run_cfg, project_paths, project_config)

            # Compare against Truth
            metrics = compare_voxel_grid(run_paths.output_viewshed_voxel_grid_3d, truth_voxel_path)

            results.append({"Density_Label": label, "Point_Spacing": spacing, "LoS_Radius": r, "F1_Score": metrics["F1_Score"], "Precision": metrics["Precision"], "Recall": metrics["Recall"]})

    df = pd.DataFrame(results)
    df.to_csv(project_paths.folder / "comprehensive_sweep_results.csv", index=False)
    return df


def plot_comprehensive_chart(project_paths: ProjectPaths, metric: str = "F1_Score"):
    df = pd.read_csv(project_paths.folder / "comprehensive_sweep_results.csv")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # Plot lines grouped by the density label
    sns.lineplot(data=df, x="LoS_Radius", y=metric, hue="Density_Label", style="Density_Label", markers=True, dashes=False, linewidth=2.5, palette="viridis")

    plt.title("Impact of Facade Densification on Optimal LoS Radius", fontweight="bold", fontsize=14)
    plt.xlabel("Line-of-Sight Cylinder Radius (meters)", fontsize=12)
    plt.ylabel(f"{metric} vs 3DBAG Ground Truth", fontsize=12)

    # Optional annotation to highlight the narrative
    plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.legend(title="Facade Enhancement", loc="lower right")
    plt.xlim(0, 0.75)
    plt.ylim(0.6, 1)
    plt.tight_layout()

    plt.savefig(project_paths.folder / f"thesis_plot_interaction_{metric}.png", dpi=300)
    plt.close()
    print(f"Combined plot generated successfully: experiments/thesis_plot_interaction_{metric}.png")


def main():

    # Base run config
    z_height = 20.0
    resolution = 3.0

    # Setup project
    project_config = ProjectConfig(
        name="tune_building_threshold_delft_ahn5_0-75_CircleAOI",
        crs="EPSG:28992",
        dataset=["AHN5"],
        classification_method=None,
        overwrite=False,
    )

    project_paths = ProjectPaths(project_name=project_config.name, base_dir=Path("experiments"))

    # aoi = AOIPolygon.get(input_path=project_paths.aoi, title="Select AOI for Building Threshold Tuning")
    # aoi_rd = aoi.to_crs("EPSG:28992")
    project_paths.aoi = project_paths.folder / "eval_aoi.geojson"
    project_paths.classified_copc = project_paths.input_copc
    target_path = project_paths.folder / "target.copc.laz"

    aoi, is_hag = AOICircle.get(input_path=project_paths.aoi, title="Select Observer Target", crs=project_config.crs, overwrite=project_config.overwrite)
    aoi_rd = aoi.to_crs("EPSG:28992")

    # 3. Fetch Data
    if project_paths.input_copc.exists():
        logger.info(f"Project {project_config.name} is already prepared. Reusing cached preprocessing outputs.")
    else:
        from cloudfetch import AHN5

        datasource = AHN5()

        datasource.fetch(aoi=aoi_rd.polygon, aoi_crs=aoi_rd.crs, output_path=project_paths.input_copc)

        logger.info("Project preprocessing complete")

    # 4. Resolve HAG Paradox
    project_paths.classified_copc = project_paths.input_copc
    target_path = project_paths.folder / "target.copc.laz"

    target_pt = aoi.center

    if is_hag:
        from calculate import hag_to_ortho

        # Now that we downloaded the ground, we can shift the Z height
        target_pt = hag_to_ortho([target_pt], input_path=project_paths.input_copc)[0]

    target_pt.save_to_file(target_path, crs=project_config.crs)

    logger.info("Generating 3DBAG Ground Truth...")
    threedbag_filtered_path = project_paths.folder / "threedbag_sampled_filtered.copc.laz"
    if not threedbag_filtered_path.exists():
        get_sampled_threedbag(
            aoi=aoi,
            obj_path=project_paths.folder / "threedbag.obj",
            sampled_path=project_paths.folder / "threedbag_sampled.copc.laz",
            filtered_path=threedbag_filtered_path,
        )

    # Generating 3DBAG ground truth
    run_cfg_truth = RunConfig(
        name="truth_run",
        z_height=z_height,
        resolution=resolution,
        target_source=target_path,
        los_radius=0.1,  # Use a very small radius to best approximate true LoS to the sampled points
    )
    truth_paths = RunPaths(project_paths, run_cfg_truth.name)
    truth_paths.folder.mkdir(parents=True, exist_ok=True)

    if not truth_paths.output_viewshed_voxel_grid_3d.exists():
        # Store original paths before overriding to compute truth
        original_facades = project_paths.facades_copc
        project_paths.facades_copc = threedbag_filtered_path

        calculate_3d_viewshed(project_config, project_paths, run_cfg_truth, project_config.profile)
        save_viewshed_as_voxel_grid(truth_paths, run_cfg_truth, project_paths, project_config)

        # Restore original path so sweeps aren't affected
        project_paths.facades_copc = original_facades

    run_comprehensive_sweep(
        z_height=z_height,
        resolution=resolution,
        project_paths=project_paths,
        project_config=project_config,
        target_path=target_path,
        truth_voxel_path=truth_paths.output_viewshed_voxel_grid_3d,
    )
    plot_comprehensive_chart(project_paths, metric="F1_Score")
    plot_comprehensive_chart(project_paths, metric="Recall")
    plot_comprehensive_chart(project_paths, metric="Precision")


if __name__ == "__main__":
    main()
