import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pdal
from cloudfetch import AOIPolygon

import calculate as calc
from main import calculate_3d_viewshed, prepare_project
from models import ProjectConfig, ProjectPaths, RunConfig
from utils import center_target_point_hag, generate_benchmark_aois
from visualize import save_viewshed_as_voxel_grid


def evaluate_vegetation_influence(
    config: ProjectConfig,
    num_aois: int,
    output_dir: Path,
    aoi_generator: Iterator[AOIPolygon] | None = None,
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)

    #
    aoi_features: list[dict] = []
    for index in range(1, num_aois + 1):
        aoi = next(aoi_generator)
        run_dir = output_dir / f"run_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        aoi.save_to_file(run_dir / "aoi.geojson")
        aoi_features.append({"type": "Feature", "properties": {"run_index": index}, "geometry": aoi.polygon.__geo_interface__})

        temp_config = ProjectConfig(
            name=f"vegetation_influence/temp_run_{index:02d}_{int(time.time())}",
            dataset=config.dataset,
            classification_method=config.classification_method,
            myria3d_vegetation_prob_threshold_pct=config.myria3d_vegetation_prob_threshold_pct,
            profile=config.profile,
            aoi_source=run_dir / "aoi.geojson",
            overwrite=False,
        )

        project_paths = ProjectPaths(temp_config.name)
        project_paths.folder.mkdir(parents=True, exist_ok=True)
        aoi.save_to_file(project_paths.aoi)

        project_paths = prepare_project(temp_config)
        pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(project_paths.facades_copc)}]}))
        pipeline.execute()

        vegetation_count = 0
        point_count = 0
        for chunk in pipeline.arrays:
            if "Classification" not in chunk.dtype.names:
                continue
            classifications = chunk["Classification"]
            point_count += classifications.size
            vegetation_count += int(np.count_nonzero(classifications == calc.CLASS_VEGETATION))

        input_vegetation_percentage = float(100.0 * vegetation_count / point_count) if point_count else 0.0

        target = center_target_point_hag(aoi, input_path=project_paths.facades_copc, hag_m=1.7)
        target.save_to_file(run_dir / "target.copc.laz")

        # FIX: Explicitly create the 'with' run folder before calling the calculation
        with_cfg = RunConfig(name=f"run_{index:02d}_with", target_source=run_dir / "target.copc.laz")
        (project_paths.runs_folder / with_cfg.name).mkdir(parents=True, exist_ok=True)

        with_run, with_vis, _ = calculate_3d_viewshed(project_cfg=temp_config, project_paths=project_paths, run_cfg=with_cfg, profile=config.profile)
        save_viewshed_as_voxel_grid(with_run, with_cfg, project_paths)
        with_voxel = run_dir / "3d_viewshed_voxel_grid_including_vegetation.copc.laz"
        shutil.copy2(with_run.output_viewshed_voxel_grid_3d, with_voxel)

        original_threshold = calc.VEGETATION_DENSITY_THRESHOLD
        calc.VEGETATION_DENSITY_THRESHOLD = float("inf")
        try:
            # FIX: Explicitly create the 'without' run folder before calling the calculation
            without_cfg = RunConfig(name=f"run_{index:02d}_without", target_source=run_dir / "target.copc.laz")
            (project_paths.runs_folder / without_cfg.name).mkdir(parents=True, exist_ok=True)

            without_run, without_vis, _ = calculate_3d_viewshed(project_cfg=temp_config, project_paths=project_paths, run_cfg=without_cfg, profile=config.profile)
            save_viewshed_as_voxel_grid(without_run, without_cfg, project_paths)
        finally:
            calc.VEGETATION_DENSITY_THRESHOLD = original_threshold

        without_voxel = run_dir / "3d_viewshed_voxel_grid_excluding_vegetation.copc.laz"
        shutil.copy2(without_run.output_viewshed_voxel_grid_3d, without_voxel)

        delta = np.abs(with_vis - without_vis)
        affected = delta > 0
        visible = with_vis > 0
        visible_affected = affected & visible

        run_metrics = {
            "voxel_count": int(delta.size),
            "affected_voxel_percentage": float(100.0 * affected.mean()) if delta.size else 0.0,
            "visible_voxel_percentage_affected": float(100.0 * visible_affected.sum() / visible.sum()) if visible.any() else 0.0,
            "mean_absolute_difference": float(delta.mean()) if delta.size else 0.0,
            "std_absolute_difference": float(delta.std()) if delta.size else 0.0,
            "mean_effected_difference": float(delta[affected].mean()) if affected.any() else 0.0,
            "std_effected_difference": float(delta[affected].std()) if affected.any() else 0.0,
        }

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

    return output_dir


def main():
    name = "vegetation_influence"
    output_dir = Path(f"data/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_path = output_dir / "sampling_aoi.geojson"

    if not aoi_path.exists():
        sampling_aoi = AOIPolygon.get_from_user()
        sampling_aoi.save_to_file(aoi_path)
    else:
        sampling_aoi = AOIPolygon.get_from_file(aoi_path)

    evaluate_vegetation_influence(
        ProjectConfig(dataset=["AHN6", "AHN5"], name=name, aoi_source=aoi_path),
        num_aois=3,
        output_dir=output_dir,
        aoi_generator=generate_benchmark_aois(size=100, area=sampling_aoi, seed=42),
    )


if __name__ == "__main__":
    main()
