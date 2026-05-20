import json
from pathlib import Path

import numpy as np
import pdal

from calculate import calculate_3d_viewshed, generate_grid
from main import prepare_project
from models import AOIPolygon, ProjectConfig, ProjectPaths, RunConfig, RunPaths
from query_threedbag import ThreeDBAG
from sample_threedbag import sample_on_mesh
from utils import generate_benchmark_aois, get_logger
from visualize import save_viewshed_as_voxel_grid

logger = get_logger(name="TuneBuilding")


def get_sampled_threedbag(aoi: AOIPolygon, obj_path: Path, sampled_path: Path, filtered_path: Path) -> Path:

    # get "perfect" point cloud by sampling the 3DBAG LoD22 mesh at a high density
    ThreeDBAG.fetch(aoi, output_path=obj_path)
    sample_on_mesh(
        input_path=obj_path,
        output_path=sampled_path,
        density=50.0,
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


def load_voxel_grid(path: Path) -> np.ndarray:
    """Helper to load a point cloud array using PDAL."""
    pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(path)}]}))
    pipeline.execute()
    return pipeline.arrays[0]


def compare_voxel_grid(path_my_method: Path, path_3dbag: Path, visibility_threshold: float = 0.0) -> dict[str, float]:
    """
    Compares two viewshed voxel grids and returns a confusion matrix.
    Assumes 3DBAG is the 'Ground Truth' and My Method is the 'Prediction'.
    """
    logger.info(f"Comparing {path_my_method.name} vs {path_3dbag.name}")

    # Load the data arrays
    arr_my = load_voxel_grid(path_my_method)
    arr_3dbag = load_voxel_grid(path_3dbag)

    # Safety check: Ensure both arrays are exactly the same size
    if len(arr_my) != len(arr_3dbag):
        raise ValueError(f"Grid size mismatch! My method: {len(arr_my)}, 3DBAG: {len(arr_3dbag)}.")

    # Sort both arrays geometrically (by Z, then Y, then X) to guarantee alignment
    order_my = np.lexsort((arr_my["X"], arr_my["Y"], arr_my["Z"]))
    arr_my = arr_my[order_my]

    order_3d = np.lexsort((arr_3dbag["X"], arr_3dbag["Y"], arr_3dbag["Z"]))
    arr_3dbag = arr_3dbag[order_3d]

    # Binarize visibility based on the threshold
    # Assuming any Visibility > 0 means the voxel is visible
    pred_visible = arr_my["Visibility"] > visibility_threshold
    truth_visible = arr_3dbag["Visibility"] > visibility_threshold

    # Calculate Confusion Matrix
    total = len(pred_visible)

    # True Positives: Both say it's visible
    tp = np.sum(pred_visible & truth_visible)

    # True Negatives: Both say it's blocked
    tn = np.sum((~pred_visible) & (~truth_visible))

    # False Positives: Prediction says visible, but Truth says blocked
    fp = np.sum(pred_visible & (~truth_visible))

    # False Negatives: Prediction says blocked, but Truth says visible
    fn = np.sum((~pred_visible) & truth_visible)

    # Convert to percentages
    metrics = {"total_voxels": int(total), "TP_pct": float(tp / total) * 100, "TN_pct": float(tn / total) * 100, "FP_pct": float(fp / total) * 100, "FN_pct": float(fn / total) * 100, "Accuracy_pct": float((tp + tn) / total) * 100}

    logger.info(f"Results: Accuracy: {metrics['Accuracy_pct']:.2f}% | TP: {metrics['TP_pct']:.2f}%, TN: {metrics['TN_pct']:.2f}%, FP: {metrics['FP_pct']:.2f}%, FN: {metrics['FN_pct']:.2f}%")

    return metrics


def main():

    base_dir = Path("experiments/tune_building_threshold")
    base_dir.mkdir(parents=True, exist_ok=True)

    aoi_samples = generate_benchmark_aois(
        size=100,  # size of AOI in meters
        area=AOIPolygon.get(input_path=base_dir / "sample_area.json", title="Sample Area"),
    )

    # Build up test set of prepared projects
    for idx in range(1):
        project_config = ProjectConfig(
            name=f"tune_building_threshold_{idx}",
            crs="EPSG:28992",
            dataset=["AHN5"],
            classification_method=None,
            overwrite=False,
        )
        project_paths = ProjectPaths(project_name=project_config.name, base_dir=base_dir)
        aoi = next(aoi_samples)
        AOIPolygon.save_to_file(aoi, project_paths.aoi)

        # Setup project paths and data for this AOI
        project_paths = prepare_project(project_config, aoi=aoi, base_dir=base_dir)
        just_buildings_path = project_paths.folder / "just_buildings.copc.laz"
        filter_buildings(input_path=project_paths.facades_copc, output_path=just_buildings_path)

        # Add 3DBAG "perfect" point cloud for this AOI
        get_sampled_threedbag(
            aoi=aoi,
            obj_path=project_paths.folder / "threedbag.obj",
            sampled_path=project_paths.folder / "threedbag_sampled.copc.laz",
            filtered_path=project_paths.folder / "threedbag_sampled_filtered.copc.laz",
        )

        # run analysis comparing just_buildings_path to threedbag_sampled_filtered.copc.laz
        # generate grid of target points
        targets = generate_grid(
            Area=aoi.to_crs("EPSG:28992"),
            resolution=10.0,
            z_height=1.8,
            hag_base=project_paths.input_copc,
        )

        # calculate viewsheds for each target point
        # calculate viewsheds for each target point
        for idx, target in enumerate(targets):
            # Normal method run
            run_config_my = RunConfig(
                name=f"run_{idx}_normal_method",
                overwrite=False,
                z_height=10.0,
            )
            run_paths_my = RunPaths(project_paths, run_config_my.name)
            run_paths_my.folder.mkdir(parents=True, exist_ok=True)

            # Save the grid target
            target.save_to_file(run_paths_my.target_point_copc)
            run_config_my.target_source = run_paths_my.target_point_copc

            project_paths.facades_copc = just_buildings_path
            # Execute calculate and voxelize
            calculate_3d_viewshed(
                project_cfg=project_config,
                project_paths=project_paths,
                run_cfg=run_config_my,
                profile=project_config.profile,
            )
            save_viewshed_as_voxel_grid(
                run_paths=run_paths_my,
                run_cfg=run_config_my,
                project_paths=project_paths,
                project_cfg=project_config,
            )

            # 3DBAG run
            run_config_3dbag = RunConfig(
                name=f"run_{idx}_3dbag",
                overwrite=False,
                z_height=10.0,
            )
            run_paths_3dbag = RunPaths(project_paths, run_config_3dbag.name)
            run_paths_3dbag.folder.mkdir(parents=True, exist_ok=True)

            # Save the same grid target to the new run folder
            target.save_to_file(run_paths_3dbag.target_point_copc)
            run_config_3dbag.target_source = run_paths_3dbag.target_point_copc

            # Swap facades to 3DBAG mesh
            project_paths.facades_copc = project_paths.folder / "threedbag_sampled_filtered.copc.laz"

            # Execute calculate and voxelize
            calculate_3d_viewshed(
                project_cfg=project_config,
                project_paths=project_paths,
                run_cfg=run_config_3dbag,
                profile=project_config.profile,
            )
            save_viewshed_as_voxel_grid(
                run_paths=run_paths_3dbag,
                run_cfg=run_config_3dbag,
                project_paths=project_paths,
                project_cfg=project_config,
            )

            # Now that both voxel grids are saved to disk, compare them
            metrics = compare_voxel_grid(
                path_my_method=run_paths_my.output_viewshed_voxel_grid_3d,
                path_3dbag=run_paths_3dbag.output_viewshed_voxel_grid_3d,
                visibility_threshold=0.0,  # Any voxel with > 0.0 visibility is considered 'Visible'
            )

            # Save metrics to a JSON file in the main project folder
            comparison_log = project_paths.folder / f"comparison_{idx}.json"
            with open(comparison_log, "w") as f:
                json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
