import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
import seaborn as sns
from cloudfetch import AOIPolygon

import calculate
from main import calculate_3d_viewshed, prepare_project
from models import Point, ProjectConfig, ProjectPaths, RunConfig
from utils import get_logger

logger = get_logger("VegetationEvaluation")


def execute_viewshed_runs(
    project_cfg: ProjectConfig,
    project_paths: ProjectPaths,
    run_cfg_base: RunConfig,
) -> dict[str, Path]:
    modes = ["ignore", "binary", "probabilistic"]
    copc_paths = {}

    for mode in modes:
        print(f"\n--- Running calculation for mode: {mode} ---")
        run_cfg_base.name = f"eval_{mode}"
        run_cfg_base.vegetation_mode = mode

        # bit of a hacky solution, but it makes sure the log outpt path exists
        (project_paths.runs_folder / run_cfg_base.name).mkdir(parents=True, exist_ok=True)

        run_paths, _, _ = calculate_3d_viewshed(
            project_cfg=project_cfg,
            project_paths=project_paths,
            run_cfg=run_cfg_base,
        )

        copc_paths[mode] = run_paths.output_viewshed_copc_3d

    return copc_paths


def generate_comparison_csv(
    target_source_path: Path,
    copc_paths: dict[str, Path],
    output_csv: Path,
    bin_size_m=5,
) -> pd.DataFrame:

    results_dict = {}

    target_point = Point.get_from_file(target_source_path)
    observer_coords = target_point.array_coords

    for mode, copc_path in copc_paths.items():
        print(f"Processing COPC file for mode: {mode}...")

        # Load the relevant COPC file
        pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(copc_path)}]}))
        pipeline.execute()
        out_array = pipeline.arrays[0]

        # Extract and bin the data by distance
        flat_coords = np.column_stack((out_array["X"], out_array["Y"], out_array["Z"]))
        flat_probs = out_array["Visibility"]
        distances = np.linalg.norm(flat_coords - observer_coords, axis=1)  # Euclidean distance from target to each voxel
        bins = np.arange(0, distances.max() + bin_size_m, bin_size_m)
        distance_labels = bins[:-1]  # Left edge of the bin
        binned_distances = np.digitize(distances, bins) - 1
        df_temp = pd.DataFrame({"Distance_Bin": distance_labels[binned_distances], f"Vol_{mode.capitalize()}": flat_probs})
        binned_sums = df_temp.groupby("Distance_Bin").sum().reset_index()
        results_dict[mode] = binned_sums

    # Merge into one df
    final_df = results_dict["ignore"]
    final_df = final_df.merge(results_dict["binary"], on="Distance_Bin")
    final_df = final_df.merge(results_dict["probabilistic"], on="Distance_Bin")
    final_df.to_csv(output_csv, index=False)

    return final_df


def plot_thesis_comparisons(csv_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    plt.plot(df["Distance_Bin"], df["Vol_Ignore"], label="Ignore vegetation", linestyle="--", color="blue", linewidth=2)
    plt.plot(df["Distance_Bin"], df["Vol_Binary"], label="Opaque vegetation", linestyle="--", color="red", linewidth=2)
    plt.plot(df["Distance_Bin"], df["Vol_Probabilistic"], label="Probabilistic", linestyle="-", color="green", linewidth=3)

    plt.title("Visibility Volume Decay over Distance", fontsize=14)
    plt.xlabel("Distance from Observer (meters)", fontsize=12)
    plt.ylabel("Visible Airspace Volume (Voxel equivalents)", fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "plot_visibility_decay.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path


def augment_vegetation_density(
    input_path: Path,
    output_path: Path,
    increase_fraction: float = 0.3,
    jitter_std: float = 0.15,
    seed: int = 42,
) -> None:
    """Increase vegetation density by generating synthetic vegetation points and merging them into the original cloud."""

    # Read only vegetation points.
    veg_pipeline = pdal.Pipeline(
        json.dumps({
            "pipeline": [
                {"type": "readers.copc", "filename": str(input_path)},
                {"type": "filters.range", "limits": "Classification[5:5]"},
            ]
        })
    )
    veg_pipeline.execute()
    veg_points = veg_pipeline.arrays[0]

    num_new_points = int(veg_points.size * increase_fraction)
    logger.info(f"Original vegetation points: {veg_points.size}. Generating {num_new_points} synthetic points.")

    rng = np.random.default_rng(seed=seed)
    seed_indices = rng.choice(veg_points.size, size=num_new_points, replace=True)
    synthetic_points = np.copy(veg_points[seed_indices])

    synthetic_points["X"] += rng.normal(loc=0.0, scale=jitter_std, size=num_new_points)
    synthetic_points["Y"] += rng.normal(loc=0.0, scale=jitter_std, size=num_new_points)
    synthetic_points["Z"] += rng.normal(loc=0.0, scale=jitter_std, size=num_new_points)

    synthetic_points["Classification"] = 5
    synthetic_points["UserData"] = 1

    synthetic_tmp = output_path.parent / f"{output_path.name}.synthetic_tmp.copc.laz"
    try:
        write_synth_pipeline = {
            "pipeline": [
                {
                    "type": "writers.copc",
                    "filename": str(synthetic_tmp),
                    "forward": "all",
                    "extra_dims": "all",
                }
            ]
        }
        pdal.Pipeline(json.dumps(write_synth_pipeline), arrays=[synthetic_points]).execute()

        merge_pipeline = {
            "pipeline": [
                {"type": "readers.copc", "filename": str(input_path)},
                {"type": "readers.copc", "filename": str(synthetic_tmp)},
                {"type": "filters.merge"},
                {
                    "type": "writers.copc",
                    "filename": str(output_path),
                    "forward": "all",
                    "extra_dims": "all",
                },
            ]
        }
        pdal.Pipeline(json.dumps(merge_pipeline)).execute()
        logger.info(f"Saved augmented point cloud to: {output_path}")
    finally:
        if synthetic_tmp.exists():
            synthetic_tmp.unlink()


def execute_seasonal_runs(
    project_cfg: ProjectConfig,
    project_paths: ProjectPaths,
    run_cfg_base: RunConfig,
) -> dict[str, Path]:

    scenarios = {
        "baseline_winter": {"k": 0.05, "augment_fraction": 0.0},
        "simulated_summer_high_k": {"k": 0.075, "augment_fraction": 0.0},
        "simulated_summer_dense_cloud": {"k": 0.05, "augment_fraction": 0.50},  # +50% leaves
    }

    copc_paths = {}
    original_facades_path = project_paths.facades_copc
    backup_facades_path = project_paths.folder / "facades_original_backup.copc.laz"

    for mode, params in scenarios.items():
        print(f"\n--- Running Seasonal Scenario: {mode} ---")
        run_cfg_base.name = f"eval_{mode}"

        # Not-so-nice changing if the constant is needed
        original_k = calculate.BEER_LAMBERT_COEFFICIENT
        calculate.BEER_LAMBERT_COEFFICIENT = params["k"]
        print(f"Set Beer-Lambert k to: {params['k']}")

        # Increasing density
        if params["augment_fraction"] > 0:
            print(f"Augmenting point cloud density by {params['augment_fraction'] * 100}%...")
            shutil.copy2(original_facades_path, backup_facades_path)
            augment_vegetation_density(backup_facades_path, original_facades_path, increase_fraction=params["augment_fraction"])

        (project_paths.runs_folder / run_cfg_base.name).mkdir(parents=True, exist_ok=True)

        try:
            # Run Calculation
            run_paths, _, _ = calculate_3d_viewshed(
                project_cfg=project_cfg,
                project_paths=project_paths,
                run_cfg=run_cfg_base,
            )
            copc_paths[mode] = run_paths.output_viewshed_copc_3d

        finally:
            # Cleanup
            calculate.BEER_LAMBERT_COEFFICIENT = original_k
            if params["augment_fraction"] > 0 and backup_facades_path.exists():
                print("Restoring original point cloud...")
                shutil.move(backup_facades_path, original_facades_path)

    return copc_paths


def generate_seasonal_comparison_csv(
    target_source_path: Path,
    copc_paths: dict[str, Path],
    output_csv: Path,
    bin_size_m=5,
) -> pd.DataFrame:

    results_dict = {}
    target_point = Point.get_from_file(target_source_path)
    observer_coords = target_point.array_coords

    for mode, copc_path in copc_paths.items():
        print(f"Processing COPC file for mode: {mode}...")

        pipeline = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(copc_path)}]}))
        pipeline.execute()
        out_array = pipeline.arrays[0]

        flat_coords = np.column_stack((out_array["X"], out_array["Y"], out_array["Z"]))
        flat_probs = out_array["Visibility"]

        distances = np.linalg.norm(flat_coords - observer_coords, axis=1)

        bins = np.arange(0, distances.max() + bin_size_m, bin_size_m)
        distance_labels = bins[:-1]
        binned_distances = np.digitize(distances, bins) - 1

        df_temp = pd.DataFrame({
            "Distance_Bin": distance_labels[binned_distances],
            "Visibility": flat_probs,
        })

        binned_means = df_temp.groupby("Distance_Bin")["Visibility"].mean().reset_index()
        binned_means.rename(columns={"Visibility": f"VisRatio_{mode}"}, inplace=True)
        results_dict[mode] = binned_means

    # Merge DataFrames
    final_df = results_dict["baseline_winter"]
    final_df = final_df.merge(results_dict["simulated_summer_high_k"], on="Distance_Bin")
    final_df = final_df.merge(results_dict["simulated_summer_dense_cloud"], on="Distance_Bin")

    final_df.to_csv(output_csv, index=False)
    print(f"\nSaved seasonal comparison data to {output_csv}")
    return final_df


def plot_seasonal_comparisons(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))

    # Plot Baseline Winter
    plt.plot(df["Distance_Bin"], df["VisRatio_baseline_winter"], label="Baseline Winter (Sparse Cloud, k=0.05)", linestyle="-", color="blue", linewidth=2)

    # Plot Math Simulation
    plt.plot(df["Distance_Bin"], df["VisRatio_simulated_summer_high_k"], label="Math Summer (Sparse Cloud, k=0.075)", linestyle="--", color="orange", linewidth=2)

    # Plot Physical Simulation
    plt.plot(df["Distance_Bin"], df["VisRatio_simulated_summer_dense_cloud"], label="Physical Summer (+50% Density, k=0.05)", linestyle="--", color="purple", linewidth=2)

    plt.title("Impact of Seasonal Simulation Methods on Visibility", fontsize=14)
    plt.xlabel("Distance from Observer (meters)", fontsize=12)
    plt.ylabel("Visibility Ratio (0.0 = Blocked, 1.0 = Clear)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_seasonal_comparison.png", dpi=300)
    plt.close()
    print(f"Plot saved to {output_dir / 'plot_seasonal_comparison.png'}")


def project_setup(name: str, dataset: str) -> tuple[ProjectConfig, ProjectPaths, RunConfig, Path]:

    project_cfg = ProjectConfig(
        name=name,
        dataset=[dataset],
        classification_method="myria3d",
        myria3d_vegetation_prob_threshold_pct=60 if dataset == ["AHN6"] else 90,
        profile="testing",
    )
    project_paths = prepare_project(project_cfg, base_dir=Path("experiments"))

    if not project_paths.aoi.exists():
        print("Please draw an AOI that contains both buildings and vegetation.")
        aoi = AOIPolygon.get_from_user(title="Draw AOI for Single Comparison")
        aoi.save_to_file(project_paths.aoi)
    else:
        aoi = AOIPolygon.get_from_file(project_paths.aoi)

    aoi_rd = aoi.to_crs(project_cfg.crs)

    target_path = project_paths.folder / "target.copc.laz"
    if not target_path.exists():
        print("Please select the target (observer) point on the map.")
        Point.get(
            hag_sample_input_path=project_paths.input_copc,
            input_path=target_path,
            title="Select Observer Target",
            aoi=aoi_rd,
        )

    run_cfg = RunConfig(
        name="base_run",
        target_source=target_path,
        resolution=5.0,
        z_height=20.0,
        los_radius=0.15,
        los_step_length=0.15,
    )

    return project_cfg, project_paths, run_cfg, target_path


def compare_modes(project_cfg: ProjectConfig, project_paths: ProjectPaths, run_cfg: RunConfig, target_path: Path, bin_size: int):
    # Run the heavy calculations
    copc_paths = execute_viewshed_runs(project_cfg, project_paths, run_cfg)

    # Extract data, bin by distance, and generate CSV
    output_csv = project_paths.folder / "distance_comparison.csv"

    generate_comparison_csv(
        target_source_path=target_path,
        copc_paths=copc_paths,
        output_csv=output_csv,
        bin_size_m=bin_size,
    )
    # Generate Plots
    plot_thesis_comparisons(output_csv, project_paths.folder)


def compare_seasonal_methods(project_cfg: ProjectConfig, project_paths: ProjectPaths, run_cfg: RunConfig, target_path: Path):
    output_csv = project_paths.folder / "seasonal_comparison.csv"

    print("COPC files not found. Running seasonal 3D viewshed computations...")
    copc_paths = execute_seasonal_runs(project_cfg, project_paths, run_cfg)

    generate_seasonal_comparison_csv(
        target_source_path=target_path,
        copc_paths=copc_paths,
        output_csv=output_csv,
        bin_size_m=5,
    )

    plot_seasonal_comparisons(output_csv, project_paths.folder)


def main():

    name = "vegetation_influence_test"
    dataset = "AHN6"

    # Setup project
    project_cfg, project_paths, run_cfg, target_path = project_setup(name, dataset)

    # Compare the 3 vegetation handling modes (ignore, binary, probabilistic) with the same base configuration
    bin_size = 5
    # compare_modes(project_cfg, project_paths, run_cfg, target_path, bin_size=bin_size)

    # Compare seasonal simulation methods (math vs physical) to the baseline winter scenario
    compare_seasonal_methods(project_cfg, project_paths, run_cfg, target_path)

    # Evaluate K constant sensitivity
    # TODO: some plot that shows the influence of the k coefficient on visibility decay, maybe with a few different values?


if __name__ == "__main__":
    main()
