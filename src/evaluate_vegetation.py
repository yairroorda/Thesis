import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
import seaborn as sns
from cloudfetch import AOIPolygon

from main import calculate_3d_viewshed, prepare_project
from models import Point, ProjectConfig, ProjectPaths, RunConfig


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

        # Extract coordinates and probabilities
        flat_coords = np.column_stack((out_array["X"], out_array["Y"], out_array["Z"]))
        flat_probs = out_array["Visibility"]

        # Calculate Euclidean distance from observer for every voxel
        distances = np.linalg.norm(flat_coords - observer_coords, axis=1)

        # Create distance bins
        bins = np.arange(0, distances.max() + bin_size_m, bin_size_m)
        distance_labels = bins[:-1]  # Left edge of the bin
        binned_distances = np.digitize(distances, bins) - 1

        # Store in a temporary dataframe
        df_temp = pd.DataFrame({
            "Distance_Bin": distance_labels[binned_distances],
            f"Vol_{mode.capitalize()}": flat_probs,  # Using Vol_... for sum as you requested
        })

        # Group by the bin and sum the volumes
        binned_sums = df_temp.groupby("Distance_Bin").sum().reset_index()
        results_dict[mode] = binned_sums

    # Merge the three modes into one DataFrame
    final_df = results_dict["ignore"]
    final_df = final_df.merge(results_dict["binary"], on="Distance_Bin")
    final_df = final_df.merge(results_dict["probabilistic"], on="Distance_Bin")
    final_df.to_csv(output_csv, index=False)

    return final_df


def plot_thesis_comparisons(csv_path: Path, output_dir: Path) -> None:
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
    plt.savefig(output_dir / "plot_visibility_decay.png", dpi=300)
    plt.close()
    print(f"Plot saved to {output_dir / 'plot_visibility_decay.png'}")


def main():
    # Setup Project Configuration
    project_cfg = ProjectConfig(
        name="vegetation_influence_groningen",
        dataset=["AHN6", "AHN5"],
        classification_method="myria3d",
        myria3d_vegetation_prob_threshold_pct=70,
        profile="testing",
    )

    # Prepare the project
    project_paths = prepare_project(project_cfg, base_dir=Path("experiments"))

    # Ensure we have an AOI
    if not project_paths.aoi.exists():
        print("Please draw an AOI that contains both buildings and vegetation.")
        aoi = AOIPolygon.get_from_user(title="Draw AOI for Single Comparison")
        aoi.save_to_file(project_paths.aoi)
    else:
        aoi = AOIPolygon.get_from_file(project_paths.aoi)

    aoi_rd = aoi.to_crs(project_cfg.crs)

    # Ensure we have a target point
    target_path = project_paths.folder / "target.copc.laz"
    if not target_path.exists():
        print("Please select the target (observer) point on the map.")
        Point.get(
            hag_sample_input_path=project_paths.input_copc,
            input_path=target_path,
            title="Select Observer Target",
            aoi=aoi_rd,
        )

    # Base Run Configuration
    run_cfg = RunConfig(
        name="base_run",
        target_source=target_path,
        resolution=5.0,
        z_height=20.0,
        los_radius=0.15,
        los_step_length=0.15,
    )

    # Run the heavy calculations
    copc_paths = execute_viewshed_runs(project_cfg, project_paths, run_cfg)

    # Extract data, bin by distance, and generate CSV
    output_csv = project_paths.folder / "distance_comparison.csv"

    generate_comparison_csv(
        target_source_path=target_path,
        copc_paths=copc_paths,
        output_csv=output_csv,
        bin_size_m=5,
    )

    # Generate Plots
    plot_thesis_comparisons(output_csv, project_paths.folder)


if __name__ == "__main__":
    main()
