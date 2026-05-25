import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LogLocator

from calculate import export_grid_to_copc, get_distance_mask, load_points_for_runs, sample_ground
from main import prepare_project
from models import AOIPolygon, Cylinder, Point, ProjectConfig, Segment
from utils import get_logger

logger = get_logger("EvaluateQueryRadius")


def spatial_query_benchmark(segment: Segment, s: float, cylinder_radius: float, KDtree, array_coords):
    cylinder = Cylinder(
        segment=segment,
        min_radius=cylinder_radius,
        max_radius=cylinder_radius,
        step_length=s,
        radius_mode="fixed",
    )

    num_steps = max(1, int(np.ceil(segment.length / s)))

    t_chunk = np.linspace(0.0, 1.0, num_steps + 1)
    chunk_samples = segment.point1.array_coords + t_chunk[:, None] * segment.vector

    query_radius = np.sqrt(cylinder_radius**2 + (s**2 / 4))

    candidate_lists = KDtree.query_ball_point(chunk_samples, r=query_radius, workers=1)

    valid_candidates = [c for c in candidate_lists if c]
    if not valid_candidates:
        return

    flat_indices = np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in valid_candidates]))

    candidate_coords = array_coords[flat_indices]
    distance_mask, _ = get_distance_mask(candidate_coords, cylinder)

    _ = flat_indices[distance_mask]


def run_evaluation(export_ray_points: bool = False):

    # Setup project
    project_cfg = ProjectConfig(
        name="evaluate_query_radius",
        dataset=["AHN5"],
        crs="EPSG:28992",
        classification_method=None,
    )

    logger.info("Initializing project for benchmark...")
    project_paths = prepare_project(project_cfg, base_dir=Path("experiments"))
    input_path = project_paths.facades_copc
    aoi = AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs)

    # Set up benchmark LoS segments
    center_pt = aoi.polygon.centroid
    ground_z_array = sample_ground(input_path=project_paths.input_copc, points=[Point(center_pt.x, center_pt.y, 0.0)])
    test_z = float(ground_z_array[0]) + 1.8  # eye level
    origin = Point(center_pt.x, center_pt.y, test_z)

    all_LoS = {
        "Short (50m)": Segment(origin, Point(center_pt.x + 50, center_pt.y, test_z)),
        "Medium (200m)": Segment(origin, Point(center_pt.x + 200, center_pt.y, test_z)),
        "Long (500m)": Segment(origin, Point(center_pt.x + 500, center_pt.y, test_z)),
    }

    if export_ray_points:
        points_to_export = [origin] + [segment.point2 for segment in all_LoS.values()]
        export_path = project_paths.folder / "benchmark_test_points.copc.laz"
        logger.info(f"Exporting {len(points_to_export)} test points to {export_path}")
        export_grid_to_copc(points_to_export, export_path, project_cfg)

    logger.info("Loading KD-tree for evaluation region...")
    cylinder_radius = 0.15
    load_radius = 10.0  # Large enough to cover all test segments and their query radii
    _, array_coords, KDtree = load_points_for_runs(list(all_LoS.values()), load_radius, input_path)

    # Run benchmarks for different step sizes
    step_lengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    runs_per_test = 100
    results = []

    logger.info("Starting spatial benchmarking sequence...")

    for LoS_name, segment in all_LoS.items():
        logger.info(f"Testing {LoS_name} Ray")

        for s in step_lengths:
            spatial_query_benchmark(segment, s, cylinder_radius, KDtree, array_coords)

            for run_idx in range(runs_per_test):
                start = time.perf_counter()
                spatial_query_benchmark(segment, s, cylinder_radius, KDtree, array_coords)

                # Convert to microseconds directly
                run_time_ms = (time.perf_counter() - start) * 1000
                time_per_m_us = (run_time_ms / segment.length) * 1000

                results.append({
                    "ray_name": LoS_name,
                    "length_m": segment.length,
                    "step_size_m": s,
                    "run_idx": run_idx + 1,
                    "time_per_m_us": time_per_m_us,
                })

            logger.info(f"  Step Size (s)={s:<5.2f}m | Completed {runs_per_test} runs")

    # Save results to CSV
    output_csv = project_paths.folder / "benchmark.csv"
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ray_name", "length_m", "step_size_m", "run_idx", "time_per_m_us"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Benchmarking complete. Results saved to {output_csv}")


def plot_benchmark_results(csv_path: Path):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    line_plot = sns.lineplot(
        data=df,
        x="step_size_m",
        y="time_per_m_us",
        hue="ray_name",
        style="ray_name",
        palette="colorblind",
        markers=["o", "s", "D"],  # Circle, Square, Diamond
        dashes="",
        errorbar="sd",  # Standard deviation
        linewidth=2.5,
        markersize=8,
    )

    # Axes
    plt.xlabel("Distance Between Spheres ($s$) [m]", labelpad=10)
    line_plot.set_xscale("log")
    line_plot.set_xlim(0.1, 10.0)
    line_plot.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    line_plot.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=50))

    plt.ylabel(r"Execution Time per Meter of LoS [$\mu$s / m]", labelpad=10)
    line_plot.set_yscale("log")
    line_plot.set_ylim(1.0, 100.0)
    line_plot.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    line_plot.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=50))

    # Background grid
    formatter = FuncFormatter(lambda x, pos: f"{x:g}")  # strip trailing zeros

    line_plot.xaxis.set_major_formatter(formatter)
    line_plot.yaxis.set_major_formatter(formatter)
    line_plot.tick_params(axis="both", which="major", labelsize=11)
    line_plot.grid(True, which="major", color="gray", linewidth=1.0)

    line_plot.xaxis.set_minor_formatter(formatter)
    line_plot.yaxis.set_minor_formatter(formatter)
    line_plot.tick_params(axis="both", which="minor", labelsize=8)
    line_plot.grid(True, which="minor", color="lightgray", linewidth=0.6)

    # Legend
    plt.legend(title="Line of Sight (LoS) Length", frameon=True, alignment="left")

    # Save to svg
    plt.savefig(csv_path.parent / "query_for_different_step_sizes.svg", bbox_inches="tight")


if __name__ == "__main__":
    run_evaluation()
    plot_benchmark_results(Path("experiments/evaluate_query_radius/benchmark.csv"))
