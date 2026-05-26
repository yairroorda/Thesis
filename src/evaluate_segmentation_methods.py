import itertools
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
import seaborn as sns
from cloudfetch import AHN5
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from models import AOIPolygon
from segment import rescale_ahn_colors
from utils import get_logger

logger = get_logger("EvaluateSegmentation")


def create_channel_variants(base_path: Path, run_folder: Path) -> dict[str, Path]:
    """Generates the rescaled input and the channel-stripped versions for evaluation."""
    logger.info("Creating channel variants for Myria3D evaluation...")

    path_all = run_folder / "rescaled_all.copc.laz"
    rescale_ahn_colors(base_path, path_all)

    path_no_ir = run_folder / "rescaled_no_ir.copc.laz"
    pdal.Pipeline(
        json.dumps({
            "pipeline": [
                {"type": "readers.copc", "filename": str(path_all)},
                {"type": "filters.assign", "value": ["Infrared=0"]},
                {"type": "writers.copc", "filename": str(path_no_ir)},
            ]
        })
    ).execute()

    path_no_color = run_folder / "rescaled_no_color.copc.laz"
    pdal.Pipeline(
        json.dumps({
            "pipeline": [
                {"type": "readers.copc", "filename": str(path_all)},
                {"type": "filters.assign", "value": ["Red=0", "Green=0", "Blue=0", "Infrared=0"]},
                {"type": "writers.copc", "filename": str(path_no_color)},
            ]
        })
    ).execute()

    return {"Myria3D (All Channels)": path_all, "Myria3D (No IR)": path_no_ir, "Myria3D (No Color/IR)": path_no_color}


def generate_rule_features(input_path: Path, output_path: Path) -> Path:
    logger.info("Computing Planarity, HAG, and NDVI for rule-based sweep...")
    pipeline = pdal.Pipeline(
        json.dumps({
            "pipeline": [
                {"type": "readers.copc", "filename": str(input_path)},
                {"type": "filters.hag_nn"},
                {"type": "filters.covariancefeatures", "knn": 15, "feature_set": ["Planarity"]},
                {"type": "filters.assign", "value": ["NDVI = (Infrared - Red) / (Infrared + Red + 0.001)"]},
                {"type": "writers.copc", "filename": str(output_path), "extra_dims": "all"},
            ]
        })
    )
    pipeline.execute()
    return output_path


def load_copc_to_numpy(filepath: Path) -> np.ndarray:
    pipeline = pdal.Pipeline(
        json.dumps({
            "pipeline": [
                {
                    "type": "readers.copc",
                    "filename": str(filepath),
                },
            ]
        })
    )
    pipeline.execute()
    return pipeline.arrays[0]


def run_myria_sweep(run_folder: Path) -> Path:
    run_folder.mkdir(parents=True, exist_ok=True)

    # Setup test data
    aoi_path = run_folder / "aoi.geojson"
    aoi = AOIPolygon.get(input_path=aoi_path, title="Draw test area for evaluation", overwrite=False)
    aoi_rd = aoi.to_crs("EPSG:28992")

    input_copc = run_folder / "input.copc.laz"
    if not input_copc.exists():
        logger.info("Fetching AHN5 data for test area...")
        ahn5 = AHN5(data_dir=run_folder)
        ahn5.fetch(aoi=aoi_rd.polygon, aoi_crs=aoi_rd.crs, output_path=input_copc)

    # Prepare Data Variants
    variants = create_channel_variants(input_copc, run_folder)

    myria3d_outputs = {}
    for name, path in variants.items():
        out_path = run_folder / f"{path.stem}_classified.copc.laz"
        if not out_path.exists():
            logger.info(f"Running Myria3D inference for {name} via subprocess...")
            subprocess.run(
                [
                    "pixi",
                    "run",
                    "-e",
                    "myria3d",
                    "python",
                    "-W",
                    "ignore",
                    "src/segment.py",
                    "--explicit",
                    "myria3d",
                    str(path),
                    str(out_path),
                    "99.0",
                ],
                check=True,
            )
        myria3d_outputs[name] = out_path

    rule_based_feat_path = run_folder / "rule_based_features.copc.laz"
    if not rule_based_feat_path.exists():
        generate_rule_features(input_copc, rule_based_feat_path)

    # Sweep Logic in Memory
    results = []
    logger.info("Sweeping Myria3D thresholds...")
    myria3d_thresholds = np.linspace(10, 95, 50)

    for method_name, out_path in myria3d_outputs.items():
        arr = load_copc_to_numpy(out_path)
        orig_class = arr["Classification"]
        veg_prob = arr["vegetation"]
        total_points = len(orig_class)

        known_fp_mask = np.isin(orig_class, [2, 6, 9])  # TODO check for all AHN classes that are not vegetation or "other/overig"

        for thresh in myria3d_thresholds:
            pred_mask = veg_prob >= (thresh / 100.0)
            pp = np.count_nonzero(pred_mask)
            fp = np.count_nonzero(pred_mask & known_fp_mask)

            pp_pct = (pp / total_points) * 100
            fp_pct = (fp / pp) * 100 if pp > 0 else 0.0

            results.append({"Method": method_name, "Threshold": thresh, "Vegetation_Coverage_Pct": pp_pct, "False_Discovery_Rate_Pct": fp_pct})

    # Save and Plot
    csv_path = run_folder / "myria_sweep.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    return df, run_folder


def extract_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the non-dominated Pareto front (Max Coverage, Min FDR)."""
    df_sorted = df.sort_values(
        by=["Vegetation_Coverage_Pct", "False_Discovery_Rate_Pct"],
        ascending=[False, True],
    )
    pareto_front = []
    min_fdr_seen = float("inf")

    for index, row in df_sorted.iterrows():
        if row["False_Discovery_Rate_Pct"] < min_fdr_seen:
            pareto_front.append(row)
            min_fdr_seen = row["False_Discovery_Rate_Pct"]

    df_pareto = pd.DataFrame(pareto_front).sort_values(by="Vegetation_Coverage_Pct")
    df_pareto["Method"] = "Rule-Based (Theoretical)"
    return df_pareto


def plot_proxy_roc(df_base: pd.DataFrame, df_rules: pd.DataFrame, output_dir: Path):
    logger.info("Plotting normalized proxy ROC curves (with Pareto Front)...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    df_pareto = extract_pareto_front(df_rules)
    df_plot = pd.concat([df_base, df_pareto]).sort_values(by=["Method", "Vegetation_Coverage_Pct"])

    line_plot = sns.lineplot(
        data=df_plot,
        x="Vegetation_Coverage_Pct",
        y="False_Discovery_Rate_Pct",
        hue="Method",
        style="Method",
        palette="colorblind",
        linewidth=2.5,
        markers=False,
        dashes=True,
    )

    plt.xlabel("Total Points Classified as Vegetation [%]", labelpad=10)
    plt.ylabel("False Discovery Rate [%]", labelpad=10)

    formatter = FuncFormatter(lambda x, pos: f"{x:g}%")
    line_plot.xaxis.set_major_formatter(formatter)
    line_plot.yaxis.set_major_formatter(formatter)
    plt.xlim(0, 100)
    plt.yscale("log")
    plt.ylim(0, 100)

    plt.legend(title="Classification Method", frameon=True)

    svg_path = output_dir / "proxy_roc_curve.svg"
    plt.savefig(svg_path, bbox_inches="tight")
    logger.info(f"Plot saved to {svg_path}")


def plot_myria_threshold_dynamics(df_myria: pd.DataFrame, output_dir: Path):
    logger.info("Plotting threshold dynamics for all methods (Annotated Parametric Curves)...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = df_myria["Method"].unique()
    palette = sns.color_palette("colorblind", len(methods))

    myria_key_thresholds = [10, 50, 70, 80, 90, 95]

    for idx, method in enumerate(methods):
        subset = df_myria[df_myria["Method"] == method].sort_values(by="Vegetation_Coverage_Pct")

        ax.plot(
            subset["Vegetation_Coverage_Pct"],
            subset["False_Discovery_Rate_Pct"],
            label=method,
            color=palette[idx],
            linewidth=2.5,
            alpha=1.0,
            markersize=4,
            marker=["o", "s", "D"][idx],
            markevery=10000,
        )

        for thresh in myria_key_thresholds:
            closest_idx = (subset["Threshold"] - thresh).abs().argmin()
            closest_row = subset.iloc[closest_idx]
            x_val = closest_row["Vegetation_Coverage_Pct"]
            y_val = closest_row["False_Discovery_Rate_Pct"]

            ax.plot(
                x_val,
                y_val,
                marker=["o", "s", "D"][idx],
                color=palette[idx],
                markersize=6,
            )

            if idx == 0:  # blue
                offset_x = 18
                offset_y = -3
            elif idx == 1:  # yellow
                offset_x = 0
                offset_y = 8
            elif idx == 2:  # green
                offset_x = -18
                offset_y = -3
            else:
                offset_x = 8
                offset_y = 8

            ax.annotate(
                f"{int(thresh)}%",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(offset_x, offset_y),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=palette[idx],
            )

    # Format Axes & Labels
    ax.set_xlabel("Total Points Classified as Vegetation [%]", labelpad=10)
    ax.set_ylabel("False Discovery Rate (FDR) [%]", labelpad=10)

    ax.set_yscale("log")
    ax.set_xlim(15)
    ax.set_ylim(None, 10)

    formatter = FuncFormatter(lambda x, pos: f"{x:g}%")
    ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    # ax.tick_params(axis="both", which="major", labelsize=11)
    # ax.grid(True, which="major", color="gray", linewidth=1.0)

    ax.legend(title="Classification Method", frameon=True)

    svg_path = output_dir / "myria_threshold_dynamics.svg"
    plt.savefig(svg_path, bbox_inches="tight")
    logger.info(f"Plot saved to {svg_path}")


def run_rule_sweep(rule_based_feat_path: Path) -> Path:
    logger.info("ExecutingParameter Sweep (Planarity, HAG, NDVI, Returns)...")
    rule_arr = load_copc_to_numpy(rule_based_feat_path)

    orig_class = rule_arr["Classification"]
    hag = rule_arr["HeightAboveGround"]
    planarity = rule_arr["Planarity"]
    ndvi = rule_arr["NDVI"]
    returns = rule_arr["NumberOfReturns"]

    total_points = len(orig_class)
    rule_known_fp_mask = np.isin(orig_class, [2, 6, 9])

    hag_thresholds = np.linspace(-0.1, 3.0, 5)
    planarity_thresholds = np.linspace(0.0, 1.0, 5)
    ndvi_thresholds = np.linspace(-1.0, 1.0, 5)
    return_thresholds = range(0, 10)

    results = []

    sweep_space = list(
        itertools.product(
            hag_thresholds,
            planarity_thresholds,
            ndvi_thresholds,
            return_thresholds,
        )
    )

    for min_hag, plan_thresh, min_ndvi, min_returns in tqdm(sweep_space, total=len(sweep_space)):
        pred_mask = (hag >= min_hag) & (planarity <= plan_thresh) & (ndvi >= min_ndvi) & (returns >= min_returns)

        pp = np.count_nonzero(pred_mask)
        fp = np.count_nonzero(pred_mask & rule_known_fp_mask)

        pp_pct = (pp / total_points) * 100
        fp_pct = (fp / pp) * 100 if pp > 0 else 0.0

        results.append({
            "Min_HAG": min_hag,
            "Max_Planarity": plan_thresh,
            "Min_NDVI": min_ndvi,
            "Min_Returns": min_returns,
            "Vegetation_Coverage_Pct": pp_pct,
            "False_Discovery_Rate_Pct": fp_pct,
        })

    output_csv = Path(rule_based_feat_path.parent / "rule_sweep.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    logger.info(f"sweep results saved to {output_csv}")

    return output_csv


def plot_static_pareto_front(df: pd.DataFrame, output_dir: Path):
    logger.info("Calculating and plotting Static Pareto Front...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Identify the Pareto Front
    df_sorted = df.sort_values(by=["Vegetation_Coverage_Pct", "False_Discovery_Rate_Pct"], ascending=[False, True])

    pareto_front = []
    min_fdr_seen = float("inf")

    for index, row in df_sorted.iterrows():
        if row["False_Discovery_Rate_Pct"] < min_fdr_seen:
            pareto_front.append(row)
            min_fdr_seen = row["False_Discovery_Rate_Pct"]

    df_pareto = pd.DataFrame(pareto_front).sort_values(by="Vegetation_Coverage_Pct")

    # Plot combinations and boundary
    ax.scatter(
        df["Vegetation_Coverage_Pct"],
        df["False_Discovery_Rate_Pct"],
        color="grey",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Suboptimal Parameter Combinations",
    )
    ax.plot(
        df_pareto["Vegetation_Coverage_Pct"],
        df_pareto["False_Discovery_Rate_Pct"],
        color="r",
        marker="o",
        markersize=8,
        linewidth=1.5,
        label="Pareto Optimal Front",
    )

    # Format Axes & Labels
    ax.set_xlabel("Total Points Classified as Vegetation [%]", labelpad=10)
    ax.set_ylabel("False Discovery Rate (FDR) [%]", labelpad=10)

    formatter = FuncFormatter(lambda x, pos: f"{x:g}%")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim(bottom=0)

    ax.legend(frameon=True)

    svg_path = output_dir / "rule_based_pareto_front.svg"
    plt.savefig(svg_path, bbox_inches="tight")
    logger.info(f"Plot saved to {svg_path}")


def plot_parameter_diagnostics(df: pd.DataFrame, output_dir: Path):
    logger.info("Plotting 2x2 Parameter Diagnostic Grid...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Calculate the Pareto Front
    df_sorted = df.sort_values(by=["Vegetation_Coverage_Pct", "False_Discovery_Rate_Pct"], ascending=[False, True])
    pareto_front = []
    min_fdr_seen = float("inf")
    for index, row in df_sorted.iterrows():
        if row["False_Discovery_Rate_Pct"] < min_fdr_seen:
            pareto_front.append(row)
            min_fdr_seen = row["False_Discovery_Rate_Pct"]

    df_pareto = pd.DataFrame(pareto_front).sort_values(by="Vegetation_Coverage_Pct")

    parameters = [
        ("Min_HAG", "Minimum Height Above Ground [m]", "viridis"),
        ("Max_Planarity", "Maximum Planarity Threshold", "magma"),
        ("Min_NDVI", "Minimum NDVI Threshold", "crest"),
        ("Min_Returns", "Minimum Number of Returns", "flare"),
    ]

    for i, (param_col, title, palette) in enumerate(parameters):
        ax = axes[i]
        sns.scatterplot(data=df, x="Vegetation_Coverage_Pct", y="False_Discovery_Rate_Pct", hue=param_col, palette=palette, ax=ax, alpha=0.7, edgecolor="none", s=40)

        ax.plot(
            df_pareto["Vegetation_Coverage_Pct"],
            df_pareto["False_Discovery_Rate_Pct"],
            color="#E63946",
            linewidth=2,
            zorder=5,
        )

        ax.set_title(f"Colored by {title}", fontweight="bold", pad=10)
        ax.set_xlabel("Total Vegetation Coverage [%]")
        ax.set_ylabel("False Discovery Rate (FDR) [%]")
        ax.set_ylim(bottom=0)

        formatter = FuncFormatter(lambda x, pos: f"{x:g}%")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.legend(title=param_col.replace("_", " "), fontsize=8, title_fontsize=9)

    plt.suptitle("Hyperparameter Edge Diagnostics\n(Red line indicates the global Pareto optimum)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    svg_path = output_dir / "rule_based_parameter_diagnostics.svg"
    plt.savefig(svg_path, bbox_inches="tight")
    logger.info(f"Plot saved to {svg_path}")


def export_inspection_files(run_dir: Path, df_base: pd.DataFrame, df_rules: pd.DataFrame):
    logger.info("Generating COPC files for visual inspection...")
    out_dir = run_dir / "inspection_files"
    out_dir.mkdir(exist_ok=True)

    targets = [10, 20, 25, 30, 33]
    df_pareto = extract_pareto_front(df_rules)

    file_map = {
        "Myria3D (All Channels)": run_dir / "rescaled_all.copc_classified.copc.laz",
        "Myria3D (No IR)": run_dir / "rescaled_no_ir.copc_classified.copc.laz",
        "Myria3D (No Color/IR)": run_dir / "rescaled_no_color.copc_classified.copc.laz",
        "Rule-Based (Theoretical)": run_dir / "rule_based_features.copc.laz",
    }

    name_map = {
        "Myria3D (All Channels)": "myria3d_all",
        "Myria3D (No IR)": "myria3d_no_ir",
        "Myria3D (No Color/IR)": "myria3d_no_color",
        "Rule-Based (Theoretical)": "ruletheoretical",
    }

    for target in targets:
        logger.info(f"\n--- Generating files for ~{target}% Coverage Target ---")

        for method, base_file in file_map.items():
            if not base_file.exists():
                logger.warning(f"Missing base file for {method}. Skipping...")
                continue

            df_search = df_pareto if "Pareto" in method else df_base[df_base["Method"] == method]
            if df_search.empty:
                continue

            closest_idx = (df_search["Vegetation_Coverage_Pct"] - target).abs().argmin()
            row = df_search.iloc[closest_idx]
            actual_cov = row["Vegetation_Coverage_Pct"]
            fdr = row["False_Discovery_Rate_Pct"]

            safe_name = name_map[method]
            out_file = out_dir / f"{safe_name}_target_{target}pct_actual_{actual_cov:.1f}pct.copc.laz"

            if method.startswith("Myria3D"):
                thresh = row["Threshold"] / 100.0
                condition = f"Classification==1 && vegetation >= {thresh}"
                logger.info(f"[{safe_name}] Target: {target}% -> Actual: {actual_cov:.1f}% (FDR: {fdr:.2f}%) | Prob >= {thresh:.2f}")
            else:
                min_hag = row["Min_HAG"]
                max_plan = row["Max_Planarity"]
                min_ndvi = row["Min_NDVI"]
                min_ret = row["Min_Returns"]
                condition = f"Classification==1 && HeightAboveGround > {min_hag} && Planarity < {max_plan} && NDVI >= {min_ndvi} && NumberOfReturns >= {min_ret}"
                logger.info(
                    f"[{safe_name}] Target: {target}% -> Actual: {actual_cov:.1f}% (FDR: {fdr:.2f}%) | HAG>{min_hag}, Plan<{max_plan}, NDVI>={min_ndvi}, Ret>={min_ret}",
                )

            pipeline = pdal.Pipeline(
                json.dumps({
                    "pipeline": [
                        {"type": "readers.copc", "filename": str(base_file)},
                        {"type": "filters.assign", "value": ["Classification=1 WHERE Classification==5"]},
                        {"type": "filters.assign", "value": [f"Classification=5 WHERE ({condition})"]},
                        {"type": "writers.copc", "filename": str(out_file)},
                    ]
                })
            )
            pipeline.execute()

    logger.info(f"\nAll inspection files successfully generated in: {out_dir}")


if __name__ == "__main__":
    run_dir = Path("experiments/evaluate_segmentation_bouwkunde")

    # Run sweeps
    # myria_sweep_path = run_myria_sweep(run_dir)
    # rule_sweep_path = run_rule_sweep(run_dir / "rule_based_features.copc.laz")

    # Generate Comparative Plots
    df_rules = pd.read_csv("experiments/evaluate_segmentation_bouwkunde/rule_sweep.csv")
    df_myria = pd.read_csv("experiments/evaluate_segmentation_bouwkunde/myria_sweep.csv")
    plot_proxy_roc(df_myria, df_rules, run_dir)
    plot_myria_threshold_dynamics(df_myria, run_dir)

    # Generate Rule-Specific Diagnostic Plots
    plot_static_pareto_front(df_rules, run_dir)
    plot_parameter_diagnostics(df_rules, run_dir)

    # Export COPC files for visual inspection in PDAL/LASview
    # export_inspection_files(run_dir, df_myria, df_rules)
