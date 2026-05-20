import shutil
import subprocess
from pathlib import Path

from cloudfetch import AHN1, AHN2, AHN3, AHN4, AHN5, AHN6, CanElevation, IGNLidarHD, ProviderChain

from calculate import (
    calculate_2d_viewshed,
    calculate_3d_viewshed,
    calculate_cumulative_viewshed,
)
from enhance_facades import generate_facades
from flight_height import calculate_3d_flight_height
from lookout import calculate_optimal_lookout
from models import AOIPolygon, Point, ProjectConfig, ProjectPaths, RunConfig
from pathfinding import calculate_optimal_route
from segment import classify_vegetation_rule_based
from utils import get_logger, load_profile, timed
from visualize import save_viewshed_as_voxel_grid

logger = get_logger(name="Main")


@timed("Prepare project")
def prepare_project(config: ProjectConfig, aoi: AOIPolygon = None, base_dir: Path = None) -> ProjectPaths:
    paths = ProjectPaths(config.name, base_dir=base_dir)

    profile_cfg = load_profile(config.profile)
    log_level = profile_cfg.logging_level
    global logger
    logger = get_logger(name="Main", logfile_path=paths.project_log, level=log_level)

    logger.info(f"Using config profile: {config.profile}")
    logger.debug(f"Config: {profile_cfg.name}")
    logger.info(f"Project: {config.name}")
    logger.info(f"Project folder: {paths.folder}")

    if config.overwrite:
        for artifact in [paths.input_copc, paths.rescaled_copc, paths.classified_copc, paths.facades_copc]:
            if artifact.exists():
                artifact.unlink()

    aoi = aoi or AOIPolygon.get(
        input_path=paths.aoi,
        title=f"Draw AOI for project {config.name}",
        overwrite=config.overwrite,
    )
    aoi_rd = aoi.to_crs(config.crs)
    logger.info("Project AOI ready")

    if paths.is_prepared:
        logger.info(f"Project {config.name} is already prepared. Reusing cached preprocessing outputs.")
        return paths

    logger.info("Downloading project point cloud for AOI")
    dataset_map = {
        "IGNLidarHD": IGNLidarHD,
        "IGN_LIDAR_HD": IGNLidarHD,
        "AHN6": AHN6,
        "AHN5": AHN5,
        "AHN4": AHN4,
        "AHN3": AHN3,
        "AHN2": AHN2,
        "AHN1": AHN1,
        "CanElevation": CanElevation,
    }
    providers = [dataset_map[name](data_dir=paths.folder) for name in config.dataset]
    provider_chain = ProviderChain(providers)
    result = provider_chain.fetch(aoi=aoi_rd.polygon, aoi_crs=aoi_rd.crs, output_path=paths.input_copc)
    if not result or not result.exists():
        raise RuntimeError(f"Could not query requested data. Tried datasets in order: {config.dataset}")

    # Classify vegetation
    logger.info(f"Classifying vegetation using method: {config.classification_method}")
    if config.classification_method is None:
        logger.info("No classification method specified. Skipping vegetation classification.")
        paths.classified_copc = paths.input_copc
    elif config.classification_method == "myria3d":
        logger.debug(f"Applying Myria3D vegetation probability threshold: {config.myria3d_vegetation_prob_threshold_pct:.1f}%")
        result = subprocess.run(
            [
                "pixi",
                "run",
                "-e",
                "myria3d",
                "python",
                "src/segment.py",
                paths.name,
                config.classification_method,
                str(config.myria3d_vegetation_prob_threshold_pct),
                str(paths.folder),
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Myria3D classification failed with exit code {result.returncode}.")
    elif config.classification_method == "rule-based":
        classify_vegetation_rule_based(paths.input_copc, paths.classified_copc)
    else:
        raise ValueError(f"Unknown classification method: {config.classification_method}")

    logger.info("Generating facade-enhanced point cloud")
    generate_facades(paths.classified_copc, paths.facades_copc)
    if not paths.facades_copc.exists():
        logger.warning("Facade generation produced no output. Falling back to classified cloud.")
        shutil.copy2(paths.classified_copc, paths.facades_copc)

    logger.info("Project preprocessing complete")
    return paths


def main():

    # ---------- PROJECT ----------

    # Setup project configuration
    project_cfg = ProjectConfig(
        name="test_project_nl",
        dataset=["AHN5"],
        crs="EPSG:28992",
        classification_method="myria3d",
    )

    # Prepare project (download data, preprocess, etc.)
    project_paths = prepare_project(project_cfg)

    # ---------- RUN --------------

    run_name = "test_run"

    # Setup run configuration
    run_cfg = RunConfig(
        name=run_name,
        overwrite=False,
        resolution=2.0,
        los_mode="fixed",
        los_radius=0.15,
        los_step_length=0.15,
        z_height=30.0,
        log_level="INFO",
        target_source=None,  # Set later based on which analyses are run
    )

    # Set which analyses to run
    run_2d_viewshed = False
    run_3d_viewshed = False
    cumulative_viewshed = False  # Forces 3d viewshed as a voxel grid to keep runtime reasonable

    if run_2d_viewshed:
        run_cfg.target_source = project_paths.runs_folder / Path(run_name) / "target_point.copc.laz"

        logger.debug(f"Getting target point for 2D viewshed from: {run_cfg.target_source}")

        target = Point.get(
            hag_sample_input_path=project_paths.input_copc,
            input_path=run_cfg.target_source,
            title="Select target point for run",
            aoi=AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs),
        )

        logger.debug(f"Point XYZ: {target.x}, {target.y}, {target.z}")

        run_paths = calculate_2d_viewshed(
            project_cfg=project_cfg,
            project_paths=project_paths,
            run_cfg=run_cfg,
            profile=project_cfg.profile,
        )

    if run_3d_viewshed:
        run_cfg.target_source = project_paths.runs_folder / Path(run_name) / "target_point.copc.laz"
        Point.get(
            hag_sample_input_path=project_paths.input_copc,
            input_path=run_cfg.target_source,
            title="Select target point for run",
            aoi=AOIPolygon.get_from_file(project_paths.aoi).to_crs(project_cfg.crs),
        )
        run_paths, _, _ = calculate_3d_viewshed(
            project_cfg=project_cfg,
            project_paths=project_paths,
            run_cfg=run_cfg,
            profile=project_cfg.profile,
        )
        save_viewshed_as_voxel_grid(
            run_paths=run_paths,
            run_cfg=run_cfg,
            project_paths=project_paths,
            project_cfg=project_cfg,
        )

    if cumulative_viewshed:
        run_paths = calculate_cumulative_viewshed(
            number_of_targets=2,
            project_cfg=project_cfg,
            project_paths=project_paths,
            run_cfg=run_cfg,
            save_to_disk=False,
        )

    # ---------- ANALYSIS -------

    run_flight_height = False
    run_optimal_route = False
    run_optimal_lookout = False  # This will overwrite the target point and viewshed

    if run_flight_height:
        run_paths = calculate_3d_flight_height(
            project_cfg=project_cfg,
            project_paths=project_paths,
            run_cfg=run_cfg,
            threshold=0,  # a point is considered visible if visibility > threshold
        )

    if run_optimal_route:
        run_paths = calculate_optimal_route(
            project_cfg=project_cfg,
            project_paths=project_paths,
            run_cfg=run_cfg,
            log_level=run_cfg.log_level,
        )

    if run_optimal_lookout:
        run_paths, best_pt = calculate_optimal_lookout(
            project_cfg=project_cfg,
            project_paths=project_paths,
            run_cfg=run_cfg,
            profile=project_cfg.profile,
            coarse_res=10,
            fine_res=2.0,
            trail_sample_distance=3.0,
            threshold=0.8,  # a point is considered visible if visibility > threshold
        )

        # save best lookout point
        best_pt_path = run_paths.folder / "best_lookout_point.copc.laz"
        best_pt.save_to_file(best_pt_path, crs=project_cfg.crs)

        run_cfg.target_source = best_pt_path

        calculate_3d_viewshed(
            project_cfg=project_cfg,
            project_paths=project_paths,
            run_cfg=run_cfg,
            profile=project_cfg.profile,
        )


if __name__ == "__main__":
    main()
