import json
import sys
from importlib import import_module
from pathlib import Path

import pdal
from pointcloudlib import AHN5

from models import AOIPolygon
from utils import get_logger, timed

logger = get_logger(name="Classify")


@timed("Vegetation classification")
def classify_vegetation_rule_based(input_path: Path, output_path: Path):
    """
    Classifies vegetation in AHN data by targeting 'Other' points (Class 1) using Planarity.
    """

    # Define the PDAL pipeline
    pipeline_dict = {
        "pipeline": [
            # 1. Read the input COPC/LAZ file
            {"type": "readers.copc", "filename": str(input_path)},
            # 2. Calculate Height Above Ground using existing Class 2 (Ground)
            {"type": "filters.hag_nn"},
            # 3. Calculate planarity
            {"type": "filters.covariancefeatures", "knn": 15, "feature_set": ["Planarity"]},
            # 4. Assignment Logic:
            # - We only touch Classification 1 (Other)
            # - If NOT flat (Planarity < 0.4) and Height Above Ground > 0.5m → Class 5 (Vegetation)
            {
                "type": "filters.assign",
                "value": [
                    "Classification=5 WHERE (Classification==1 && HeightAboveGround > 0.5 && Planarity < 0.4)",
                ],
            },
            # 5. Write the result to a new COPC file
            {
                "type": "writers.copc",
                "filename": str(output_path),
            },
        ]
    }

    # Convert dict to JSON for PDAL
    pipeline_json = json.dumps(pipeline_dict)
    pipeline = pdal.Pipeline(pipeline_json)

    try:
        count = pipeline.execute()
        logger.info(f"Successfully processed {count} points.")
        logger.info(f"Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


MYRIA3D_ROOT = Path("third_party/myria3d")
MYRIA3D_CONFIG_DIR = MYRIA3D_ROOT / "configs"
MYRIA3D_CKPT = MYRIA3D_ROOT / "trained_model_assets" / "FRACTAL-LidarHD_7cl_randlanet.ckpt"
AHN_EPSG = 7415
MYRIA3D_BATCH_SIZE = 5
MYRIA3D_PROBAS = ["vegetation"]
AHN_OVERIG_CLASS = 1
AHN_VEGETATION_CLASS = 5


@timed("Vegetation classification with Myria3D")
def classify_vegetation_myria3d(input_path: Path, output_path: Path, vegetation_proba_threshold_pct: float):
    # Ensure vendored Myria3D package is importable from the workspace checkout.
    sys.path.insert(0, str(MYRIA3D_ROOT))

    # Use explicit syntax to stop VSCode from complaining.
    hydra = import_module("hydra")
    GlobalHydra = import_module("hydra.core.global_hydra").GlobalHydra
    HydraConfig = import_module("hydra.core.hydra_config").HydraConfig
    myria3d_predict = import_module("myria3d.predict").predict

    probas_override = f"[{','.join(MYRIA3D_PROBAS)}]"
    overrides = [
        "task.task_name=predict",
        f"predict.src_las={input_path}",
        f"predict.output_dir={output_path.parent}",
        f"predict.ckpt_path={MYRIA3D_CKPT.resolve()}",
        f"datamodule.epsg={AHN_EPSG}",
        f"datamodule.batch_size={MYRIA3D_BATCH_SIZE}",
        f"hydra.run.dir={output_path.parent}",
        "predict.interpolator.predicted_classification_channel=PredictedClassification",
        f"predict.interpolator.probas_to_save={probas_override}",
        "logger=csv",
    ]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(config_dir=str(MYRIA3D_CONFIG_DIR.resolve())):
        cfg = hydra.compose(config_name="config", overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    myria3d_predict(cfg)

    # Myria3D writes with pdal Writer.las, convert to COPC directly to output_path.
    pdal.Pipeline(
        json.dumps({
            "pipeline": [
                {"type": "readers.las", "filename": str(input_path)},
                {"type": "writers.copc", "filename": str(output_path)},
            ]
        })
    ).execute()

    apply_myria3d_vegetation_to_ahn_overig(
        input_path=output_path,
        output_path=output_path,
        overig_class=AHN_OVERIG_CLASS,
        vegetation_class=AHN_VEGETATION_CLASS,
        vegetation_proba_channel=MYRIA3D_PROBAS[0],
        vegetation_proba_threshold_pct=vegetation_proba_threshold_pct,
    )

    logger.info(f"Myria3D vegetation prediction saved to: {output_path}")
    return output_path


@timed("Apply Myria3D vegetation to AHN class")
def apply_myria3d_vegetation_to_ahn_overig(
    input_path: Path,
    output_path: Path,
    overig_class: int,
    vegetation_class: int,
    vegetation_proba_channel: str,
    vegetation_proba_threshold_pct: float,
):
    """Promote only AHN `overig` points to vegetation when vegetation proba exceeds threshold."""

    threshold = vegetation_proba_threshold_pct / 100.0

    assign_rule = f"Classification={vegetation_class} WHERE (Classification=={overig_class} && {vegetation_proba_channel}>{threshold})"
    pipeline_steps = [
        {"type": "readers.copc", "filename": str(input_path)},
        {"type": "filters.assign", "value": [assign_rule]},
        {"type": "writers.copc", "filename": str(output_path)},
    ]
    pdal.Pipeline(json.dumps({"pipeline": pipeline_steps})).execute()


@timed("Rescale AHN Colors")
def rescale_ahn_colors(input_path: Path, output_path: Path):
    """
    Rescales AHN 16-bit colors (0-65535) to 8-bit (0-255)
    to satisfy Myria3D normalization constraints.
    Preserves Infrared so Myria3D can compute NDVI from real NIR values.
    """

    # Probe available dimensions first to avoid assign failures on missing channels.
    probe = pdal.Pipeline(json.dumps({"pipeline": [{"type": "readers.copc", "filename": str(input_path)}]}))
    probe.execute()
    arr = probe.arrays[0]
    dims = set(arr.dtype.names)

    assign_values = []
    channels = ["Red", "Green", "Blue", "Infrared"]
    for channel in channels:
        if channel not in dims:
            logger.warning(f"Input has no {channel} channel. Myria3D will synthesize {channel}=0.")
            continue
        # Only rescale if channel is in 16-bit-like range; keep 8-bit inputs untouched.
        if float(arr[channel].max()) > 255.0:
            assign_values.append(f"{channel}={channel}/256")

    pipeline_steps = [{"type": "readers.copc", "filename": str(input_path)}]
    if assign_values:
        pipeline_steps.append({"type": "filters.assign", "value": assign_values})
    pipeline_steps.append({"type": "writers.copc", "filename": str(output_path)})
    pdal.Pipeline(json.dumps({"pipeline": pipeline_steps})).execute()
    logger.info(f"Rescaled colors saved to: {output_path}")


def run_test_myria3d_threshold_sweep(percentages: list, folder_name: str) -> None:
    run_folder = Path("data") / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)

    aoi_path = run_folder / "aoi.geojson"
    if not (run_folder / "aoi.geojson").exists():
        aoi = AOIPolygon.get_from_user(title=f"Draw AOI for {folder_name}")
        aoi.save_to_file(aoi_path, crs="EPSG:4326")
        logger.info(f"Saved AOI to: {aoi_path}")
    else:
        aoi = AOIPolygon.get_from_file(aoi_path)
        logger.info(f"Loaded existing AOI from: {aoi_path}")

    input_copc_path = run_folder / "input.copc.laz"
    rescaled_path = run_folder / "rescaled.copc.laz"

    aoi_rd = aoi.to_crs("EPSG:28992")
    ahn5 = AHN5(data_dir=run_folder)
    fetch_result = ahn5.fetch(aoi=aoi_rd.polygon, aoi_crs=aoi_rd.crs, output_path=input_copc_path)
    if not fetch_result or not fetch_result.exists():
        raise RuntimeError("Failed to fetch AHN5 data for selected AOI.")
    logger.info(f"Downloaded AHN5 data to: {input_copc_path}")

    rescale_ahn_colors(input_path=input_copc_path, output_path=rescaled_path)

    for threshold in percentages:
        threshold_label = int(threshold)
        threshold_input_path = run_folder / f"rescaled_threshold_{threshold_label}.copc.laz"
        output_classified_path = run_folder / f"classified_threshold_{threshold_label}.copc.laz"

        # Myria3D saves using the input basename in output_dir; isolate each sweep run
        # so the shared rescaled source is never overwritten across thresholds.
        pdal.Pipeline(
            json.dumps({
                "pipeline": [
                    {"type": "readers.copc", "filename": str(rescaled_path)},
                    {"type": "writers.copc", "filename": str(threshold_input_path)},
                ]
            })
        ).execute()

        classify_vegetation_myria3d(
            input_path=threshold_input_path,
            output_path=output_classified_path,
            vegetation_proba_threshold_pct=threshold,
        )
        logger.info(f"Saved classified output for {threshold_label}% threshold to: {output_classified_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        logger.info("No CLI arguments provided. Running Myria3D threshold sweep workflow.")
        percentages = [50.0, 70.0, 90.0, 99.0]
        folder_name = "test_myria3d"
        run_test_myria3d_threshold_sweep(percentages=percentages, folder_name=folder_name)
    else:
        if len(sys.argv) < 3:
            raise SystemExit("Usage: python src/segment.py <run_name> <myria3d|rule-based> [threshold_pct]")

        name = sys.argv[1]
        classification_method = sys.argv[2]  # Options: "myria3d", "rule-based"
        vegetation_proba_threshold_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
        logger.info(f"Starting vegetation classification in mode: {classification_method}")

        run_folder = Path("data") / name
        input_copc_path = run_folder / "input.copc.laz"
        rescaled_path = run_folder / "rescaled.copc.laz"
        output_classified_path = run_folder / "classified.copc.laz"

        if classification_method == "myria3d":
            rescale_ahn_colors(input_path=input_copc_path, output_path=rescaled_path)
            classify_vegetation_myria3d(input_path=rescaled_path, output_path=output_classified_path, vegetation_proba_threshold_pct=vegetation_proba_threshold_pct)
        else:
            classify_vegetation_rule_based(input_copc_path, output_classified_path)
