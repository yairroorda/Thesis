import subprocess
from pathlib import Path

from calculate import Point, calculate_viewshed_2d, export_grid_to_copc, hag_to_nap
from enhance_facades import generate_facades
from query_copc import AOIPolygon, get_pointcloud_aoi
from segment import classify_vegetation_rule_based
from utils import get_logger, timed
from visualize import save_viewshed_as_tif

logger = get_logger(name="Main")


@timed("Total processing")
def main(name: str = "test", dataset: list[str] | None = None, classification_method: str = "myria3d", resolution: float = 0.5, radius: float = 0.15) -> None:

    # Query the relevant points
    # aoi = AOIPolygon.get_from_user("Select area of interest")
    aoi = AOIPolygon.get_from_file(Path("data/Groningen_plein.geojson"))

    # Query the relevant points
    output_copc_path = Path(f"data/{name}.copc.laz")
    get_pointcloud_aoi(aoi, aoi_crs="EPSG:28992", include=dataset, output_path=output_copc_path)

    # Classify vegetation
    output_classified_path = Path(f"data/{name}_classified.copc.laz")

    if classification_method == "myria3d":
        logger.info("Delegating vegetation classification to Myria3D Pixi environment")
        result = subprocess.run(f"pixi run -e myria3d python src/segment.py {name} {classification_method}", shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Myria3D classification failed with exit code {result.returncode}.")

    elif classification_method == "rule-based":
        classify_vegetation_rule_based(output_copc_path, output_classified_path)

    else:
        raise ValueError(f"Unknown classification method: {classification_method}")

    # Generate building facades from roof edges
    output_facades_path = Path(f"data/{name}_facades.copc.laz")
    generate_facades(output_classified_path, output_facades_path)

    # Generate 2D viewshed
    output_facades_path = Path(f"data/{name}_facades.copc.laz")
    target_hag = Point(233851.5, 581986.8, 1.7)
    target = hag_to_nap([target_hag])[0]
    # target = Point.get_from_user("Select target point for viewshed analysis")
    export_grid_to_copc([target], output_path=Path(f"data/{name}_target_point.copc.laz"))

    output_path = Path(f"data/{name}_viewshed")

    _, _, visibility_points = calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        radius=radius,
        resolution=resolution,
        input_path=output_facades_path,
        output_path=output_path,
        z_offset=0.3,
    )

    save_viewshed_as_tif(
        x_coords=visibility_points["X"],
        y_coords=visibility_points["Y"],
        visibility_values=visibility_points["Visibility"],
        aoi=aoi,
        resolution=resolution,
        output_path=output_path.with_suffix(".tif"),
    )


if __name__ == "__main__":
    NAME = "test_ahn4"
    DATASET = "AHN4"  # Options: None (defaults to newest), or list of dataset names (e.g. ["AHN6", "AHN5"])
    CLASSIFICATION_METHOD = "myria3d"  # Options: "myria3d", "rule-based"
    RESOLUTION = 0.5
    RADIUS = 0.15

    main(
        name=NAME,
        dataset=DATASET,
        classification_method=CLASSIFICATION_METHOD,
        resolution=RESOLUTION,
        radius=RADIUS,
    )
