from pathlib import Path

from calculate import Point, Segment, calculate_point_to_point, calculate_viewshed_2d, export_grid_to_copc, hag_to_nap
from enhance_facades import generate_facades
from query_copc import Polygon, get_pointcloud_aoi
from segment import classify_vegetation_rule_based as classify_vegetation
from utils import get_logger, timed

logger = get_logger(name="Main")


@timed("Total processing")
def main():
    # Query the relevant points
    # aoi = Polygon.get_from_user("Select area of interest")
    aoi = Polygon(
        [
            (233691.30497727558, 581987.2869825428),
            (233875.81124650215, 582056.8082196123),
            (233921.904485486, 581956.0270049961),
            (233758.77601513162, 581894.0032933606),
            (233691.30497727558, 581987.2869825428),
        ]
    )
    output_copc_path = Path("data/groningen_plein_AHN4.copc.laz")
    dataset = "AHN4"
    get_pointcloud_aoi(aoi, include=[dataset], output_path=output_copc_path)

    # Classify vegetation
    output_classified_path = Path("data/groningen_plein_AHN4_classified.copc.laz")
    classify_vegetation(output_copc_path, output_classified_path)

    # Generate building facades from roof edges
    output_facades_path = Path("data/groningen_plein_AHN4_facades.copc.laz")
    generate_facades(output_classified_path, output_facades_path)

    # Generate 2D viewshed
    target_NAP = Point(233851.5, 581986.8, 1.7)
    target = hag_to_nap([target_NAP])[0]
    export_grid_to_copc([target], output_path=Path("data/target_point.copc.laz"))
    resolution = 1.0
    radius = 0.15
    output_path = Path("data/groningen_plein_AHN4_viewshed")

    calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        radius=radius,
        resolution=resolution,
        input_path=output_facades_path,
        output_path=output_path,
        z_offset=0.3,
    )

    # Calculate line of sight
    # p1 = Point(233974.5, 582114.2, 8.0)
    # p2 = Point(233912.2, 582187.5, 10.0)
    # pair = Segment.get_from_user("Select points for intervisibility")
    # radius = 3.0
    # visibility = calculate_point_to_point(pair, radius)
    # print(f"Calculated visibility: {visibility:.4f}")


if __name__ == "__main__":
    main()
