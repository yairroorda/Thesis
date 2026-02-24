from pathlib import Path

from calculate import Point, Segment, calculate_point_to_point, calculate_viewshed_2d, export_grid_to_copc, hag_to_nap
from enhance_facades import generate_facades
from query_copc import Polygon
from query_copc import query_ahn_2d as query_copc
from segment import classify_vegetation_rule_based as classify_vegetation
<<<<<<< HEAD
from utils import get_logger, timed
=======
from calculate import Point, Segment, calculate_point_to_point, calculate_viewshed
from utils import timed, get_logger
>>>>>>> 88a92d3 (Trying to support AHN5 as a fallback)

logger = get_logger(name="Main")


@timed("Total processing")
def main():
    # Query the relevant points

    polygon_path = Path("data/groningen_polygon.gpkg")
    query_copc(polygon_path=polygon_path)

    aoi = Polygon.get_from_user("Select area of interest")
    query_copc(polygon=aoi)

    # Classify vegetation
    input_copc_path = Path("data/output_merged.copc.laz")
    output_classified_path = Path("data/output_classified.copc.laz")
    classify_vegetation(input_copc_path, output_classified_path)

    # Generate building facades from roof edges
    output_facades_path = Path("data/classified_with_facades.copc.laz")
    generate_facades(output_classified_path, output_facades_path)

    # Generate 2D viewshed
    target = Point.get_from_user("Select target point for viewshed calculation")
    resolution = 1.0
    radius = 0.5
    output_path = Path("data/viewshed_2d_output")
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
<<<<<<< HEAD
=======

    search_radius = 50.0
    thinning_factor = 1
    radius = 3.0

    source = Point.get_from_user("Select source point for viewshed")
    calculate_viewshed(source, search_radius, radius, thinning_factor=thinning_factor)
>>>>>>> 88a92d3 (Trying to support AHN5 as a fallback)


if __name__ == "__main__":
    main()
