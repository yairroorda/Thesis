from pathlib import Path

from query_copc import query_ahn_2d as query_copc, Polygon
from segment import classify_vegetation_rule_based as classify_vegetation
from calculate import Point, Segment, calculate_point_to_point, calculate_viewshed
from utils import timed, get_logger

logger = get_logger(name="Main")


@timed("Total processing")
def main():
    # Query the relevant points

    # polygon_path = Path("data/groningen_polygon.gpkg")
    # query_copc(polygon_path=polygon_path)

    aoi = Polygon.get_from_user("Select area of interest")
    query_copc(polygon=aoi)

    # Classify vegetation
    input_copc_path = Path("data/output_merged.copc.laz")
    output_classified_path = Path("data/output_classified.copc.laz")
    classify_vegetation(input_copc_path, output_classified_path)

    # Calculate line of sight
    # p1 = Point(233974.5, 582114.2, 8.0)
    # p2 = Point(233912.2, 582187.5, 10.0)
    # pair = Segment.get_from_user("Select points for intervisibility")
    # radius = 3.0
    # visibility = calculate_point_to_point(pair, radius)
    # print(f"Calculated visibility: {visibility:.4f}")

    search_radius = 50.0
    thinning_factor = 1
    radius = 3.0

    source = Point.get_from_user("Select source point for viewshed")
    calculate_viewshed(source, search_radius, radius, thinning_factor=thinning_factor)


if __name__ == "__main__":
    main()
