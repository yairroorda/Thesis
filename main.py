from query_copc import query_ahn_2d as query_copc
from segment import classify_vegetation_rule_based as classify_vegetation
from calculate import calculate_line_of_sight as calculate_line_of_sight
from utils import timed, get_logger

logger = get_logger(name="Main")

@timed("Total processing")
def main():
    # Query the relevant points
    polygon_path = r"data/groningen_polygon.gpkg"
    query_copc(polygon_path=polygon_path)

    # Classify vegetation
    input_copc_path = r"data/output_merged.copc.laz"
    output_classified_path = r"data/output_classified.copc.laz"
    classify_vegetation(input_copc_path, output_classified_path)

    # Calculate line of sight
    p1 = (233974.5, 582114.2, 8.0)
    p2 = (233912.2, 582187.5, 10.0)
    radius = 3
    calculate_line_of_sight(p1, p2, radius)

if __name__ == "__main__":
    main()