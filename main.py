from query_copc import query_ahn_2d as query_copc
from segment import classify_vegetation_rule_based as classify_vegetation
from utils import timed

@timed("Total processing")
def main():
    # Query the relevant points
    polygon_path = r"data/groningen_polygon.gpkg"
    query_copc(polygon_path=polygon_path)

    # Classify vegetation
    input_copc_path = r"data/output_merged.copc.laz"
    output_classified_path = r"data/output_classified.laz"
    classify_vegetation(input_copc_path, output_classified_path)

if __name__ == "__main__":
    main()