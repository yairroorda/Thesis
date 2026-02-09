import pdal
import json

from utils import timed, get_logger

logger = get_logger(name="Classify")

input_file_path = r"data/output_merged.copc.laz"
output_file_path = r"data/output_classified.copc.laz"

@timed("Vegetation classification")
def classify_vegetation_rule_based(input_path, output_path):
    """
    Classifies vegetation in AHN data by targeting 'Other' points (Class 1)
    using Height Above Ground (HAG) and Planarity.
    """
    
    # Define the PDAL pipeline
    pipeline_dict = {
        "pipeline": [
            # 1. Read the input COPC/LAZ file
            {
                "type": "readers.copc",
                "filename": input_path
            },
            
            # 2. Calculate Height Above Ground using existing Class 2 (Ground)
            {
                "type": "filters.hag_nn"
            },
            
            # 3. Calculate planarity
            {
                "type": "filters.covariancefeatures",
                "knn": 15,
                "feature_set": ["Planarity"]
            },
            
            # 4. Assignment Logic:
            # - We only touch Classification 1 (Other)
            # - If it's low (0.2m - 2m) and NOT flat (Planarity < 0.4) -> Class 3 (Low Veg)
            # - If it's high (> 2m) and NOT flat (Planarity < 0.4) -> Class 5 (High Veg)
            {
                "type": "filters.assign",
                "value": [
                    "Classification=3 WHERE (Classification==1 && HeightAboveGround > 0.2 && HeightAboveGround <= 2.0 && Planarity < 0.4)",
                    "Classification=5 WHERE (Classification==1 && HeightAboveGround > 2.0 && Planarity < 0.4)"
                ]
            },
            
            # 5. Write the result to a new COPC file
            {
                "type": "writers.copc",
                "filename": output_path,
                "extra_dims": "all" # Keeps HAG and Planarity dimensions for inspection
            }
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

if __name__ == "__main__":
    classify_vegetation_rule_based(input_file_path, output_file_path)