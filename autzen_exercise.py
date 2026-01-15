import pdal
import json

# 1. Define arbitrary WKT Polygon 
# The field of autzen stadium
wkt_polygon = "POLYGON((637185.1 851912.5, 637239.4 852087.0, 637575.3 851980.0, 637526.4 851806.0, 637185.1 851912.5))"

# 2. Path to COPC file (remote URL or local path)
remote_url = "data/autzen-classified.copc.laz"

# 3. Construct the PDAL Pipeline
pipeline_def = [
    {
        "type": "readers.copc",
        "filename": remote_url,
        "requests": 4  # Number of concurrent requests for speed
    },
    {
        "type": "filters.crop",
        "polygon": wkt_polygon
    },
    {
        "type": "writers.las",
        "filename": "data/output_crop.laz",
        "forward": "all" # Keep original metadata/CRS
    }
]

# 4. Execute the Pipeline
try:
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    count = pipeline.execute()
    
    print(f"Successfully processed {count} points.")
    print("Output saved to: output_crop.laz")
    
except Exception as e:
    print(f"An error occurred: {e}")