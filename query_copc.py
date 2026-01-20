import pdal
import json
import os
import geopandas as gpd

# 0. setup - ensure output directory exists
os.makedirs("data", exist_ok=True)

# 1. Define arbitrary WKT Polygon 
# The field of autzen stadium
wkt_string_autzen = "POLYGON((637185.1 851912.5, 637239.4 852087.0, 637575.3 851980.0, 637526.4 851806.0, 637185.1 851912.5))"
wkt_string_AHN6 = "POLYGON((234305 582712, 234759 582712, 234759 582300, 234305 582300, 234305 582712))"


# 2. Path to COPC file (remote URL or local path)
remote_url_autzen = "https://github.com/PDAL/data/raw/refs/heads/main/autzen/autzen-classified.copc.laz?download="
local_file_AHN6 = r"C:\Users\yairr\Downloads\test\hwh-ahn\AHN6\01_LAZ\AHN6_2025_C_233000_582000.COPC.LAZ"
remote_url_AHN6 = "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ/AHN6_2025_C_233000_582000.COPC.LAZ"

def main(file_path, wkt_polygon):
    
    # 3. Construct the PDAL Pipeline
    pipeline_def = [
        {
            "type": "readers.copc",
            "filename": file_path,
            "requests": 4  # Number of concurrent requests for speed
        },
        {
            "type": "filters.crop",
            "polygon": wkt_polygon
        },
        {
            "type": "writers.copc",
            "filename": "data/output_crop.copc.laz",
            "forward": "all" # Keep original metadata/CRS
        }
    ]

    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        count = pipeline.execute()
        
        print(f"Successfully processed {count} points.")
        print("Output saved to: output_crop.laz")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    gdf = gpd.read_file(r"C:\Users\yairr\OneDrive\Thesis\groningen_polygon.gpkg")
    wkt_polygon_AHN6 = gdf.geometry.iloc[0].wkt #type:ignore

    main(
        file_path=remote_url_AHN6,
        wkt_polygon=wkt_polygon_AHN6
    )

