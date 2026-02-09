import pdal
import json
import os
import geopandas as gpd
from utils import timed

# 0. setup - ensure output directory exists
os.makedirs("data", exist_ok=True)

# 1. Define arbitrary WKT Polygon 
# The field of autzen stadium
# wkt_string_autzen = "POLYGON((637185.1 851912.5, 637239.4 852087.0, 637575.3 851980.0, 637526.4 851806.0, 637185.1 851912.5))"
# wkt_string_AHN6 = "POLYGON((234305 582712, 234759 582712, 234759 582300, 234305 582300, 234305 582712))"


# 2. Path to COPC file (remote URL or local path)
# remote_url_autzen = "https://github.com/PDAL/data/raw/refs/heads/main/autzen/autzen-classified.copc.laz?download="
# local_file_AHN6 = r"C:\Users\yairr\Downloads\test\hwh-ahn\AHN6\01_LAZ\AHN6_2025_C_233000_582000.COPC.LAZ"
# remote_url_AHN6 = "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ/AHN6_2025_C_233000_582000.COPC.LAZ"

def query_tiles_2d(tile_urls, wkt_polygon):
    # Create a reader for every tile URL
    pipeline_def = [
        {"type": "readers.copc", "filename": url, "requests": 16} 
        for url in tile_urls
    ]

    # Add Merge, Crop, and Writer to the list
    pipeline_def.extend([
        {"type": "filters.merge"},
        {"type": "filters.crop", "polygon": wkt_polygon},
        {
            "type": "writers.copc",
            "filename": "data/output_merged.copc.laz",
            "forward": "all"
        }
    ])

    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        count = pipeline.execute()
        print(f"Successfully processed {count} points from {len(tile_urls)} tiles.")
    except Exception as e:
        print(f"An error occurred: {e}")

def find_tiles(gdf_polygon):
    '''Find the relevant AHN6 tiles for a given polygon.'''
    # Load AHN6 tile index
    index_url = "https://basisdata.nl/hwh-ahn/AUX/bladwijzer_AHN6.gpkg"
    index_gdf = gpd.read_file(index_url)

    # Spatial join to find intersecting tiles
    intersecting_tiles = gpd.sjoin(index_gdf, gdf_polygon, how="inner", predicate='intersects')

    tile_urls = []
    # Base URL for AHN6 streaming
    base_url = "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ/"

    # Construct URLs using the 'left' and 'bottom' columns
    for _, row in intersecting_tiles.iterrows():
        # Using 'left' as X and 'bottom' as Y
        # We ensure they are strings with the leading zeros if necessary
        x = str(int(row['left'])).zfill(6)
        y = str(int(row['bottom'])).zfill(6)
        
        # Build the official AHN6 filename pattern
        # Note: 2025 is the standard AHN6 year, but may vary by tile
        filename = f"AHN6_2025_C_{x}_{y}.COPC.LAZ"
        tile_urls.append(base_url + filename)

    return tile_urls

@timed("COPC query")
def query_ahn_2d(polygon_path):

    gdf = gpd.read_file(polygon_path)
    wkt_polygon_AHN6 = gdf.geometry.iloc[0].wkt #type:ignore

    remote_url_AHN6 = find_tiles(gdf)

    query_tiles_2d(tile_urls=remote_url_AHN6, wkt_polygon=wkt_polygon_AHN6)

if __name__ == "__main__":

    polygon_path = r"data/groningen_polygon.gpkg"

    query_ahn_2d(polygon_path=polygon_path)

