import pdal
import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon as ShapelyPolygon
import urllib.request
from urllib.parse import urlparse
from utils import timed, get_logger
from gui import make_map, _TO_RD

logger = get_logger(name="Query")


class Polygon(ShapelyPolygon):
    """Shapely Polygon subclass with a GUI constructor."""

    @classmethod
    def get_from_user(cls, title: str = "Draw polygon") -> "Polygon":
        """Let the user draw a polygon on the map."""
        import tkinter as tk

        root, map_widget, controls = make_map(title)

        points_latlon: list[tuple[float, float]] = []
        polygon = {"obj": None}
        marker_list: list = []

        def redraw():
            if polygon["obj"] is not None:
                polygon["obj"].delete()
            for m in marker_list:
                m.delete()
            marker_list.clear()

            for pt in points_latlon:
                marker_list.append(map_widget.set_marker(*pt))
            if len(points_latlon) == 2:
                polygon["obj"] = map_widget.set_path(points_latlon)
            elif len(points_latlon) >= 3:
                polygon["obj"] = map_widget.set_polygon(points_latlon)

        def on_click(coords):
            points_latlon.append((float(coords[0]), float(coords[1])))
            redraw()

        def clear():
            points_latlon.clear()
            redraw()

        tk.Button(controls, text="Clear", command=clear).pack(fill=tk.X)
        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(8, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()
        root.destroy()

        return cls([_TO_RD.transform(lon, lat) for lat, lon in points_latlon])


# 0. setup - ensure output directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# 1. Define arbitrary WKT Polygon
# The field of autzen stadium
# wkt_string_autzen = "POLYGON((637185.1 851912.5, 637239.4 852087.0, 637575.3 851980.0, 637526.4 851806.0, 637185.1 851912.5))"
# wkt_string_AHN6 = "POLYGON((234305 582712, 234759 582712, 234759 582300, 234305 582300, 234305 582712))"


# 2. Path to COPC file (remote URL or local path)
# remote_url_autzen = "https://github.com/PDAL/data/raw/refs/heads/main/autzen/autzen-classified.copc.laz?download="
# local_file_AHN6 = r"C:\Users\yairr\Downloads\test\hwh-ahn\AHN6\01_LAZ\AHN6_2025_C_233000_582000.COPC.LAZ"
# remote_url_AHN6 = "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ/AHN6_2025_C_233000_582000.COPC.LAZ"


def download_ahn_index(index_url, target_dir: Path = DATA_DIR):
    """
    Downloads an index file from index_url if it doesn't already exist.
    """
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(index_url).path).name
    local_path = Path(target_dir) / filename

    if local_path.exists():
        logger.info(f"Using existing local index: {local_path}")
    else:
        logger.info(f"Downloading AHN index from {index_url}...")
        try:
            # Set User-Agent to mimic a browser and bypass 403 restrictions
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
            urllib.request.install_opener(opener)

            urllib.request.urlretrieve(index_url, local_path)
            logger.info(f"Download complete: {local_path}")
        except Exception as e:
            logger.error(f"Failed to download index: {e}")
            raise

    return local_path


def query_tiles_2d(tile_urls, wkt_polygon):
    # Create a reader for every tile URL
    pipeline_def = [{"type": "readers.copc", "filename": url, "requests": 16} for url in tile_urls]

    # Add Merge, Crop, and Writer to the list
    pipeline_def.extend([{"type": "filters.merge"}, {"type": "filters.crop", "polygon": wkt_polygon}, {"type": "writers.copc", "filename": str(DATA_DIR / "output_merged.copc.laz"), "forward": "all"}])

    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        count = pipeline.execute()
        logger.info(f"Successfully processed {count} points from {len(tile_urls)} tiles.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def find_tiles(gdf_polygon):
    """Find the relevant AHN6 tiles for a given polygon."""
    # Load AHN6 tile index
    index_url = "https://basisdata.nl/hwh-ahn/AUX/bladwijzer_AHN6.gpkg"
    local_index_path = download_ahn_index(index_url)
    index_gdf = gpd.read_file(local_index_path, layer="bladindeling")

    # Spatial join to find intersecting tiles
    intersecting_tiles = gpd.sjoin(index_gdf, gdf_polygon, how="inner", predicate="intersects")

    tile_urls = []
    # Base URL for AHN6 streaming
    base_url = "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ/"

    # Construct URLs using the 'left' and 'bottom' columns
    for _, row in intersecting_tiles.iterrows():
        # Using 'left' as X and 'bottom' as Y
        # We ensure they are strings with the leading zeros if necessary
        x = str(int(row["left"])).zfill(6)
        y = str(int(row["bottom"])).zfill(6)

        # Build the official AHN6 filename pattern
        # Note: 2025 is the standard AHN6 year, but may vary by tile
        filename = f"AHN6_2025_C_{x}_{y}.COPC.LAZ"
        tile_urls.append(base_url + filename)

    return tile_urls


@timed("COPC query")
def query_ahn_2d(polygon: Polygon | None = None, polygon_path: Path | None = None):
    if polygon is not None:
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:28992")
    else:
        gdf = gpd.read_file(polygon_path)

    wkt_polygon_AHN6 = gdf.geometry.iloc[0].wkt  # type:ignore

    remote_url_AHN6 = find_tiles(gdf)

    query_tiles_2d(tile_urls=remote_url_AHN6, wkt_polygon=wkt_polygon_AHN6)


if __name__ == "__main__":
    # polygon_path = Path("data/groningen_polygon.gpkg")
    # query_ahn_2d(polygon_path=polygon_path)

    aoi = Polygon.get_from_user("Select polygon AOI for AHN query")
    query_ahn_2d(polygon=aoi)
