import importlib
import tkinter as tk
from pyproj import Transformer

tkintermapview = importlib.import_module("tkintermapview")
_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
_TO_LONLAT = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# Default centre (Groningen) in RD
_START_X, _START_Y = 233974.5, 582114.2
_START_LON, _START_LAT = _TO_LONLAT.transform(_START_X, _START_Y)


def make_map(title):
    """Create a tkinter window with a map widget and a side panel."""
    root = tk.Tk()
    root.title(title)
    root.geometry("1100x700")

    map_widget = tkintermapview.TkinterMapView(root, width=850, height=700, corner_radius=0)
    map_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    controls = tk.Frame(root, padx=10, pady=10)
    controls.pack(side=tk.RIGHT, fill=tk.Y)

    map_widget.set_position(float(_START_LAT), float(_START_LON))
    map_widget.set_zoom(13)

    return root, map_widget, controls


def main():
    from query_copc import Polygon
    from calculate import Segment, Point
    from utils import get_logger

    logger = get_logger(name="GUI Test")

    polygon = Polygon.get_from_user("Test polygon input")
    logger.info(f"Collected polygon: {polygon}")
    points = Segment.get_from_user("Test points input")
    logger.info(f"Collected points: SEGMENT(POINT({points.point1.x}, {points.point1.y}, {points.point1.z}), POINT({points.point2.x}, {points.point2.y}, {points.point2.z}))")
    point = Point.get_from_user("Test point input")
    logger.info(f"Collected point: POINT({point.x}, {point.y}, {point.z})")


if __name__ == "__main__":
    main()
