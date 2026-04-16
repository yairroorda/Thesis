import importlib
import tkinter as tk

tkintermapview = importlib.import_module("tkintermapview")

# Default centre
_START_LAT, _START_LON = 52.005818, 4.370221  # Delft
_ZOOM = 18


def make_map(title, aoi=None):
    """Create a tkinter window with a map widget and a side panel."""
    root = tk.Tk()
    root.title(title)
    root.geometry("1100x700")

    map_widget = tkintermapview.TkinterMapView(root, width=850, height=700, corner_radius=0)
    map_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    controls = tk.Frame(root, padx=10, pady=10)
    controls.pack(side=tk.RIGHT, fill=tk.Y)

    lat, lon = _START_LAT, _START_LON
    if aoi is not None:
        aoi_wgs84 = aoi.to_crs("EPSG:4326") if aoi.crs != "EPSG:4326" else aoi
        centroid = aoi_wgs84.centroid
        lat, lon = float(centroid.y), float(centroid.x)

        outline_latlon = [(float(y), float(x)) for x, y in aoi_wgs84.exterior.coords]
        if len(outline_latlon) >= 2:
            map_widget.set_path(outline_latlon, color="red", width=3)

    map_widget.set_position(float(lat), float(lon))
    map_widget.set_zoom(_ZOOM)

    return root, map_widget, controls


def main():
    from calculate import Point, Segment
    from models import AOIPolygon
    from utils import get_logger

    logger = get_logger(name="GUI Test")

    polygon = AOIPolygon.get_from_user("Test polygon input")
    logger.info(f"Collected polygon: {polygon}")
    points = Segment.get_from_user("Test segment input", aoi=polygon)
    logger.info(f"Collected points: SEGMENT(POINT({points.point1.x}, {points.point1.y}, {points.point1.z}), POINT({points.point2.x}, {points.point2.y}, {points.point2.z}))")
    point = Point.get_from_user("Test point input", aoi=polygon)
    logger.info(f"Collected point: POINT({point.x}, {point.y}, {point.z})")


if __name__ == "__main__":
    main()
