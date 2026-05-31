import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import geopandas as gpd
import numpy as np
import pdal
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from gui import make_map

RadiusMode = Literal["fixed", "widening_linear"]


def _validate_points_in_aoi(points_xy: list[tuple[float, float]], aoi: "AOIPolygon", labels: list[str]) -> None:
    """Raise when any selected point lies outside the AOI polygon."""
    for (x, y), label in zip(points_xy, labels):
        if not aoi.covers(ShapelyPoint(x, y)):
            raise ValueError(f"{label} is outside the AOI. Please select a point inside the AOI.")


@dataclass
class ProfileConfig:
    name: str
    logging_level: str = "INFO"
    remove: List[str] = None


@dataclass
class ProjectConfig:
    name: str = "test_project"
    crs: str = "EPSG:28992"
    dataset: list[str] = field(default_factory=lambda: ["AHN6", "AHN5"])
    classification_method: str = "myria3d"
    myria3d_vegetation_prob_threshold_pct: float = 90.0
    profile: str = "testing"
    aoi_source: Path | None = None
    overwrite: bool = False


@dataclass
class RunConfig:
    name: str = "test_run"
    overwrite: bool = False
    resolution: float = 2.0
    los_mode: str = "fixed"
    los_radius: float = 0.15
    los_start_radius: float = 0.15
    los_end_radius: float = 0.15
    los_step_length: float = 0.15
    z_height: float = 50.0
    target_source: Path | None = None
    log_level: str = "WARNING"
    vegetation_mode: str = "probabilistic"  # options are "probabilistic", "binary", "ignore"

    def __post_init__(self):
        if self.los_mode == "fixed":
            self.los_start_radius = self.los_radius
            self.los_end_radius = self.los_radius


class ProjectPaths:
    def __init__(self, project_name: str, base_dir: Path = Path("data")):
        self.name = project_name
        self.folder = base_dir / project_name
        self.folder.mkdir(parents=True, exist_ok=True)

        self.runs_folder = self.folder

        self.aoi = self.folder / "aoi.geojson"
        self.input_copc = self.folder / "input.copc.laz"
        self.rescaled_copc = self.folder / "rescaled.copc.laz"
        self.classified_copc = self.folder / "classified.copc.laz"
        self.facades_copc = self.folder / "facades.copc.laz"
        self.project_log = self.folder / "project.log"

    @property
    def is_prepared(self) -> bool:
        return self.facades_copc.exists()


class RunPaths:
    def __init__(self, project_paths: ProjectPaths, run_name: str):
        self.project = project_paths
        self.name = run_name
        self.folder = project_paths.runs_folder / run_name
        self.folder.mkdir(parents=True, exist_ok=True)

        self.run_log = self.folder / "run.log"
        self.metadata = self.folder / "metadata.json"
        self.start_point_copc = self.folder / "start_point.copc.laz"
        self.target_point_copc = self.folder / "target_point.copc.laz"
        self.optimal_path_copc = self.folder / "optimal_path.copc.laz"
        self.output_viewshed_copc_2d = self.folder / "viewshed_2d.copc.laz"
        self.output_viewshed_tif_2d = self.folder / "viewshed_2d.tif"
        self.grid_shell_copc = self.folder / "grid_points_3d_shell.copc.laz"
        self.output_viewshed_copc_3d = self.folder / "viewshed_3d.copc.laz"
        self.output_viewshed_voxel_grid_3d = self.folder / "viewshed_3d_voxel.copc.laz"
        self.output_flight_height_tif = self.folder / "flight_height.tif"
        self.viewable_volume_copc = self.folder / "viewshed_3d_viewable_volume.copc.laz"
        self.viewshed_3d_with_obstacle_distance_copc = self.folder / "viewshed_3d_with_obstacle_distance.copc.laz"


class AOIPolygon:
    def __init__(self, polygon: ShapelyPolygon, crs: str = "EPSG:28992"):
        self.polygon = polygon
        self.crs = crs

    @classmethod
    def get(cls, input_path: Path, title: str = "Draw polygon", overwrite: bool = False) -> "AOIPolygon":
        """Get an AOI polygon from a file or ask the user for input if the file doesn't exist or overwrite is True."""
        if input_path.exists() and not overwrite:
            return cls.get_from_file(input_path)
        else:
            aoi = cls.get_from_user(title=title)
            aoi.save_to_file(input_path)
            return aoi

    @classmethod
    def get_from_user(cls, title: str = "Draw polygon") -> "AOIPolygon":
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

        poly = ShapelyPolygon([(lon, lat) for lat, lon in points_latlon])
        return cls(poly, crs="EPSG:4326")

    def save_to_file(self, path: Path, crs: str | None = None) -> None:
        output_crs = crs or self.crs
        gdf = gpd.GeoDataFrame(geometry=[self.polygon], crs=output_crs)
        gdf.to_file(path, driver="GeoJSON")

    @classmethod
    def get_from_file(cls, path: Path) -> "AOIPolygon":
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"No geometry found in {path}")
        source_crs = gdf.crs.to_string() if gdf.crs else "EPSG:4326"
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return cls(gdf.geometry.iloc[0], crs=source_crs)

    def to_crs(self, crs: str) -> "AOIPolygon":
        gdf = gpd.GeoDataFrame(geometry=[self.polygon], crs=self.crs)
        gdf_projected = gdf.to_crs(crs)
        return AOIPolygon(gdf_projected.geometry.iloc[0], crs=crs)

    @property
    def wkt(self):
        return self.polygon.wkt

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying polygon
        return getattr(self.polygon, attr)


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.array_coords = np.array([x, y, z], dtype=np.float64)
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def get(cls, hag_sample_input_path: Path, input_path: Path, title: str = "Set point", overwrite: bool = False, aoi: AOIPolygon | None = None) -> "Point":
        """
        Get a point from a file or ask the user for input if the file doesn't exist or overwrite is True.

        Args:
            hag_sample_input_path (Path): The path to the HAG sample file.
            input_path (Path): The path to the file containing the point data.
            overwrite (bool, optional): Whether to overwrite the existing file. Defaults to False.

        Returns:
            Point: The point obtained from the file or user input.
        """
        if input_path and input_path.exists() and not overwrite:
            point = Point.get_from_file(input_path)
        else:
            point = Point.get_from_user(hag_sample_input_path=hag_sample_input_path, title=title, aoi=aoi)
            point.save_to_file(input_path)
        return point

    @classmethod
    def get_from_file(cls, path: Path) -> "Point":
        """Read first point from COPC/LAZ and return it as a Point."""
        reader_type = "readers.copc" if ".copc" in path.name.lower() else "readers.las"
        pipeline = pdal.Pipeline(
            json.dumps({
                "pipeline": [
                    {"type": reader_type, "filename": str(path)},
                ]
            })
        )
        count = pipeline.execute()
        if count == 0 or not pipeline.arrays:
            raise ValueError(f"No points found in target source file: {path}")
        first = pipeline.arrays[0][0]
        return cls(first["X"], first["Y"], first["Z"])

    @classmethod
    def get_from_user(cls, hag_sample_input_path: Path, title: str = "Set point", aoi: AOIPolygon | None = None) -> "Point":
        """Let the user pick one point on the map. Returns (x, y, z) in project CRS."""
        import tkinter as tk

        root, map_widget, controls = make_map(title, aoi=aoi)

        p_xy = {"v": None}
        marker = {"p": None}

        transformer = Transformer.from_crs("EPSG:4326", aoi.crs, always_xy=True)

        def on_click(coords):
            lat_c, lon_c = coords[0], coords[1]
            x, y = transformer.transform(lon_c, lat_c)

            if marker["p"] is not None:
                marker["p"].delete()
            marker["p"] = map_widget.set_marker(lat_c, lon_c, text="P1")
            p_xy["v"] = (x, y)

        tk.Label(controls, text="Point P1").pack(anchor="w")

        tk.Label(controls, text="P1 Z").pack(anchor="w", pady=(8, 0))
        pz = tk.Entry(controls)
        pz.insert(0, "1.8")
        pz.pack(fill=tk.X)

        is_hag = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Z is HAG (vs orthometric height)", variable=is_hag).pack(anchor="w", pady=(5, 0))

        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(10, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()

        print(f"You entered Z value: {pz.get()} and HAG mode is {'on' if is_hag.get() else 'off'}")

        if aoi is not None and p_xy["v"] is not None:
            _validate_points_in_aoi([p_xy["v"]], aoi, labels=["Selected point"])
        else:
            raise ValueError("No point was selected.")

        try:
            z_val = float(pz.get())
        except ValueError:
            raise ValueError("Invalid Z value entered. Please enter a valid number for the Z coordinate.")

        p = (*p_xy["v"], z_val)

        root.destroy()

        pt = cls(*p)

        if is_hag.get():
            from calculate import hag_to_ortho

            return hag_to_ortho([pt], input_path=hag_sample_input_path)[0]

        return pt

    def save_to_file(self, path: Path, crs: str = "EPSG:28992") -> None:
        """Save this point to a COPC/LAZ file for later retrieval."""
        dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
        point_data = np.array([(self.x, self.y, self.z)], dtype=dtype)
        from calculate import write_to_copc

        write_to_copc(point_data, path, crs)


class Segment:
    def __init__(self, point1: Point, point2: Point):
        self.point1 = point1
        self.point2 = point2
        self.vector = np.array(
            [point2.x - point1.x, point2.y - point1.y, point2.z - point1.z],
            dtype=np.float64,
        )
        self.length_squared = np.dot(self.vector, self.vector)
        self.length = np.sqrt(self.length_squared)

    @classmethod
    def get_from_user(cls, hag_sample_input_path: Path, title: str = "Set P1/P2", aoi: AOIPolygon | None = None) -> "Segment":
        """Let the user pick two points on the map. Returns (p1, p2) with p1/p2 as (x, y, z) in project CRS."""
        import tkinter as tk

        root, map_widget, controls = make_map(title, aoi=aoi)

        mode = tk.StringVar(value="p1")
        p1_xy = {"v": None}
        p2_xy = {"v": None}
        markers = {"p1": None, "p2": None}

        transformer = Transformer.from_crs("EPSG:4326", aoi.crs, always_xy=True)

        def on_click(coords):
            lat_c, lon_c = coords[0], coords[1]
            x, y = transformer.transform(lon_c, lat_c)
            key = mode.get()

            if markers[key] is not None:
                markers[key].delete()
            markers[key] = map_widget.set_marker(lat_c, lon_c, text=key.upper())
            (p1_xy if key == "p1" else p2_xy)["v"] = (x, y)

            if key == "p1":
                mode.set("p2")

        tk.Label(controls, text="Mode").pack(anchor="w")
        tk.Radiobutton(controls, text="Point P1", variable=mode, value="p1").pack(anchor="w")
        tk.Radiobutton(controls, text="Point P2", variable=mode, value="p2").pack(anchor="w")

        tk.Label(controls, text="P1 Z").pack(anchor="w", pady=(8, 0))
        p1z = tk.Entry(controls)
        p1z.insert(0, "1.8")
        p1z.pack(fill=tk.X)

        tk.Label(controls, text="P2 Z").pack(anchor="w", pady=(8, 0))
        p2z = tk.Entry(controls)
        p2z.insert(0, "1.8")
        p2z.pack(fill=tk.X)

        is_hag = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Z is HAG (vs orthometric height)", variable=is_hag).pack(anchor="w", pady=(5, 0))

        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(10, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()

        if aoi is not None:
            _validate_points_in_aoi([p1_xy["v"], p2_xy["v"]], aoi, labels=["P1", "P2"])

        p1 = (*p1_xy["v"], float(p1z.get()))
        p2 = (*p2_xy["v"], float(p2z.get()))
        root.destroy()

        pts = [Point(*p1), Point(*p2)]
        if is_hag.get():
            from calculate import hag_to_ortho

            pts = hag_to_ortho(pts, input_path=hag_sample_input_path)
        return cls(pts[0], pts[1])


class Cylinder:
    def __init__(
        self,
        segment: Segment,
        min_radius: float,
        max_radius: float,
        step_length: float,
        radius_mode: RadiusMode = "fixed",
    ):
        self.segment = segment
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.step_length = step_length
        self.radius_mode = radius_mode

    @property
    def radius(self) -> float:
        return self.max_radius

    def radius_at_t(self, t: np.ndarray | float) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.float64)
        t_clipped = np.clip(t_arr, 0.0, 1.0)

        if self.radius_mode == "fixed" or self.min_radius == self.max_radius:
            return np.full_like(t_clipped, self.max_radius, dtype=np.float64)

        return self.min_radius + t_clipped * (self.max_radius - self.min_radius)


class ObserverPath:
    def __init__(self, linestring: LineString, crs: str = "EPSG:28992"):
        self.linestring = linestring
        self.crs = crs

    def to_crs(self, crs: str) -> "ObserverPath":
        """Reprojects the path to the target CRS."""
        if self.crs == crs:
            return self

        gdf = gpd.GeoDataFrame(geometry=[self.linestring], crs=self.crs)
        gdf_projected = gdf.to_crs(crs)
        return ObserverPath(gdf_projected.geometry.iloc[0], crs=crs)

    @classmethod
    def get_from_user(cls, hag_sample_input_path: Path, aoi: AOIPolygon, title: str = "Draw Trail for Lookout Search", crs: str = "EPSG:28992") -> "ObserverPath":
        import tkinter as tk

        root, map_widget, controls = make_map(title, aoi=aoi)

        points_latlon: list[tuple[float, float]] = []
        path_obj = {"obj": None}
        marker_list: list = []

        def redraw():
            if path_obj["obj"] is not None:
                path_obj["obj"].delete()
            for m in marker_list:
                m.delete()
            marker_list.clear()
            for pt in points_latlon:
                marker_list.append(map_widget.set_marker(*pt))
            if len(points_latlon) >= 2:
                path_obj["obj"] = map_widget.set_path(points_latlon)

        def on_click(coords):
            points_latlon.append((float(coords[0]), float(coords[1])))
            redraw()

        def clear():
            points_latlon.clear()
            redraw()

        tk.Button(controls, text="Clear Path", command=clear).pack(fill=tk.X)
        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(8, 0))
        tk.Label(controls, text="Click map to draw the trail.\nPress Done when finished.", justify=tk.LEFT).pack(anchor="w", pady=(10, 0))

        map_widget.add_left_click_map_command(on_click)
        root.mainloop()
        root.destroy()

        if len(points_latlon) < 2:
            raise ValueError("A trail must have at least two points.")

        target_crs = aoi.crs if aoi else "EPSG:28992"
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        transformed_points = [transformer.transform(lon, lat) for lat, lon in points_latlon]

        return cls(LineString(transformed_points), crs=target_crs)

    def sample_points(self, project_paths: ProjectPaths, step_size: float, z_height: float = 0.0) -> list[Point]:
        """Samples points evenly along the polyline at the requested step size."""
        length = self.linestring.length
        num_samples = max(2, int(np.ceil(length / step_size)) + 1)
        distances = np.linspace(0, length, num_samples)

        from calculate import hag_to_ortho

        points = []
        for d in distances:
            pt = self.linestring.interpolate(d)
            points.append(Point(pt.x, pt.y, z_height))
        points = hag_to_ortho(points, input_path=project_paths.input_copc)
        return points

    def save_to_file(self, path: Path, crs: str = "EPSG:28992") -> None:
        """Save this path to a GeoJSON file for later retrieval."""
        gdf = gpd.GeoDataFrame(geometry=[self.linestring], crs=crs)
        gdf.to_file(path, driver="GeoJSON")

    @classmethod
    def get_from_file(cls, path: Path) -> "ObserverPath":
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"No geometry found in {path}")
        source_crs = gdf.crs.to_string() if gdf.crs else "EPSG:4326"
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return cls(gdf.geometry.iloc[0], crs=source_crs)

    @classmethod
    def get(cls, input_path: Path, title: str = "Select Route", overwrite: bool = False, aoi: AOIPolygon | None = None) -> "ObserverPath":
        """
        Get a point from a file or ask the user for input if the file doesn't exist or overwrite is True.

        Args:
            input_path (Path): The path to the file containing the point data.
            overwrite (bool, optional): Whether to overwrite the existing file. Defaults to False.

        Returns:
            ObserverPath: The path obtained from the file or user input.
        """
        if input_path and input_path.exists() and not overwrite:
            path = ObserverPath.get_from_file(input_path)
        else:
            path = ObserverPath.get_from_user(hag_sample_input_path=input_path, title=title, aoi=aoi)
            path.save_to_file(input_path)

        if path.crs != aoi.crs:
            path = path.to_crs(aoi.crs)

        return path


class AOICircle:
    """A circular area of interest defined by a center Point and a radius."""

    def __init__(self, center: "Point", radius: float, crs: str = "EPSG:28992"):
        self.center = center
        self.radius = radius
        self.crs = crs
        self.polygon = ShapelyPoint(center.x, center.y).buffer(radius)

    @classmethod
    def get(cls, input_path: Path, title: str = "Set Target and Evaluation Radius", overwrite: bool = False, crs: str = "EPSG:28992") -> tuple["AOICircle", bool]:
        if input_path.exists() and not overwrite:
            # If loading from an existing file, we assume HAG was already resolved in the previous run.
            return cls.get_from_file(input_path), False
        else:
            circle, is_hag = cls.get_from_user(title=title, crs=crs)
            circle.save_to_file(input_path)
            return circle, is_hag

    @classmethod
    def get_from_user(cls, title: str = "Set Target and Evaluation Radius", crs: str = "EPSG:28992") -> tuple["AOICircle", bool]:
        import tkinter as tk

        root, map_widget, controls = make_map(title)

        state = {"center_latlon": None, "polygon_obj": None, "marker_obj": None}

        to_rd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        tk.Label(controls, text="Click map to set the observer target.").pack(anchor="w", pady=(0, 10))

        tk.Label(controls, text="Evaluation Radius (meters)").pack(anchor="w")
        radius_entry = tk.Entry(controls)
        radius_entry.insert(0, "100.0")
        radius_entry.pack(fill=tk.X)

        tk.Label(controls, text="Observer Z Height").pack(anchor="w", pady=(8, 0))
        z_entry = tk.Entry(controls)
        z_entry.insert(0, "1.8")
        z_entry.pack(fill=tk.X)

        is_hag = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Z is Height Above Ground (HAG)", variable=is_hag).pack(anchor="w", pady=(5, 0))

        def redraw(*args):
            if state["polygon_obj"] is not None:
                state["polygon_obj"].delete()
            if state["marker_obj"] is not None:
                state["marker_obj"].delete()

            if state["center_latlon"] is None:
                return

            lat, lon = state["center_latlon"]
            state["marker_obj"] = map_widget.set_marker(lat, lon, text="Observer")

            try:
                r = float(radius_entry.get())
            except ValueError:
                r = 100.0  # Fallback if typing is incomplete

            # Calculate circle projection for live map preview
            x, y = to_rd.transform(lon, lat)
            circle_poly = ShapelyPoint(x, y).buffer(r)

            # Project exterior coordinates back to Lat/Lon for the map widget
            latlon_path = [(clat, clon) for clon, clat in [to_latlon.transform(cx, cy) for cx, cy in circle_poly.exterior.coords]]
            state["polygon_obj"] = map_widget.set_polygon(latlon_path, outline_color="blue", fill_color="", border_width=2)

        radius_entry.bind("<KeyRelease>", redraw)

        def on_click(coords):
            state["center_latlon"] = (float(coords[0]), float(coords[1]))
            redraw()

        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(15, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()

        if state["center_latlon"] is None:
            raise ValueError("No center point was selected.")

        lat, lon = state["center_latlon"]
        x, y = to_rd.transform(lon, lat)
        z_val = float(z_entry.get())
        radius_val = float(radius_entry.get())
        root.destroy()

        # Notice we return the point without HAG applied yet, and pass the flag up!
        from models import Point

        pt = Point(x, y, z_val)
        return cls(center=pt, radius=radius_val, crs=crs), is_hag.get()

    @classmethod
    def get_from_file(cls, path: Path) -> "AOICircle":

        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"No geometry found in {path}")

        source_crs = gdf.crs.to_string() if gdf.crs else "EPSG:4326"
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        polygon = gdf.geometry.iloc[0]

        # Reconstruct center and radius from the saved polygon
        centroid = polygon.centroid
        minx, miny, maxx, maxy = polygon.bounds
        radius = (maxx - minx) / 2.0  # Half the width of the bounding box

        # Note: 2D GeoJSON polygons drop the Z-height.
        # We default to 0.0 here, which is fine because the true 3D target is safely saved in target.copc.laz
        reconstructed_center = Point(centroid.x, centroid.y, 0.0)

        return cls(center=reconstructed_center, radius=radius, crs=source_crs)

    @property
    def wkt(self):
        return self.polygon.wkt

    def to_crs(self, crs: str) -> "AOICircle":
        if self.crs == crs:
            return self
        gdf = gpd.GeoDataFrame(geometry=[self.polygon], crs=self.crs)
        return AOICircle(gdf.to_crs(crs).geometry.iloc[0], crs=crs)

    def save_to_file(self, path: Path, crs: str | None = None) -> None:
        gdf = gpd.GeoDataFrame(geometry=[self.polygon], crs=crs or self.crs)
        gdf.to_file(path, driver="GeoJSON")

    def __getattr__(self, attr):
        return getattr(self.polygon, attr)
