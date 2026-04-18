import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pdal
from gui import make_map
from pyproj import Transformer
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

RadiusMode = Literal["fixed", "widening_linear"]
_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def _validate_points_in_aoi(points_xy: list[tuple[float, float]], aoi: "AOIPolygon", labels: list[str]) -> None:
    """Raise when any selected point lies outside the AOI polygon."""
    aoi_rd = aoi.to_crs("EPSG:28992") if aoi.crs != "EPSG:28992" else aoi
    for (x, y), label in zip(points_xy, labels):
        if not aoi_rd.covers(ShapelyPoint(x, y)):
            raise ValueError(f"{label} is outside the AOI. Please select a point inside the AOI.")


@dataclass
class ProjectConfig:
    name: str = "test_project"
    dataset: list[str] = field(default_factory=lambda: ["AHN6", "AHN5"])
    classification_method: str = "myria3d"
    profile: str = "testing"
    aoi_source: Path | None = None
    overwrite: bool = False


@dataclass
class RunConfig:
    name: str = "test_run"
    resolution: float = 2.0
    los_mode: str = "fixed"
    los_radius: float = 0.15
    los_start_radius: float = 0.15
    los_end_radius: float = 0.15
    los_step_length: float = 0.15
    z_height: float = 50.0
    target_source: Path | None = None

    def __post_init__(self):
        if self.los_mode == "fixed":
            self.los_start_radius = self.los_radius
            self.los_end_radius = self.los_radius


class ProjectPaths:
    def __init__(self, project_name: str, base_dir: Path = Path("data")):
        self.name = project_name
        self.folder = base_dir / project_name
        self.folder.mkdir(parents=True, exist_ok=True)

        self.runs_folder = self.folder / "runs"
        self.runs_folder.mkdir(parents=True, exist_ok=True)

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
        self.target_point_copc = self.folder / "target_point.copc.laz"
        self.output_viewshed_copc_2d = self.folder / "viewshed_2d.copc.laz"
        self.output_viewshed_tif_2d = self.folder / "viewshed_2d.tif"
        self.grid_shell_copc = self.folder / "grid_points_3d_shell.copc.laz"
        self.output_viewshed_copc_3d = self.folder / "viewshed_3d.copc.laz"
        self.output_viewshed_voxel_grid_3d = self.folder / "viewshed_3d_voxel.copc.laz"
        self.output_flight_height_tif = self.folder / "flight_height.tif"
        self.viewable_volume_copc = self.folder / "viewshed_3d_viewable_volume.copc.laz"


class AOIPolygon:
    def __init__(self, polygon: ShapelyPolygon, crs: str = "EPSG:28992"):
        self.polygon = polygon
        self.crs = crs

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
    def get_from_file(cls, path: Path) -> "Point":
        """Read first point from COPC/LAZ and return it as a Point."""
        reader_type = "readers.copc" if ".copc" in path.name.lower() else "readers.las"
        pipeline = pdal.Pipeline(
            json.dumps(
                {
                    "pipeline": [
                        {"type": reader_type, "filename": str(path)},
                    ]
                }
            )
        )
        count = pipeline.execute()
        if count == 0 or not pipeline.arrays:
            raise ValueError(f"No points found in target source file: {path}")
        first = pipeline.arrays[0][0]
        return cls(first["X"], first["Y"], first["Z"])

    @classmethod
    def get_from_user(cls, title: str = "Set point", aoi: AOIPolygon | None = None) -> "Point":
        """Let the user pick one point on the map. Returns (x, y, z) in RD."""
        import tkinter as tk

        root, map_widget, controls = make_map(title, aoi=aoi)

        p_xy = {"v": None}
        marker = {"p": None}

        def on_click(coords):
            lat_c, lon_c = coords[0], coords[1]
            x, y = _TO_RD.transform(lon_c, lat_c)

            if marker["p"] is not None:
                marker["p"].delete()
            marker["p"] = map_widget.set_marker(lat_c, lon_c, text="P1")
            p_xy["v"] = (x, y)

        tk.Label(controls, text="Point P1").pack(anchor="w")

        tk.Label(controls, text="P1 Z").pack(anchor="w", pady=(8, 0))
        pz = tk.Entry(controls)
        pz.insert(0, "8.0")
        pz.pack(fill=tk.X)

        is_hag = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Z is HAG (vs NAP)", variable=is_hag).pack(anchor="w", pady=(5, 0))

        tk.Button(controls, text="Done", command=root.quit).pack(fill=tk.X, pady=(10, 0))
        map_widget.add_left_click_map_command(on_click)

        root.mainloop()

        if aoi is not None:
            _validate_points_in_aoi([p_xy["v"]], aoi, labels=["Selected point"])

        p = (*p_xy["v"], float(pz.get()))
        root.destroy()

        pt = cls(*p)
        if is_hag.get():
            from calculate import hag_to_nap

            return hag_to_nap([pt])[0]
        return pt

    def save_to_file(self, path: Path, crs: str = "EPSG:28992") -> None:
        """Save this point to a COPC/LAZ file for later retrieval."""
        dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
        point_data = np.array([(self.x, self.y, self.z)], dtype=dtype)
        from calculate import write_to_copc

        write_to_copc(point_data, path, crs=crs)


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
    def get_from_user(cls, title: str = "Set P1/P2", aoi: AOIPolygon | None = None) -> "Segment":
        """Let the user pick two points on the map. Returns (p1, p2) with p1/p2 as (x, y, z) in RD."""
        import tkinter as tk

        root, map_widget, controls = make_map(title, aoi=aoi)

        mode = tk.StringVar(value="p1")
        p1_xy = {"v": None}
        p2_xy = {"v": None}
        markers = {"p1": None, "p2": None}

        def on_click(coords):
            lat_c, lon_c = coords[0], coords[1]
            x, y = _TO_RD.transform(lon_c, lat_c)
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
        p1z.insert(0, "8.0")
        p1z.pack(fill=tk.X)

        tk.Label(controls, text="P2 Z").pack(anchor="w", pady=(8, 0))
        p2z = tk.Entry(controls)
        p2z.insert(0, "10.0")
        p2z.pack(fill=tk.X)

        is_hag = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Z is HAG (vs NAP)", variable=is_hag).pack(anchor="w", pady=(5, 0))

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
            from calculate import hag_to_nap

            pts = hag_to_nap(pts)
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
