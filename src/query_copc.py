import json
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pdal
from shapely.geometry import Polygon as ShapelyPolygon

from gui import _TO_RD, make_map
from utils import get_logger, timed

logger = get_logger(name="Query")

DATA_DIR = Path("data")
DEFAULT_OUTPUT_PATH = DATA_DIR / "output_merged_BK.copc.laz"


_DATASETS = [
    {
        "name": "AHN6",
        "file_type": "COPC",
        "index_url": "https://basisdata.nl/hwh-ahn/AUX/bladwijzer_AHN6.gpkg",
        "index_cache_name": "index_waterschapshuis",
        "layer": "bladindeling",
        "base_url": "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ/",
    },
] + [
    {
        "name": f"AHN{v}",
        "file_type": "LAS",
        "index_url": "https://static.fwrite.org/2022/01/index_sheets.gpkg_.zip",
        "index_cache_name": "index_geotiles",
        "layer": "AHN_subunits",
        "tile_col": "GT_AHNSUB",
        "base_url": f"https://geotiles.citg.tudelft.nl/AHN{v}_T",
    }
    for v in range(5, 0, -1)
]


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


def _download_index(cache_name: str, index_url: str) -> Path:
    local_path = DATA_DIR / f"{cache_name}.gpkg"
    if not local_path.exists():
        logger.info(f"Downloading {cache_name} index ...")
        if index_url.endswith(".zip"):
            tmp_zip = DATA_DIR / "tmp_index.zip"
            urllib.request.urlretrieve(index_url, tmp_zip)
            with zipfile.ZipFile(tmp_zip) as zf:
                gpkg_name = next(n for n in zf.namelist() if n.endswith(".gpkg"))
                local_path.write_bytes(zf.read(gpkg_name))
            tmp_zip.unlink()
        else:
            urllib.request.urlretrieve(index_url, local_path)
    return local_path


def _find_tiles(gdf_polygon: gpd.GeoDataFrame, dataset: dict) -> list[str]:
    index_gdf = gpd.read_file(_download_index(dataset["index_cache_name"], dataset["index_url"]), layer=dataset["layer"])
    hits = gpd.sjoin(index_gdf, gdf_polygon[["geometry"]], how="inner", predicate="intersects")

    if "tile_col" in dataset:
        # AHN1-5: tile name comes directly from an index column
        return [f"{dataset['base_url']}/{tile}.LAZ" for tile in dict.fromkeys(hits[dataset["tile_col"]])]
    else:
        # AHN6: build tile URL from x/y bounding box coordinates
        urls = []
        for _, row in hits.iterrows():
            x = str(int(row["left"])).zfill(6)
            y = str(int(row["bottom"])).zfill(6)
            urls.append(f"{dataset['base_url']}AHN6_2025_C_{x}_{y}.COPC.LAZ")
        return list(dict.fromkeys(urls))


def _execute_pdal(tile_urls: list[str], aoi: Polygon, file_type: str, output_path: Path) -> Path:
    reader_type = "readers.copc" if file_type == "COPC" else "readers.las"
    readers = []
    for url in tile_urls:
        reader = {"type": reader_type, "filename": url}
        if file_type == "COPC":
            reader["requests"] = 16
        readers.append(reader)
    pipeline = readers + [
        {"type": "filters.merge"},
        {"type": "filters.crop", "polygon": aoi.wkt},
        {"type": "writers.copc", "filename": str(output_path), "forward": "all"},
    ]
    count = pdal.Pipeline(json.dumps(pipeline)).execute()
    logger.info(f"Processed {count} points from {len(tile_urls)} tiles into {output_path}.")
    return output_path


@timed("Pointcloud query")
def get_pointcloud_aoi(aoi: Polygon, output_path: Path = DEFAULT_OUTPUT_PATH, include: list[str] | None = None) -> Path:
    """Download AHN point cloud for the given area and save to output_path.

    Parameters:
        aoi (Polygon): Area of interest as a Polygon in RD coordinates.
        output_path (Path): Where to save the resulting point cloud file.
        include (list[str] | None): Optional list of dataset names to include (e.g. ["AHN6", "AHN5"]). If None, defaults to the newest dataset.

    Returns:
        Path: The path to the saved point cloud file.

    """
    datasets = [d for d in _DATASETS if include is None or d["name"] in include]
    gdf = gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:28992")

    for dataset in datasets:
        try:
            tile_urls = _find_tiles(gdf, dataset)
            if not tile_urls:
                logger.warning(f"Dataset {dataset['name']} returned no intersecting tiles.")
                continue
            logger.info(f"Using {dataset['name']}, found {len(tile_urls)} intersecting tiles.")
            return _execute_pdal(tile_urls, aoi, dataset["file_type"], output_path)
        except Exception as exc:
            logger.warning(f"Dataset {dataset['name']} failed: {exc}")
            if output_path.exists():
                output_path.unlink()

    raise RuntimeError("Could not query AHN data.")


if __name__ == "__main__":
    # aoi = Polygon.get_from_user("Select polygon AOI for AHN query")
    datasets = ["AHN6", "AHN5", "AHN4", "AHN3", "AHN2", "AHN1"]
    aoi = Polygon(
        [
            (233691.30497727558, 581987.2869825428),
            (233875.81124650215, 582056.8082196123),
            (233921.904485486, 581956.0270049961),
            (233758.77601513162, 581894.0032933606),
            (233691.30497727558, 581987.2869825428),
        ]
    )

    for dataset in datasets:
        try:
            get_pointcloud_aoi(aoi, include=[dataset], output_path=DATA_DIR / f"groningen_plein_{dataset}.copc.laz")
        except RuntimeError as exc:
            logger.warning(f"Skipping {dataset}: {exc}")
