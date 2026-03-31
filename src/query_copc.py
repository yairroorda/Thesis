import json
import re
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pdal
from shapely.geometry import Polygon as ShapelyPolygon

from gui import make_map
from utils import get_logger, status_spinner, timed

logger = get_logger(name="Query")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

_DATASETS = [
    {
        "name": "IGN_LIDAR_HD",
        "file_type": "COPC",
        "wfs_url": "https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES=IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle&OUTPUTFORMAT=application/json",
        "tile_col": "url",
        "crs": "EPSG:2154",
    },
    {
        "name": "AHN6",
        "file_type": "COPC",
        "index_url": "https://basisdata.nl/hwh-ahn/AUX/bladwijzer_AHN6.gpkg",
        "index_cache_name": "index_waterschapshuis",
        "layer": "bladindeling",
        "base_url": "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ/",
        "crs": "EPSG:28992",
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
        "crs": "EPSG:28992",
    }
    for v in range(5, 0, -1)
]


class AOIPolygon:
    def __init__(self, polygon: ShapelyPolygon):
        self.polygon = polygon

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
        return cls(poly)

    def save_to_file(self, path: Path, crs="EPSG:4326") -> None:
        gdf = gpd.GeoDataFrame(geometry=[self.polygon], crs=crs)
        gdf.to_file(path, driver="GeoJSON")

    @classmethod
    def get_from_file(cls, path: Path) -> "AOIPolygon":
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"No geometry found in {path}")
        return cls(gdf.geometry.iloc[0])

    @property
    def wkt(self):
        return self.polygon.wkt

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying polygon
        return getattr(self.polygon, attr)


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

    # Get tile index
    if "wfs_url" in dataset:
        bounds = gdf_polygon.total_bounds
        crs_code = dataset["crs"].split(":")[1]
        bbox_str = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]},urn:ogc:def:crs:EPSG::{crs_code}"

        request_url = f"{dataset['wfs_url']}&BBOX={bbox_str}"
        logger.info("Querying index via WFS ...")

        index_gdf = gpd.read_file(request_url)
        if index_gdf.empty:
            return []
    else:
        kwargs = {"layer": dataset["layer"]} if "layer" in dataset else {}
        index_gdf = gpd.read_file(_download_index(dataset["index_cache_name"], dataset["index_url"]), **kwargs)

    # Get index and aoi in the same CRS before intersecting
    if index_gdf.crs != gdf_polygon.crs:
        index_gdf = index_gdf.to_crs(gdf_polygon.crs)

    # Find intersecting tiles
    hits = gpd.sjoin(index_gdf, gdf_polygon[["geometry"]], how="inner", predicate="intersects")

    if hits.empty:
        return []

    if dataset.get("name") == "IGN_LIDAR_HD":
        return list(dict.fromkeys(hits[dataset["tile_col"]].dropna().tolist()))
    elif "tile_col" in dataset:
        return [f"{dataset['base_url']}/{tile}.LAZ" for tile in dict.fromkeys(hits[dataset["tile_col"]])]
    else:
        urls = []
        for _, row in hits.iterrows():
            x = str(int(row["left"])).zfill(6)
            y = str(int(row["bottom"])).zfill(6)
            urls.append(f"{dataset['base_url']}AHN6_2025_C_{x}_{y}.COPC.LAZ")
        return list(dict.fromkeys(urls))


def _execute_pdal(tile_urls: list[str], aoi: AOIPolygon, file_type: str, output_path: Path) -> Path:
    reader_type = "readers.copc" if file_type == "COPC" else "readers.las"
    stages = []
    merge_inputs = []

    for i, url in enumerate(tile_urls):
        if "data.geopf.fr" in url:
            # Reconstruct the direct OVH S3 Classified bucket URL
            OVH_BASE_URL = "https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/"
            orig_filename = url.split("/")[-1]
            match = re.search(r"LAMB93_([A-Z]{2})_", url)
            subfolder = match.group(1) if match else ""
            # Try both O and C variants
            # I dont know exactly what the difference is but some tiles are only available in one of them, so we check both
            # I cant be bothered to look into it for now
            found_url = None
            for letter in ["O", "C"]:
                filename = orig_filename.replace("PTS_LAMB93", f"PTS_{letter}_LAMB93")
                test_url = f"{OVH_BASE_URL}{subfolder}/{filename}"
                try:
                    with urllib.request.urlopen(test_url) as resp:
                        if resp.status == 200:
                            found_url = test_url
                            break
                except Exception:
                    continue
            if found_url:
                url = found_url
                logger.debug(f"Rewrote URL to OVH S3: {url}")
            else:
                logger.warning(f"Could not find valid OVH S3 URL for {orig_filename}")

        reader_tag = f"reader_{i}"
        reader = {"type": reader_type, "filename": url, "tag": reader_tag}

        if reader_type == "readers.copc":
            reader["polygon"] = aoi.wkt
            reader["requests"] = 64
            stages.append(reader)
            merge_inputs.append(reader_tag)
        else:
            crop_tag = f"crop_{i}"
            crop = {"type": "filters.crop", "polygon": aoi.wkt, "inputs": [reader_tag], "tag": crop_tag}
            stages.extend([reader, crop])
            merge_inputs.append(crop_tag)

        pipeline = stages + [
            {"type": "filters.merge", "inputs": merge_inputs},
            {"type": "writers.copc", "filename": str(output_path), "forward": "all"},
        ]

    with status_spinner("Processing point cloud with PDAL ..."):
        count = pdal.Pipeline(json.dumps(pipeline)).execute()

    logger.info(f"Processed {count} points from {len(tile_urls)} tiles into {output_path}.")
    return output_path


@timed("Pointcloud query")
def get_pointcloud_aoi(aoi: AOIPolygon, output_path: Path, aoi_crs: str = "EPSG:28992", include: list[str] | None = None) -> Path:
    """Download point cloud for the given area and save to output_path."""
    datasets = [d for d in _DATASETS if include is None or d["name"] in include]

    gdf_original = gpd.GeoDataFrame(geometry=[aoi], crs=aoi_crs)

    for dataset in datasets:
        try:
            target_crs = dataset.get("crs", "EPSG:28992")
            gdf = gdf_original.to_crs(target_crs)

            tile_urls = _find_tiles(gdf, dataset)
            if not tile_urls:
                logger.warning(f"Dataset {dataset['name']} returned no intersecting tiles.")
                continue

            logger.info(f"Using {dataset['name']}, found {len(tile_urls)} intersecting tiles.")
            projected_aoi = gdf.geometry.iloc[0]

            return _execute_pdal(tile_urls, projected_aoi, dataset["file_type"], output_path)

        except Exception as exc:
            logger.warning(f"Dataset {dataset['name']} failed: {exc}")
            if output_path.exists():
                output_path.unlink()

    raise RuntimeError("Could not query requested data.")


def demo_france():
    logger.info("Starting LiDAR HD Query GUI...")
    aoi_wgs84 = AOIPolygon(
        ShapelyPolygon([(2.335270987781712, 48.862575335381095), (2.333844052585789, 48.86009786319193), (2.3366013634530987, 48.85942024260344), (2.339294301304051, 48.85932848077683), (2.3401311505166973, 48.86090958411185), (2.337888823780247, 48.861876573590436), (2.335270987781712, 48.862575335381095)])
    )
    # aoi_wgs84 = AOIPolygon.get_from_user("Select polygon AOI for IGN LiDAR HD query")

    get_pointcloud_aoi(aoi=aoi_wgs84, aoi_crs="EPSG:4326", include=["IGN_LIDAR_HD"], output_path=DATA_DIR / "ign_test.copc.laz")


def demo_ahn():
    # aoi = AOIPolygon.get_from_user("Select polygon AOI for AHN query")
    datasets = ["AHN6", "AHN5", "AHN4"]

    aoi_RDnew = AOIPolygon.get_from_file(Path("data/Groningen_plein.geojson"))

    # long_strip = AOIPolygon(ShapelyPolygon([(6.549750540324112, 53.23743153832469), (6.5780044234527395, 53.20113982971779), (6.578433576895122, 53.201165536330386), (6.549885585048543, 53.237447338051304), (6.549750540324112, 53.23743153832469)]))
    # aoi = AOIPolygon.get_from_user("Select polygon AOI for AHN query")

    for dataset in datasets:
        try:
            get_pointcloud_aoi(aoi_RDnew, include=[dataset], output_path=DATA_DIR / f"groningen_plein_{dataset}.copc.laz")
        except RuntimeError as exc:
            logger.warning(f"Skipping {dataset}: {exc}")


if __name__ == "__main__":
    # demo_france()
    # demo_ahn()

    aoi = AOIPolygon.get_from_user("Select area of interest for main processing demo")

    aoi.save_to_file(path=DATA_DIR / "test.geojson")
    aoi_from_file = AOIPolygon.get_from_file(DATA_DIR / "test.geojson")

    get_pointcloud_aoi(aoi_from_file, aoi_crs="EPSG:4326", include=["IGN_LIDAR_HD"], output_path=DATA_DIR / "test.copc.laz")
