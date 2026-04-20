from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import affinity
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from models import AOIPolygon, Point


def generate_benchmark_aois(
    size: float | tuple[float, float],
    area: AOIPolygon,
    seed: int = 42,
) -> Iterator[AOIPolygon]:
    """Yield random square AOIPolygons sampled from the area bounds."""
    rng = np.random.default_rng(seed)

    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError("size must be a number or a 2-item range")
        size_min = float(size[0])
        size_max = float(size[1])
        if size_min <= 0 or size_max <= 0:
            return
        if size_max < size_min:
            raise ValueError("size range must be ordered as (min, max)")
    else:
        size_min = size_max = float(size)
        if size_min <= 0:
            return

    # Build AOIs in projected CRS so base_size is interpreted in meters.
    working_area = area.to_crs("EPSG:28992") if area.crs.upper() in {"EPSG:4326", "CRS84", "OGC:CRS84"} else area

    minx, miny, maxx, maxy = working_area.bounds
    max_attempts = 1000
    while True:
        for attempts in range(1, max_attempts + 1):
            x = float(rng.uniform(minx, maxx))
            y = float(rng.uniform(miny, maxy))

            if not working_area.contains(ShapelyPoint(x, y)):
                continue

            curr_size = float(rng.uniform(size_min, size_max))
            half = curr_size / 2
            angle = float(rng.uniform(0, 360))
            square = Polygon([
                (x - half, y - half),
                (x + half, y - half),
                (x + half, y + half),
                (x - half, y + half),
            ])
            aoi = AOIPolygon(
                affinity.rotate(square, angle, origin="center"),
                crs=working_area.crs,
            )
            if working_area.crs != area.crs:
                aoi = aoi.to_crs(area.crs)

            yield aoi
            break
        else:
            raise RuntimeError(f"Could not place an AOI center inside the area after {max_attempts} attempts. Try reducing size or changing the area.")


def random_target_point(aoi: AOIPolygon, z_range: tuple[float, float], seed: int = 42) -> Point:
    """Return a random point within the AOI in EPSG:28992 coordinates."""
    rng = np.random.default_rng(seed)
    aoi_rd = aoi.to_crs("EPSG:28992") if aoi.crs != "EPSG:28992" else aoi
    minx, miny, maxx, maxy = aoi_rd.bounds
    minz, maxz = z_range
    for _ in range(1000):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        z = float(rng.uniform(minz, maxz))
        point = ShapelyPoint(x, y, z)
        if aoi_rd.contains(point):
            return Point(x, y, z)
    raise RuntimeError("Could not find a target point within the AOI after 1000 attempts. Try changing the AOI or seed.")


if __name__ == "__main__":
    from utils import get_logger

    logger = get_logger(name="AOI Generator Test")

    output_folder = Path("data/benchmark_aois")
    output_folder.mkdir(exist_ok=True)
    area_path = Path("data/benchmark_aois/area.geojson")

    # Example usage
    if not area_path.exists():
        area_polygon = AOIPolygon.get_from_user("Select area for AOI generation")
        area_polygon.save_to_file(output_folder / "area.geojson")
    else:
        area_polygon = AOIPolygon.get_from_file(area_path)

    aoi_generator = generate_benchmark_aois(size=(500, 1000), area=area_polygon, seed=42)
    aois = [next(aoi_generator) for i in range(15)]

    gpd.GeoDataFrame(geometry=[aoi.polygon for aoi in aois], crs=aois[0].crs if aois else area_polygon.crs).to_file(
        output_folder / "aois.geojson",
        driver="GeoJSON",
    )
    logger.info(f"Generated {len(aois)} AOIs and saved to {output_folder / 'aois.geojson'}")

    target = random_target_point(aois[0], z_range=(0, 10))
    target.save_to_file(output_folder / "target.copc.laz")
    aois[0].save_to_file(output_folder / "aoi_1.geojson")

    # for ix, aoi in enumerate(aois, start=1):
    #     aoi.save_to_file(output_folder / f"aoi_{ix}.geojson")
