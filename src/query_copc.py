from pathlib import Path

from pointcloudlib import AHN1, AHN2, AHN3, AHN4, AHN5, AHN6, CanElevation, IGNLidarHD, PointCloudProvider

from models import AOIPolygon
from utils import get_logger

logger = get_logger(name="Query")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def demo_france():
    logger.info("Starting LiDAR HD query demo...")
    aoi_wgs84 = AOIPolygon.get_from_file(Path("data/example_aoi/Paris_eiffel_tower.geojson"))
    provider = IGNLidarHD(data_dir=DATA_DIR)
    provider.fetch(aoi=aoi_wgs84.polygon, aoi_crs=aoi_wgs84.crs, output_path=DATA_DIR / "ign_test.copc.laz")


def demo_ahn():
    providers: list[type[PointCloudProvider]] = [AHN6, AHN5, AHN4]
    aoi_rdnew = AOIPolygon.get_from_file(Path("data/Groningen_plein.geojson"))

    for provider_cls in providers:
        provider = provider_cls(data_dir=DATA_DIR)
        try:
            provider.fetch(
                aoi=aoi_rdnew.polygon,
                aoi_crs=aoi_rdnew.crs,
                output_path=DATA_DIR / f"groningen_plein_{provider_cls.__name__}.copc.laz",
            )
        except RuntimeError as exc:
            logger.warning(f"Skipping {provider_cls.__name__}: {exc}")


if __name__ == "__main__":
    demo_france()
