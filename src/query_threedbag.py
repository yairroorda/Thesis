# 3DBAG LoD22 OBJ downloader and filter by AOI
# based on 3d tiles and the script(https://github.com/3DBAG/3dbag-scripts/blob/main/tile_download.py#L31) by Ravi Peters (https://github.com/Ylannl)

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
from shapely.geometry import box

from models import AOIPolygon
from utils import get_logger, status_spinner, timed

logger = get_logger(name="3DBAG")

TILE_INDEX_URL = "https://data.3dbag.nl/latest/tile_index.fgb"


def _obj_vertex_index(token: str) -> int:
    return int(token.split("/")[0])


def _obj_bbox_intersects_aoi(used_indices: set[int], vertices: list[tuple[float, float, float]], aoi_polygon) -> bool:
    if not used_indices:
        return False
    xs = [vertices[i - 1][0] for i in used_indices]
    ys = [vertices[i - 1][1] for i in used_indices]
    return box(min(xs), min(ys), max(xs), max(ys)).intersects(aoi_polygon)


def _append_filtered_lod22_obj(src_obj: Path, out_file, aoi_polygon, object_prefix: str, vertex_offset: int) -> tuple[int, int]:
    vertices: list[tuple[float, float, float]] = []
    objects: list[tuple[str, list[list[int]], set[int]]] = []
    current_name = "unnamed"
    current_faces: list[list[int]] = []
    current_used: set[int] = set()

    def flush_current():
        nonlocal current_name, current_faces, current_used
        if current_faces:
            objects.append((current_name, current_faces, current_used))
        current_faces = []
        current_used = set()

    with open(src_obj) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("o ") or line.startswith("g "):
                flush_current()
                current_name = line.strip().split(maxsplit=1)[1]
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                face = [_obj_vertex_index(tok) for tok in tokens]
                current_faces.append(face)
                current_used.update(face)
    flush_current()

    kept_count = 0
    current_offset = vertex_offset
    for name, faces, used in objects:
        if not _obj_bbox_intersects_aoi(used, vertices, aoi_polygon):
            continue
        kept_count += 1
        ordered = sorted(used)
        remap = {old: i + 1 for i, old in enumerate(ordered)}
        out_file.write(f"o {object_prefix}_{name}\n")
        for idx in ordered:
            v = vertices[idx - 1]
            out_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            mapped = [str(current_offset + remap[idx]) for idx in face]
            out_file.write("f " + " ".join(mapped) + "\n")
        current_offset += len(ordered)

    return kept_count, current_offset


class ThreeDBAG:
    name = "3DBAG"
    crs = "EPSG:28992"

    @classmethod
    @timed("3DBAG LoD22 OBJ download")
    def fetch(
        cls,
        aoi: "AOIPolygon",
        output_path: Path | str = "data/3dbag_lod22_merged.obj",
        keep_tiles: bool = False,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        aoi_rd = gpd.GeoDataFrame(geometry=[aoi.polygon], crs=aoi.crs).to_crs(cls.crs)
        aoi_poly = aoi_rd.geometry.iloc[0]
        bbox = tuple(aoi_rd.total_bounds)

        logger.info(f"Querying tile index for bbox: {bbox}")
        tiles = gpd.read_file(TILE_INDEX_URL, bbox=bbox)
        if tiles.crs != aoi_rd.crs:
            tiles = tiles.to_crs(aoi_rd.crs)
        tiles = tiles[tiles.intersects(aoi_poly)]

        if tiles.empty:
            raise RuntimeError("No 3DBAG tiles intersect AOI")

        tmp_dir = output_path.parent / "_3dbag_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tile_objs: list[Path] = []
        with status_spinner("Downloading LoD22 OBJ tiles..."):
            for _, tile in tiles.iterrows():
                tile_id = tile["tile_id"].replace("/", "-")
                zip_path = tmp_dir / f"{tile_id}.zip"
                urlretrieve(tile["obj_download"], zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    lod22_name = next(n for n in zf.namelist() if "LoD22" in n and n.endswith(".obj"))
                    extracted = tmp_dir / f"{tile_id}-LoD22.obj"
                    with zf.open(lod22_name) as src, open(extracted, "wb") as dst:
                        dst.write(src.read())
                tile_objs.append(extracted)

        total_kept = 0
        with open(output_path, "w") as merged:
            merged.write("# 3DBAG LoD22 merged OBJ\n")
            vertex_offset = 0
            for obj_path in tile_objs:
                kept, vertex_offset = _append_filtered_lod22_obj(
                    obj_path,
                    merged,
                    aoi_poly,
                    obj_path.stem,
                    vertex_offset,
                )
                total_kept += kept

        logger.info(f"Saved merged LoD22 OBJ to {output_path} ({total_kept} objects kept)")

        if not keep_tiles:
            for p in tmp_dir.glob("*"):
                if p.is_file():
                    p.unlink()
            tmp_dir.rmdir()

        return output_path


if __name__ == "__main__":
    # if Path("data/3dbag_aoi.geojson").exists():
    #     aoi = AOIPolygon.get_from_file("data/3dbag_aoi.geojson")
    # else:
    #     aoi = AOIPolygon.get_from_user("Select area of interest for 3DBAG LoD22")
    # aoi.save_to_file("data/3dbag_aoi.geojson")
    aoi = AOIPolygon.get_from_file("data/Delft_bouwkunde/aoi.geojson")
    out = ThreeDBAG.fetch(aoi, output_path="data/3dbag_lod22_merged_delft.obj")
    print(f"Downloaded to: {out}")
