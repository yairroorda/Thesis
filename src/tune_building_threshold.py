import json
import subprocess
from pathlib import Path

import numpy as np
import pdal
from pointcloudlib import AHN4

from calculate import calculate_viewshed_2d, export_grid_to_copc, generate_grid, sample_polygon_boundary
from enhance_facades import generate_facades
from models import AOIPolygon, Point
from query_threedbag import ThreeDBAG
from sample_threedbag import sample_on_mesh
from utils import get_logger
from visualize import save_viewshed_as_tif

logger = get_logger(name="TuneBuilding")


def get_sampled_threedbag(aoi: AOIPolygon, obj_path: Path, sampled_path: Path, filtered_path: Path) -> Path:

    # get "perfect" point cloud by sampling the 3DBAG LoD22 mesh at a high density
    ThreeDBAG.fetch(aoi, output_path=obj_path)
    sample_on_mesh(
        input_path=obj_path,
        output_path=sampled_path,
        density=50.0,
    )
    filter_to_aoi(
        input_path=sampled_path,
        output_path=filtered_path,
        aoi=aoi,
    )

    return filtered_path


def filter_to_aoi(input_path: Path, output_path: Path, aoi: AOIPolygon) -> Path:
    """Filter COPC file to AOI polygon boundary."""
    pipeline = {
        "pipeline": [
            {"type": "readers.copc", "filename": str(input_path)},
            {"type": "filters.crop", "polygon": aoi.wkt},
            {"type": "writers.copc", "filename": str(output_path)},
        ]
    }
    count = pdal.Pipeline(json.dumps(pipeline)).execute()
    logger.info(f"Wrote {count} filtered points to {output_path}")
    return output_path


def filter_buildings(input_path: Path, output_path: Path) -> Path:
    """Filter point cloud to building points only, using the "Classification" dimension."""
    pipeline = {
        "pipeline": [
            {"type": "readers.copc", "filename": str(input_path)},
            {"type": "filters.range", "limits": "Classification[6:6]"},
            {"type": "writers.copc", "filename": str(output_path)},
        ]
    }
    count = pdal.Pipeline(json.dumps(pipeline)).execute()
    logger.info(f"Wrote {count} building points to {output_path}")
    return output_path


def main():
    # setup run
    project_folder = Path("data/Delft_bouwkunde")
    aoi_path = project_folder / "aoi.geojson"
    aoi = AOIPolygon.get_from_file(aoi_path).to_crs("EPSG:28992")

    target_path = project_folder / "target_point.copc.laz"
    target = Point.get_from_file(target_path)

    run_name = "tune_building_threshold"
    run_folder = project_folder / run_name
    run_folder.mkdir(parents=True, exist_ok=True)

    grid_resolution = 1.0
    grid_z_height = 50.0
    los_radius = 0.15
    los_step_length = 0.15
    top_points = generate_grid(aoi, resolution=grid_resolution, z_height=grid_z_height, two_d=True)
    for pt in top_points:
        pt.z = grid_z_height
    boundary_points = sample_polygon_boundary(aoi, sample_distance=grid_resolution, z_height=0.0)
    wall_zs = np.arange(0, grid_z_height + grid_resolution, grid_resolution)
    wall_points = [Point(pt.x, pt.y, z) for pt in boundary_points for z in wall_zs]
    export_grid_to_copc(top_points + wall_points, output_path=run_folder / "grid_points_3d_shell.copc.laz")

    # get threedbag file ready
    obj_path = run_folder / "3dbag_lod22_merged_delft.obj"
    sampled_path = run_folder / "3dbag_sampled_perfect.copc.laz"
    threedbag_filtered_path = run_folder / "3dbag_filtered.copc.laz"

    get_sampled_threedbag(aoi=aoi, obj_path=obj_path, sampled_path=sampled_path, filtered_path=threedbag_filtered_path)

    # calculate 2d viewshed on 3DBAG points
    threedbag_viewshed_path = run_folder / "3dbag_viewshed_2d.copc.laz"
    _, _, threedbag_visibility = calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        resolution=grid_resolution,
        radius=los_radius,
        step_length=los_step_length,
        input_path=threedbag_filtered_path,
        output_path=threedbag_viewshed_path,
        z_offset=0.3,
    )
    save_viewshed_as_tif(
        x_coords=threedbag_visibility["X"],
        y_coords=threedbag_visibility["Y"],
        visibility_values=threedbag_visibility["Visibility"],
        aoi=aoi,
        resolution=grid_resolution,
        output_path=threedbag_viewshed_path.with_suffix(".tif"),
    )

    # get AHN4 point cloud file ready

    ahn_input = run_folder / "input.copc.laz"
    ahn_classified_path = run_folder / "classified.copc.laz"
    ahn_facades_path = run_folder / "facades.copc.laz"
    ahn_filtered_path = run_folder / "filtered.copc.laz"

    ahn4 = AHN4(data_dir=run_folder)
    ahn_fetch_result = ahn4.fetch(aoi=aoi.polygon, output_path=ahn_input, aoi_crs=aoi.crs)
    if not ahn_fetch_result or not ahn_fetch_result.exists():
        raise RuntimeError("Failed to fetch AHN4 data for AOI.")
    subprocess.run(f"pixi run -e myria3d python src/segment.py {run_name} {'myria3d'}", shell=True)
    generate_facades(ahn_classified_path, ahn_facades_path, point_spacing=0.2)
    filter_buildings(ahn_facades_path, ahn_filtered_path)

    # calculate 2d viewshed on AHN4 points
    ahn_viewshed_path = run_folder / "ahn_viewshed_2d.copc.laz"
    _, _, ahn_visibility = calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        resolution=grid_resolution,
        radius=los_radius,
        step_length=los_step_length,
        input_path=ahn_filtered_path,
        output_path=ahn_viewshed_path,
        z_offset=0.3,
    )
    save_viewshed_as_tif(
        x_coords=ahn_visibility["X"],
        y_coords=ahn_visibility["Y"],
        visibility_values=ahn_visibility["Visibility"],
        aoi=aoi,
        resolution=grid_resolution,
        output_path=ahn_viewshed_path.with_suffix(".tif"),
    )


if __name__ == "__main__":
    main()
