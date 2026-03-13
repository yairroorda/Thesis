from pathlib import Path

from calculate import Point, calculate_viewshed_2d, export_grid_to_copc, hag_to_nap
from query_copc import Polygon
from utils import get_logger, timed
from visualize import save_viewshed_as_tif

logger = get_logger(name="Main")


@timed("Total processing")
def main() -> None:
    # Query the relevant points
    # aoi = Polygon.get_from_user("Select area of interest")
    aoi = Polygon(
        [
            (233691.30497727558, 581987.2869825428),
            (233875.81124650215, 582056.8082196123),
            (233921.904485486, 581956.0270049961),
            (233758.77601513162, 581894.0032933606),
            (233691.30497727558, 581987.2869825428),
        ]
    )

    # Generate 2D viewshed
    output_facades_path = Path("data/groningen_plein_AHN4_facades.copc.laz")
    target_hag = Point(233851.5, 581986.8, 1.7)
    target = hag_to_nap([target_hag])[0]
    export_grid_to_copc([target], output_path=Path("data/target_point.copc.laz"))

    resolution = 0.5
    radius = 0.15
    output_path = Path("data/groningen_plein_AHN4_viewshed")

    _, _, visibility_points = calculate_viewshed_2d(
        target=target,
        aoi=aoi,
        radius=radius,
        resolution=resolution,
        input_path=output_facades_path,
        output_path=output_path,
        z_offset=0.3,
    )

    save_viewshed_as_tif(
        x_coords=visibility_points["X"],
        y_coords=visibility_points["Y"],
        visibility_values=visibility_points["Visibility"],
        aoi=aoi,
        resolution=resolution,
        output_path=output_path.with_suffix(".tif"),
    )


if __name__ == "__main__":
    main()
