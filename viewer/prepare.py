"""Preprocessing data for viewer"""

from pathlib import Path


def reproject_to_web_mercator(laz_file, output_file):
    """Reproject a LAZ file to Web Mercator (EPSG:3857) using PDAL."""
    import json

    import pdal

    # Define the PDAL pipeline
    pipeline = {"pipeline": [laz_file, {"type": "filters.reprojection", "out_srs": "EPSG:3857", "in_srs": "EPSG:28992"}, output_file]}

    # Convert the pipeline to JSON
    pipeline_json = json.dumps(pipeline)

    # Create and execute the PDAL pipeline
    p = pdal.Pipeline(pipeline_json)
    p.execute()


def remap_color_to_16_bit(laz_file: Path, output_file: Path):
    """Remap RGB color values from 8-bit to 16-bit to satisfy viewer requirements."""
    import json

    import pdal

    assign_values = []
    for channel in ["Red", "Green", "Blue"]:
        # Multiply by 256 to shift 8-bit values into the 16-bit range
        assign_values.append(f"{channel}={channel}*256")

    pipeline_steps = [
        {"type": "readers.copc", "filename": str(laz_file)},
        {"type": "filters.assign", "value": assign_values},
        {"type": "writers.copc", "filename": str(output_file)},
    ]

    pdal.Pipeline(json.dumps(pipeline_steps)).execute()


def main(input_file: Path, color_8_bit: bool = False):
    output_dir = Path("viewer/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / input_file.name

    reproject_to_web_mercator(str(input_file), str(output_file))

    if color_8_bit:
        remap_color_to_16_bit(str(output_file), str(output_file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess LAZ files for the viewer.")
    parser.add_argument("input_file", type=Path, help="Path to the input LAZ file.")
    parser.add_argument("--color-8-bit", action="store_true", help="Indicates if the input file has 8-bit color values.")

    args = parser.parse_args()
    main(args.input_file, args.color_8_bit)
