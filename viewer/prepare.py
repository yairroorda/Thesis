"""Preprocessing data for viewer"""

import json
from pathlib import Path

import pdal


def reproject_to_web_mercator(copc_file, output_file):
    """Reproject a LAZ file to Web Mercator (EPSG:3857) using PDAL."""

    pipeline_steps = [
        str(copc_file),
        {"type": "filters.reprojection", "out_srs": "EPSG:3857", "in_srs": "EPSG:28992"},
        {
            "type": "writers.copc",
            "filename": str(output_file),
            "forward": "all",  # This is the magic key!
            "extra_dims": "all",  # THE FIX: Keeps the actual per-point data
        },
    ]

    pipeline = {"pipeline": pipeline_steps}

    # Convert the pipeline to JSON
    pipeline_json = json.dumps(pipeline)

    # Create and execute the PDAL pipeline
    p = pdal.Pipeline(pipeline_json)
    p.execute()


def remap_color_to_16_bit(copc_file: Path, output_file: Path):
    """Remap RGB color values from 8-bit to 16-bit to satisfy viewer requirements."""

    assign_values = []
    for channel in ["Red", "Green", "Blue"]:
        # Multiply by 256 to shift 8-bit values into the 16-bit range
        assign_values.append(f"{channel}={channel}*256")

    pipeline_steps = [
        {"type": "readers.copc", "filename": str(copc_file)},
        {"type": "filters.assign", "value": assign_values},
        {
            "type": "writers.copc",
            "filename": str(output_file),
            "forward": "all",
            "extra_dims": "all",
        },
    ]

    pdal.Pipeline(json.dumps(pipeline_steps)).execute()


def hijack_intensity(copc_file: Path, output_file: Path):
    """Hijack the viewshed intensity channel for the viewer."""

    pipeline_steps = [
        str(copc_file),
        {
            "type": "filters.assign",
            "value": ["Intensity = Visibility * 65535"],
        },
        {"type": "filters.reprojection", "out_srs": "EPSG:3857", "in_srs": "EPSG:28992"},
        {
            "type": "writers.copc",
            "filename": str(output_file),
        },
    ]

    pdal.Pipeline(json.dumps(pipeline_steps)).execute()


def main(input_file: Path, color_8_bit: bool = False, hijack: bool = False):
    output_dir = Path("viewer/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / input_file.name

    reproject_to_web_mercator(str(input_file), str(output_file))

    if color_8_bit:
        remap_color_to_16_bit(str(output_file), str(output_file))

    if hijack:
        hijack_intensity(str(output_file), str(output_file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess LAZ files for the viewer.")
    parser.add_argument("input_file", type=Path, help="Path to the input LAZ file.")
    parser.add_argument("--color-8-bit", action="store_true", help="Indicates if the input file has 8-bit color values.")
    parser.add_argument("--hijack", action="store_true", help="Hijack the intensity channel to display visibility information.")

    args = parser.parse_args()
    main(args.input_file, args.color_8_bit, args.hijack)
