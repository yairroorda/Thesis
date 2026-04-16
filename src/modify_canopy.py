import json
import shutil
from pathlib import Path

import numpy as np
import pdal

from utils import get_logger, timed

logger = get_logger("Augment")


@timed("Augment vegetation density")
def augment_vegetation_density(input_path: Path, output_path: Path, increase_fraction: float = 0.3, jitter_std: float = 0.15, seed: int = 42) -> None:
    """Increase vegetation density by generating synthetic vegetation points and merging them into the original cloud."""

    # Read only vegetation points.
    veg_pipeline = pdal.Pipeline(
        json.dumps(
            {
                "pipeline": [
                    {"type": "readers.copc", "filename": str(input_path)},
                    {"type": "filters.range", "limits": "Classification[5:5]"},
                ]
            }
        )
    )
    veg_pipeline.execute()
    veg_points = veg_pipeline.arrays[0]

    num_new_points = int(veg_points.size * increase_fraction)
    logger.info(f"Original vegetation points: {veg_points.size}. Generating {num_new_points} synthetic points.")

    rng = np.random.default_rng(seed=seed)
    seed_indices = rng.choice(veg_points.size, size=num_new_points, replace=True)
    synthetic_points = np.copy(veg_points[seed_indices])

    synthetic_points["X"] += rng.normal(loc=0.0, scale=jitter_std, size=num_new_points)
    synthetic_points["Y"] += rng.normal(loc=0.0, scale=jitter_std, size=num_new_points)
    synthetic_points["Z"] += rng.normal(loc=0.0, scale=jitter_std, size=num_new_points)

    synthetic_points["Classification"] = 5
    synthetic_points["UserData"] = 1

    synthetic_tmp = output_path.parent / f"{output_path.name}.synthetic_tmp.copc.laz"
    try:
        write_synth_pipeline = {
            "pipeline": [
                {
                    "type": "writers.copc",
                    "filename": str(synthetic_tmp),
                    "forward": "all",
                    "extra_dims": "all",
                }
            ]
        }
        pdal.Pipeline(json.dumps(write_synth_pipeline), arrays=[synthetic_points]).execute()

        merge_pipeline = {
            "pipeline": [
                {"type": "readers.copc", "filename": str(input_path)},
                {"type": "readers.copc", "filename": str(synthetic_tmp)},
                {"type": "filters.merge"},
                {
                    "type": "writers.copc",
                    "filename": str(output_path),
                    "forward": "all",
                    "extra_dims": "all",
                },
            ]
        }
        pdal.Pipeline(json.dumps(merge_pipeline)).execute()
        logger.info(f"Saved augmented point cloud to: {output_path}")
    finally:
        if synthetic_tmp.exists():
            synthetic_tmp.unlink()


if __name__ == "__main__":
    augment_vegetation_density(
        input_path=Path("data/refactor_test/classified.copc.laz"),
        output_path=Path("data/refactor_test/augmented.copc.laz"),
        increase_fraction=1,
        jitter_std=0.20,
        seed=42,
    )
