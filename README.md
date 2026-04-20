# Geomatics Master Thesis

## Project Structure

```text
.
|-- src/                            # Main application code
|   |-- main.py                     # Entry point for the full pipeline
|   |-- calculate.py
|   |-- enhance_facades.py
|   |-- evaluate.py
|   |-- gui.py
|   |-- models.py
|   |-- modify_canopy.py
|   |-- query_copc.py
|   |-- query_threedbag.py
|   |-- sample_threedbag.py
|   |-- segment.py
|   |-- tune_building_threshold.py
|   |-- utils.py
|   `-- visualize.py
|-- third_party/
|   `-- myria3d/                    # Vendored Myria3D deep learning model
|-- data/                           # Sample datasets and generated run outputs
|   `- project_1/                   # Project folders (one per dataset or aoi)
|       `-- run_1/                  # Run folders contain outputs for a single pipeline execution
|           |-- viewshed.copc.laz   # Point cloud outputs in LAZ format
|           |-- metadata.json       # Metadata outputs
|           `-- ...                 # Other outputs (e.g. logs, intermediate files) depending on profile
|-- config.toml                     # Runtime configuration and profile behavior
|-- pixi.toml                       # Pixi workspace and environment configuration
|-- pixi.lock
|-- README.md
|-- LICENSE
`-- .gitignore
```

### Key Directories

- **`src/`** — Main Python source code for the pipeline
- **`third_party/myria3d/`** — Vendored [Myria3D](https://github.com/IGNF/myria3d) model for deep learning-based vegetation classification
- **`data/`** — Contains sample/reference datasets and project folders. Pipeline outputs are written to `data/<project_name>/<run_name>/`.

## Installation & Setup

### Prerequisites

This project uses **Pixi** to manage dependencies for handling complex geospatial and deep learning libraries like PDAL, GeoPandas, and PyTorch. The myria3d environment requires Linux—Windows users should use WSL.

Install Pixi from [pixi.sh](https://pixi.sh) or using the command below

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Installation on Linux

#### 1. Clone the repository

```bash
git clone https://github.com/yairroorda/Thesis.git
cd Thesis
```

#### 2. Install environments (optional)

Either install the pixi environment separately or directly run the pipeline which will automatically install the environment. In both cases is will take some time and requires at least **8.5 GB** of disk space.

```bash
pixi install
```

#### 3. Run the pipeline

```bash
pixi run pipeline
```

### Installation on Windows (WSL2)

**Windows / WSL**: The myria3d deep learning environment requires Linux so for that functionality Windows users must use WSL2.

#### 1. Install WSL2 and Ubuntu

```powershell
wsl.exe --install ubuntu
```

Then open Ubuntu and update the package manager:

```bash
wsl
sudo apt update
```

#### 2. Install Pixi in WSL

```bash
curl -fsSL https://pixi.sh/install.sh | sh
bash # restart your terminal after
```

#### 3. Clone the repository to WSL home directory

**Important**: Clone to your WSL home directory (`/home/<user>/`), not to the Windows filesystem mounted at `/mnt/c/`. This avoids file permission issues.

```bash
cd ~
mkdir <projectname>
cd <projectname>
git clone https://github.com/yairroorda/Thesis.git
cd Thesis
```

#### 4. install environments (optional)

Either install the pixi environment separately or directly run the pipeline which will automatically install the environment. In both cases is will take some time and requires at least **8.5 GB** of disk space.

```bash
pixi install
```

#### 5. Run the pipeline

```bash
pixi run pipeline
```

The full pipeline will run and generate output files in the `data/` directory.

### Output and profiles

- Outputs are stored in `data/<project_name>/<run_name>/` (one folder per run).
- File names inside a run folder are stable, for example: `input.copc.laz`, `viewshed.tif`, `metadata.json`, `run.log`.
- If run name is not set, the application uses `tmp`.
- If a run folder already exists, the pipeline raises an error unless overwrite is enabled.
- A metadata JSON and run logfile are generated per run: `metadata.json`, `run.log`.
- Profile behavior is configured in `config.toml`.
- `production` keeps minimal files.
- `testing` keeps additional intermediate outputs.

When `THESIS_PROFILE` is not set, the application defaults to `production`.

### How to run

Run the full pipeline:

```bash
pixi run pipeline
```

Or run individual steps:

- Vegetation classification only: `pixi run -e myria3d classify <filename> <method>`
- Other custom tasks: See `pixi.toml` for available tasks

## Preview results

View the results in your favorite point cloud viewer, I recommend [Cloud Compare](https://www.cloudcompare.org/)
