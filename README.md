# Geomatics Master Thesis

## Project Structure

```text
.
|-- src/                        # Main application code
|   |-- main.py                 # Entry point for the full pipeline
|   |-- calculate.py
|-- calculations
|   |-- enhance_facades.py
|   |-- gui.py
|   |-- query_copc.py
|   |-- segment.py
|   |-- utils.py                
|   |-- visualize.py
|   `-- __pycache__/
|-- third_party/
|   `-- myria3d/                # Vendored Myria3D deep learning model
|       |-- myria3d/
|       |-- configs/
|       |-- trained_model_assets/
|       |-- environment.yml
|       `-- LICENSE
|-- data/                       # Output directory
|   |-- *.copc.laz              # Processed point cloud files
|   `-- *.tif                   # GeoTIFF visualizations
|-- pixi.toml                   # Pixi workspace and environment configuration
|-- README.md
|-- LICENSE
`-- .gitignore
```

### Key Directories

- **`src/`** — Main Python source code for the pipeline
- **`third_party/myria3d/`** — Vendored [Myria3D](https://github.com/IGNF/myria3d) model for deep learning-based vegetation classification
- **`data/`** — Working directory for intermediate and output point clouds. Created automatically on first run.

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
