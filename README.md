# Geomatics Master Thesis

## Project Structure

```text
.
|-- src/
|   |-- main.py
|   |-- calculate.py
|   |-- enhance_facades.py
|   |-- gui.py
|   |-- query_copc.py
|   |-- segment.py
|   |-- utils.py
|   `-- visualize.py
|-- data/
|-- environment.yml            # Conda environment
|-- README.md
`-- .gitignore
```

## Installation & Setup

This project uses **Conda** to manage dependencies for handling complex geospatial libraries like PDAL and GeoPandas.

### Prerequisites

Ensure you have a **Conda** distribution like [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed on your system.

### Create the environment

Use the provided `environment.yml` file to set up the evironment.
From the root of this repo run:

```bash
conda env create -f environment.yml
```

Then activate it by running:

```bash
conda activate thesis_env
```

### How to run

Run the project from `src` with:

```bash
python main.py
```

## Preview results

View the results in your favorite point cloud viewer, I recommend [Cloud Compare](https://www.cloudcompare.org/)
