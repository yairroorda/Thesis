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

## Preview results
View the results in your favorite point cloud viewer, I recommend [Cloud Compare](https://www.cloudcompare.org/)
