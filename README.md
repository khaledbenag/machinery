# Machinery Data Loader

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

Machinery Data Loader is a Python package designed to facilitate the loading and preprocessing
of machinery data described in the paper titled 
["Machine Learning for Fault Detection and Diagnosis in Rotating Machines: A Benchmark Data Set"](http://papers.phmsociety.org/index.php/ijphm/article/view/3497).
The datasets can be downloaded from the [PHM Data Science Repository](https://search-data.ubfc.fr/search.php?s=collection%3ADATA-PHM).

The available datasets include:

1. **AMPERE**: Detection and diagnostics of rotor and stator faults in rotating machines.
2. **LASPI**: Detection and diagnostics of gearbox faults.
3. **METALLICADOUR**: Detection and diagnostics of multi-axis robot faults.

## Features

- **Data downloading**: Download data if no local data is given.
- **Data Loading**: Load data from CSV/XLSX files specified in a metadata DataFrame.
- **Data Splitting**: Split metadata DataFrame into training and testing sets.


## Installation

Install the Machinery Data Loader package using pip:

```bash
pip install machinery-loader

```

## Usage

### LASPI
```python
from machinery.loader.base import split_metadata
from machinery.loader.laspi import load_laspi_metadata, load_laspi_data, load_split_laspi_data

# Load metadata
# if no local data_dir is given for LASPI, the module will download the data. 
laspi_metadata_df, laspi_class_mapping = load_laspi_metadata()

# Load global data
data, target = load_laspi_data(laspi_metadata_df)

# Load split
laspi_train_df, laspi_test_df = split_metadata(laspi_metadata_df, group_by_cols=["Load_Percent"], test_size=0.25, random_state=42)
X_train, y_train, X_test, y_test = load_split_laspi_data(laspi_train_df, laspi_test_df)
```

### AMPERE-ROTOR
```python
from machinery.loader.base import split_metadata
from machinery.loader.ampere import load_ampere_rotor_metadata, load_ampere_rotor_data, load_split_ampere_rotor_data

# Load metadata
# if no local data_dir is given for AMPERE, the module will download the data.
ampere_rotor_metadata_df, ampere_rotor_class_mapping = load_ampere_rotor_metadata()

# Load global data
ampere_rotor_data, ampere_rotor_target = load_ampere_rotor_data(ampere_rotor_metadata_df)

# Load split data
ampere_rotor_train_df, ampere_rotor_test_df = split_metadata(ampere_rotor_metadata_df, group_by_cols=["Load_Percent"], test_size=0.25, random_state=42)
X_train, y_train, X_test, y_test = load_split_ampere_rotor_data(ampere_rotor_train_df, ampere_rotor_test_df)
```

### AMPERE-STATOR
```python
from machinery.loader.base import split_metadata
from machinery.loader.ampere import load_ampere_stator_metadata, load_ampere_stator_data, load_split_ampere_stator_data

# Load metadata
# if no local data_dir is given for AMPERE, the module will download the data.
ampere_stator_metadata_df, ampere_stator_class_mapping = load_ampere_stator_metadata()

# Load global data
ampere_stator_data, ampere_stator_target = load_ampere_stator_data(ampere_stator_metadata_df)

# Load split data
ampere_stator_train_df, ampere_stator_test_df = split_metadata(ampere_stator_metadata_df, group_by_cols=["Load_Percent"], test_size=0.25, random_state=42)
X_train, y_train, X_test, y_test = load_split_ampere_stator_data(ampere_stator_train_df, ampere_stator_test_df)
```

### METALLICADOUR-TOOLWEAR
```python
from machinery.loader.base import split_metadata
from machinery.loader.metallicadour import load_metallicadour_toolwear_metadata, load_metallicadour_toolwear_data, load_metallicadour_toolwear_split_data

# Load metadata
# if no local data_dir is given for METALLICADOUR, the module will download the data.
toolwear_metadata_df, toolwear_class_mapping = load_metallicadour_toolwear_metadata()

# Load global data
toolwear_data, toolwear_target = load_metallicadour_toolwear_data(toolwear_metadata_df)

# Load split data
ampere_stator_train_df, ampere_stator_test_df = split_metadata(toolwear_metadata_df, group_by_cols=["Cutting_Depth"], test_size=0.25, random_state=42)
X_train, y_train, X_test, y_test = load_metallicadour_toolwear_split_data(ampere_stator_train_df, ampere_stator_test_df)
```

### METALLICADOUR-DRIFT
```python
from machinery.loader.metallicadour import load_metallicadour_drifts_metadata, load_drifts_data

# Load metadata
# if no local data_dir is given for METALLICADOUR, the module will download the data.
tool_metadata_df, position_metadata_df, class_mapping= load_metallicadour_drifts_metadata()

# Load tool data
tool_data, tool_target = load_drifts_data(tool_metadata_df)

# Load position data
pos_data, pos_target = load_drifts_data(position_metadata_df)
```