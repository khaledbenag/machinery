import glob
import os
from typing import List, Tuple

import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from machinery.dataset.variables import (
    METALLICADOUR_DATA_PATH,
    METALLICADOUR_POSITION_PATH,
)
from machinery.loader.base import (
    load_data,
    load_metadata,
    load_split_data,
)


def get_files_paths_metadata(
    paths: List[str], metadata_cols: List[str], extension_type: str = "**/*.csv"
) -> pd.DataFrame:
    """
    Generate metadata DataFrame from a list of file paths.

    Args:
        paths (List[str]): List of directory paths.
        metadata_cols (List[str], optional): List of metadata columns.
        extension_type (str, optional): File extension type. Defaults to "**/*.csv".

    Returns:
        pd.DataFrame: Metadata DataFrame.
    """
    # Get a list of paths for all CSV files in specified directories
    csv_file_paths = [
        file
        for path in paths
        for file in glob.glob(os.path.join(path, extension_type), recursive=True)
    ]

    # Augment metadata_df with new rows for CSV files
    augmented_rows = []
    for csv_file_path in csv_file_paths:
        # Extract relevant information from the file path or modify accordingly
        case_name = os.path.basename(os.path.dirname(os.path.dirname(csv_file_path)))
        fault_type = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(csv_file_path)))
        )

        augmented_rows.append([case_name, fault_type, csv_file_path])

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(augmented_rows, columns=metadata_cols[:-1])

    # Factorize the "Case" column again (if needed)
    factorized, unique_values = pd.factorize(metadata_df["Case"])
    metadata_df["class"] = factorized

    return metadata_df


def load_metallicadour_drifts_metadata(
    data_dir: str = None,
) -> tuple[DataFrame, DataFrame, dict]:
    """
    Generate metadata for metallicadour data from the directory structure.

    Args:
        data_dir (str): Path to the LASPI data directory.

    Returns:
        tuple: A tuple containing:
            - DataFrame: A Pandas DataFrame containing METALLICADOUR metadata.
            - DataFrame: A Pandas DataFrame containing METALLICADOUR positional metadata.
            - dict: A dictionary mapping class indices to corresponding class labels.

    """
    data_type = "metallicadour_drifts"
    metadata_df, class_mapping = load_metadata(data_dir, data_type)
    metadata_cols = metadata_df.columns.tolist()

    filepaths = metadata_df.Filepath.tolist()

    # Concatenate file names to position_paths
    data_paths = [os.path.join(path, METALLICADOUR_DATA_PATH) for path in filepaths]
    position_paths = [
        os.path.join(path, METALLICADOUR_POSITION_PATH) for path in filepaths
    ]

    tool_metadata_df = get_files_paths_metadata(
        data_paths, metadata_cols=metadata_cols, extension_type="**/*.csv"
    )
    position_metadata_df = get_files_paths_metadata(
        position_paths, metadata_cols=metadata_cols, extension_type="**/*.xlsx"
    )

    return tool_metadata_df, position_metadata_df, class_mapping


def load_metallicadour_toolwear_metadata(
    data_dir: str = None,
) -> tuple[DataFrame, dict]:
    """
    Generate metadata for LASPI data from the directory structure.

    Args:
        data_dir (str): Path to the LASPI data directory.

    Returns:
        tuple: A tuple containing:
            - DataFrame: A Pandas DataFrame containing LASPI metadata columns.
            - dict: A dictionary mapping class indices to corresponding class labels.

    """
    metadata_df, class_mapping = load_metadata(data_dir, "metallicadour_toolwear")
    return metadata_df, class_mapping


def load_metallicadour_toolwear_data(
    metallicadour_toolwear_metadata_df: DataFrame,
) -> tuple[ndarray, ndarray]:
    """
    Generate metadata for LASPI data from the directory structure.

    Args:
        data_dir (str): Path to the LASPI data directory.

    Returns:
        tuple: A tuple containing:
            - Array: data.
            - Array: corresponding labels.

    """
    metallicadour_toolwear_data, metallicadour_toolwear_target = load_data(
        metallicadour_toolwear_metadata_df, "metallicadour_toolwear"
    )
    return metallicadour_toolwear_data, metallicadour_toolwear_target


def load_metallicadour_toolwear_split_data(
    metallicadour_toolwear_train_df: DataFrame,
    metallicadour_toolwear_test_df: DataFrame,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:

    X_train, y_train, X_test, y_test = load_split_data(
        metallicadour_toolwear_train_df,
        metallicadour_toolwear_test_df,
        "metallicadour_toolwear",
    )
    return X_train, y_train, X_test, y_test


def load_drifts_data(
    metadata_df: DataFrame,
) -> Tuple[ndarray, ndarray]:
    """
    Load data from CSV files specified in the metadata DataFrame and return NumPy arrays.
    """
    metallicadour_drift_data, metallicadour_drift_target = load_data(
        metadata_df, "metallicadour_drifts"
    )
    return metallicadour_drift_data, metallicadour_drift_target
