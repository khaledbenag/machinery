from numpy import ndarray
from pandas import DataFrame

from machinery.loader.base import load_data, load_metadata, load_split_data


def load_laspi_metadata(data_dir: str = None) -> tuple[DataFrame, dict]:
    """
    Generate metadata for LASPI data from the directory structure.

    Args:
        data_dir (str): Path to the LASPI data directory.

    Returns:
        tuple: A tuple containing:
            - DataFrame: A Pandas DataFrame containing LASPI metadata columns.
            - dict: A dictionary mapping class indices to corresponding class labels.

    """
    data_type = "laspi"
    metadata_df, class_mapping = load_metadata(data_dir, data_type)
    return metadata_df, class_mapping


def load_laspi_data(laspi_metadata_df: DataFrame) -> tuple[ndarray, ndarray]:
    """
    Generate metadata for LASPI data from the directory structure.

    Args:
        data_dir (str): Path to the LASPI data directory.

    Returns:
        tuple: A tuple containing:
            - Array: data.
            - Array: corresponding labels.

    """
    laspi_data, laspi_target = load_data(laspi_metadata_df, "laspi")
    return laspi_data, laspi_target


def load_split_laspi_data(
    laspi_train_df: DataFrame, laspi_test_df: DataFrame
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    X_train, y_train, X_test, y_test = load_split_data(
        laspi_train_df, laspi_test_df, "laspi"
    )
    return X_train, y_train, X_test, y_test
