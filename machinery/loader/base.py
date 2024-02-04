import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from machinery.dataset.downloader import download_data

metallicadour_cols = [
    "Case",
    "Type",
    "Filepath",
]
tool_wear_cols = [
    "Case",
    "Cutting_Depth",
    "Feed_Rate",
    "Speed",
    "Filepath",
]
standard_cols = [
    "Case",
    "Speed_Frequency",
    "Load_Percent",
    "Speed",
    "Filepath",
]


def load_metadata(
    data_dir: [str, None] = None, data_type: str = None
) -> tuple[DataFrame, dict]:
    """
    Generate metadata from the directory structure.

    Returns:
        DataFrame: A Pandas DataFrame containing metadata columns.

    """

    if data_type is None:
        raise Exception("Data type not specified")
    file_extension = ".csv"

    if data_type == "metallicadour_drifts":
        metadata_cols = metallicadour_cols
    elif data_type == "metallicadour_toolwear":
        metadata_cols = tool_wear_cols
    else:
        metadata_cols = standard_cols

    if data_dir is not None:
        if not os.path.exists(data_dir):
            raise Exception(f"The given path '{data_dir}' does not exist")
    else:
        data_dir = download_data(data_type)

    metadata: List[List] = []
    for case_name in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        for subcase_name in os.listdir(case_dir):
            subcase_dir = os.path.join(case_dir, subcase_name)
            if not os.path.isdir(subcase_dir):
                continue

            if data_type == "metallicadour_toolwear":
                match = re.match(r"(\d+)mm_(\d+)mm_mn_(\d+)rpm", subcase_name)
            elif data_type == "metallicadour_drifts":
                pattern = (
                    r"Drifts_axis_(\d+)_?([+-]?\d+(\.\d+)?)|Drifts_axis_(\d+)_?([+-]?\d+(\.\d+)?)"
                    ""
                    r"_axis_(\d+)_?([+-]?\d+(\.\d+)?)|Healthy_robot"
                )
                match = re.match(pattern, subcase_name)
            else:
                match = re.match(r"(\d+)hz_(\d+)%_(\d+)rpm", subcase_name)

            if match:
                if data_type == "metallicadour_drifts":
                    case_name = subcase_name
                    fault_type = os.path.basename(case_dir)
                    file_path = subcase_dir
                    metadata.append(
                        [
                            case_name,
                            fault_type,
                            file_path,
                        ]
                    )
                else:
                    speed_frequency, load_percent, speed = map(int, match.groups())
                    for filename in os.listdir(subcase_dir):
                        if filename.endswith(file_extension):
                            file_path = os.path.join(subcase_dir, filename)
                            metadata.append(
                                [
                                    case_name,
                                    speed_frequency,
                                    load_percent,
                                    speed,
                                    file_path,
                                ]
                            )
                        else:
                            logger.warning(
                                f"The file {filename} is excluded. It is not in the required format {file_extension}"
                            )
            else:
                logger.warning(
                    f"Folder {subcase_name} does not match the format: Xhz_Y%_Zrpm where X, Y, Z are integer values"
                )

    metadata_df = pd.DataFrame(metadata, columns=metadata_cols)
    factorized, unique_values = pd.factorize(metadata_df["Case"])
    class_mapping = dict(zip(np.unique(factorized), unique_values))
    metadata_df["class"] = factorized
    metadata_df.reset_index(drop=True)
    return metadata_df, class_mapping


def load_csv_data(filepaths: List[str], num_cols: int) -> np.ndarray:
    """
    Load data from CSV or Excel files and preprocess.

    Args:
        filepaths (List[str]): List of file paths to load.
        num_cols (int): Expected number of columns in each file.

    Returns:
        np.ndarray: Numpy array containing the loaded and preprocessed data.
    """
    data = []

    # parameter used for data with different number of rows among files
    min_rows = float("inf")

    for filepath in tqdm(filepaths):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            if str(filepath).endswith(".csv"):
                df = pd.read_csv(filepath, encoding="utf-8")
                if df.shape[1] != num_cols:
                    raise ValueError(
                        f"Inconsistent number of columns in file {filepath}. Expected: {num_cols}, Actual: {df.shape[1]}"
                    )

                # Update min_rows based on the minimum number of rows in the current file
                min_rows = min(min_rows, df.shape[0])

                data.append(df.values[:min_rows])

            # METALLICADOUR drifts positions
            elif str(filepath).endswith(".xlsx"):
                df = pd.read_excel(filepath)
                if df.shape[1] != num_cols:
                    raise ValueError(
                        f"Inconsistent number of columns in file {filepath}. Expected: {num_cols}, Actual: {df.shape[1]}"
                    )

                # Update min_rows based on the minimum number of rows in the current file
                min_rows = min(min_rows, df.shape[0])

                data.append(df.values[:min_rows])

            else:
                raise Exception("File format not accepted. Use CSV/XLSX format.")
        except Exception as e:
            raise Exception(
                f"Error while loading CSV/XLSX file: {filepath}, with error: {e}"
            )

    data = [arr[:min_rows] for arr in data]
    data = np.stack(data, axis=0)

    return data


def load_data(metadata_df: DataFrame, data_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV files specified in the metadata DataFrame and return NumPy arrays.

    Args:
        metadata_df (DataFrame): The metadata DataFrame containing file paths and class information.
        data_type (str): Type of data ('laspi', 'ampere_rotor', 'ampere_stator', 'metallicadour_tool_wear').

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing data (X) as a NumPy array and labels (y) as a NumPy array.

    """
    accepted_data_type = [
        "laspi",
        "ampere_rotor",
        "ampere_stator",
        "metallicadour_toolwear",
        "metallicadour_drifts",
    ]
    if data_type not in accepted_data_type:
        raise ValueError(f"data_type should be one of:  {accepted_data_type} ")

    filepaths = metadata_df.Filepath.tolist()
    y = metadata_df["class"].to_numpy()
    map_num_cols = {
        "laspi": 7,
        "ampere_rotor": 11,
        "ampere_stator": 11,
        "metallicadour_toolwear": 12,
        "metallicadour_drifts": 12,
    }

    num_cols = map_num_cols[data_type]

    data = load_csv_data(filepaths, num_cols)

    return data, y


def split_metadata(
    metadata_df: pd.DataFrame,
    group_by_cols: [str] = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split metadata DataFrame into training and testing sets.

    Args:
        metadata_df (pd.DataFrame): Metadata DataFrame.
        group_by_cols (str, optional): Column(s) to group by before splitting. Defaults to None.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.25.
        random_state (int, optional): Seed for random number generation. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
    """
    try:
        if group_by_cols in ["", "", None, "none", "None"]:
            groups = metadata_df.groupby(["Case"])
        else:
            cols = ["Case"]
            cols.extend(group_by_cols)
            groups = metadata_df.groupby(cols)
    except Exception:
        raise Exception(f"Column(s) {group_by_cols} is/are not valid")

    train_dfs = []
    test_dfs = []

    for group_name, group_data in groups:
        train_data, test_data = train_test_split(
            group_data, test_size=test_size, random_state=random_state
        )
        train_dfs.append(train_data)
        test_dfs.append(test_data)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    return train_df, test_df


def load_split_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, data_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess data for training and testing.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        data_type (str): Type of data to be loaded.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
            - X_train (np.ndarray): Training features.
            - y_train (np.ndarray): Training labels.
            - X_test (np.ndarray): Testing features.
            - y_test (np.ndarray): Testing labels.
    """
    X_train, y_train = load_data(train_df, data_type)
    X_test, y_test = load_data(test_df, data_type)

    # Make sure to have the same shape (shape[1]) for X_train and X_test
    min_rows = min(X_train.shape[1], X_test.shape[1])

    # Update X_train and X_test based on the minimum number of rows
    X_train = X_train[:, :min_rows, :]
    X_test = X_test[:, :min_rows, :]

    return X_train, y_train, X_test, y_test
