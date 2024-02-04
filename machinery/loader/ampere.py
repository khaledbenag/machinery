from numpy import ndarray
from pandas import DataFrame

from machinery.loader.base import load_data, load_metadata, load_split_data


def load_ampere_rotor_metadata(data_dir: str = None) -> tuple[DataFrame, dict]:
    metadata_df, class_mapping = load_metadata(data_dir, "ampere_rotor")
    return metadata_df, class_mapping


def load_ampere_stator_metadata(data_dir: str = None) -> tuple[DataFrame, dict]:
    metadata_df, class_mapping = load_metadata(data_dir, "ampere_stator")
    return metadata_df, class_mapping


def load_ampere_rotor_data(
    ampere_rotor_metadata_df: DataFrame,
) -> tuple[ndarray, ndarray]:
    ampere_rotor_type = "ampere_rotor"
    ampere_rotor_data, ampere_rotor_target = load_data(
        ampere_rotor_metadata_df, ampere_rotor_type
    )
    return ampere_rotor_data, ampere_rotor_target


def load_ampere_stator_data(
    ampere_stator_metadata_df: DataFrame,
) -> tuple[ndarray, ndarray]:
    ampere_stator_type = "ampere_stator"
    ampere_stator_data, ampere_stator_target = load_data(
        ampere_stator_metadata_df, ampere_stator_type
    )
    return ampere_stator_data, ampere_stator_target


def load_split_ampere_rotor_data(
    ampere_rotor_train_df: DataFrame, ampere_rotor_test_df: DataFrame
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    ampere_rotor_type = "ampere_rotor"
    X_train, y_train, X_test, y_test = load_split_data(
        ampere_rotor_train_df, ampere_rotor_test_df, ampere_rotor_type
    )
    return X_train, y_train, X_test, y_test


def load_split_ampere_stator_data(
    ampere_stator_train_df: DataFrame, ampere_stator_test_df: DataFrame
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    ampere_stator_type = "ampere_stator"
    X_train, y_train, X_test, y_test = load_split_data(
        ampere_stator_train_df, ampere_stator_test_df, ampere_stator_type
    )
    return X_train, y_train, X_test, y_test
