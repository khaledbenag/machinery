import os
from urllib.request import urlretrieve
from zipfile import ZipFile

from loguru import logger
from tqdm import tqdm

from machinery.dataset.variables import (
    AMPERE_ROTOR_BASE_FOLDER_NAME,
    AMPERE_STATOR_BASE_FOLDER_NAME,
    AMPERE_URL,
    LASPI_BASE_FOLDER_NAME,
    LASPI_URL,
    METALLICADOUR_DRIFTS_BASE_FOLDER_NAME,
    METALLICADOUR_TOOLWEAR_BASE_FOLDER_NAME,
    METALLICADOUR_URL,
)

get_url = {
    "laspi": LASPI_URL,
    "ampere_rotor": AMPERE_URL,
    "ampere_stator": AMPERE_URL,
    "metallicadour_drifts": METALLICADOUR_URL,
    "metallicadour_toolwear": METALLICADOUR_URL,
}

get_base_folder = {
    "laspi": LASPI_BASE_FOLDER_NAME,
    "ampere_rotor": AMPERE_ROTOR_BASE_FOLDER_NAME,
    "ampere_stator": AMPERE_STATOR_BASE_FOLDER_NAME,
    "metallicadour_drifts": METALLICADOUR_DRIFTS_BASE_FOLDER_NAME,
    "metallicadour_toolwear": METALLICADOUR_TOOLWEAR_BASE_FOLDER_NAME,
}


def download_data(data_type: str = None):
    if data_type is None:
        raise Exception("data_type must be defined")
    default_path = os.path.join(os.getcwd(), "data")
    if data_type in ["ampere_rotor", "ampere_stator"]:
        extract_folder = "ampere_extracted_data"
    elif data_type in ["metallicadour_drifts", "metallicadour_toolwear"]:
        extract_folder = "metallicadour_extracted_data"
    else:
        extract_folder = f"{data_type}_extracted_data"
    extracted_folder = os.path.join(default_path, extract_folder)
    base_folder_name = get_base_folder[data_type]
    data_path = os.path.join(extracted_folder, base_folder_name)

    # Check if data already exists at the local path
    if os.path.exists(data_path):
        logger.info(f"Data already exists at {data_path}.")
        return data_path

    try:
        os.makedirs(default_path, exist_ok=True)
        download_url = get_url[data_type]
        logger.info(f"Downloading {data_type} dataset to %s ...", default_path)

        # Use tqdm to show the progress bar
        with tqdm(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=f"Downloading {data_type} (path = {default_path} ",
        ) as t:
            urlretrieve(
                download_url,
                os.path.join(default_path, f"{data_type}.zip"),
                reporthook=lambda b, bsize, tsize: t.update(b * bsize - t.n),
            )

        # Extract the downloaded ZIP file
        zip_file_path = os.path.join(default_path, f"{data_type}.zip")

        with ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extracted_folder)

        logger.info("Extraction complete.")

        # Cleanup the ZIP file
        os.remove(zip_file_path)

        return data_path

    except KeyboardInterrupt:
        logger.error("Download or extraction interrupted by user.")
        # Cleanup in case of interruption
        cleanup(default_path)

    except Exception as error:
        logger.error(f"Error when downloading or extracting {data_type} data: {error}")

        # Cleanup in case of error
        cleanup(default_path)


def cleanup(local_path):
    if os.path.exists(local_path):
        logger.info("Cleaning up...")
        for file_or_dir in os.listdir(local_path):
            path = os.path.join(local_path, file_or_dir)
            if os.path.isdir(path):
                os.rmdir(path)
            else:
                os.remove(path)
        os.rmdir(local_path)
        logger.info("Cleanup complete.")
