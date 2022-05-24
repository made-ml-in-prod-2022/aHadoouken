import logging
import os
from typing import Tuple
import gdown
import pandas as pd
from sklearn.model_selection import train_test_split


from ml_app.utils import DataParams

logger = logging.getLogger(__name__)


def obtain_data(params: DataParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads data file from external source (if it's necessary),
    loads raw dataset from file to memory
    and returs splitted dataframe (train/test subsets)"""
    if params.downloading_data_params.download_data:
        download_dataset(params)

    dataframe = load_data(params.input_data_path)

    return train_test_split(
        dataframe,
        test_size=params.splitting_params.test_size,
        random_state=params.splitting_params.random_state,
    )


def load_data(data_path: str) -> pd.DataFrame:
    """Loads data from a file"""
    data_path = os.path.abspath(data_path)
    try:
        dataframe = pd.read_csv(data_path)
    except IOError as excepption:
        logger.error(msg=f"Unable to open {data_path}")
        raise excepption
    return dataframe


def download_dataset(params: DataParams) -> None:
    """Downloads data file from external source"""
    logger.info(msg="Downloading data from Google Drive")
    url = params.downloading_data_params.google_drive_url
    output = os.path.abspath(params.input_data_path)
    try:
        gdown.download(url, output, quiet=True)
    except Exception as excepption:
        logger.error(msg=f"Unable to download data from {url}")
        raise excepption
