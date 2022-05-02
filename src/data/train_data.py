import pandas as pd
import logging
import gdown
import os
from ..utils import DataParams

logger = logging.getLogger(__name__)


def get_train_data(params: DataParams) -> pd.pd.DataFrame:
    if params.downloading_data_params.download_data:
        download_dataset(params)
    data_path = os.path.abspath(params.input_data_path)
    try:
        df_train = pd.read_csv(data_path)
    except IOError as e:
        logger.error(msg=f"Unable to open {params.input_data_path}")
        raise e
    return df_train


def download_dataset(params: DataParams):
    logger.info(msg="Downloading data from Google Drive")
    url = params.downloading_data_params.google_drive_url
    output = os.path.abspath(params.input_data_path)
    try:
        gdown.download(url, output, quiet=True)
    except Exception as e:
        logger.error(msg=f"Unable to download data from {url}")
        raise e
