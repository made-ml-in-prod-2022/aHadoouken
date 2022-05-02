import click
from dataclasses import dataclass
import logging
import sys

from utils import read_training_params
from data import get_train_data


logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def train_model(config):
    params = read_training_params(config)
    print(params)
    logger.info(msg="1. Starting data obtaining step")
    df_train = get_train_data(params.data)
    logger.info(msg="Data was obtained")
    print(df_train)



@click.command()
@click.option("--mode", required=True, type=str, help="train/predict")
@click.option("--config", type=str, help="path to config (for train mode)")
@click.option("--model", type=str, help="path to model (for predict mode)")
@click.option("--data", type=str, help="path to data (for predict mode)")
def main(mode, config, model, data):
    if mode == "train":
        if config is None:
            logger.error(msg="Config file is not specified")
            exit(0)
        if model is not None or data is not None:
            logger.warning(
                msg="Program mode is 'train', model/data flags will be ignored"
            )
        train_model(config)


if __name__ == "__main__":
    main()
