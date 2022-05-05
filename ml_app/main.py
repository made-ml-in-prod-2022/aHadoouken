import logging
import sys
import json
import click
from sklearn.pipeline import Pipeline

from ml_app.utils import read_training_params, read_predict_params
from ml_app.data import obtain_data, load_data
from ml_app.features import DataTransformer
from ml_app.models import (
    get_model,
    eval_model,
    save_model,
    load_model,
    save_metrics,
    save_predict,
)


logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def train_model_pipeline(config):
    """Train ML model
    this method:
    1. Obtains raw dataset
    2. Creates model pipeline
    3. Trains and evaluate pipeline model
    4. Saves trained model to file

    Keyword arguments:
    config -- path to config file
    """
    logger.info(msg="Starting training model pipeline")
    params = read_training_params(config)

    df_train, df_test = obtain_data(params.data)
    logger.info(msg="Data was obtained")
    logger.debug(msg="Obtained train data:\n}")
    logger.info(f"train_df.shape is {df_train.shape}")
    logger.info(f"val_df.shape is {df_test.shape}")

    x_train = df_train.drop(params.preproc.target_col, axis=1, inplace=False)
    y_train = df_train[params.preproc.target_col]
    x_test = df_test.drop(params.preproc.target_col, axis=1, inplace=False)
    y_test = df_test[params.preproc.target_col]

    data_transformer = DataTransformer(
        params.preproc.categorical_features, params.preproc.numerical_features
    )
    model = get_model(params.model)

    pipe = Pipeline(
        [("data_transformer", data_transformer), ("classifier", model)]
    )

    pipe.fit(x_train, y_train)
    metrics = eval_model(pipe, x_test, y_test)
    save_metrics(metrics, params.model.metric_path)
    logger.info(msg="Model was trained and evaluated.")
    logger.info(msg="Model metrics:")
    logger.info(msg=json.dumps(metrics, indent=6))
    save_model(pipe, params.model.output_model_path)
    logger.info(msg=f"Model was saved to: {params.model.output_model_path}")


def predict_model_pipeline(config):
    """Predict ML model
    this method:
    1. Obtains raw dataset
    2. Loads pipeline model from file
    3. Predicts values using loaded pipeline
    4. Saves results to specified file

    Keyword arguments:
    config -- path to config file
    """
    logger.info(msg="Starting predicting model pipeline")
    params = read_predict_params(config)

    model = load_model(params.model_path)
    logger.debug(msg=f"Model was loaded from {params.model_path}")
    data = load_data(params.data_path)
    logger.debug(msg=f"Data was loaded from {params.data_path}")
    y_pred = model.predict(data)
    save_predict(y_pred, params.results_path)
    logger.info(msg="Prediction is done!")
    logger.info(msg=f"Results were saved to: {params.results_path}")


@click.command()
@click.option("--mode", required=True, type=str, help="train/predict")
@click.option(
    "--config", required=True, type=str, help="path to config (for train mode)"
)
def main(mode, config):
    if mode == "train":
        train_model_pipeline(config)
    elif mode == "predict":
        predict_model_pipeline(config)
    else:
        raise ValueError(f"Invalid mode type {mode}")


if __name__ == "__main__":
    main()
