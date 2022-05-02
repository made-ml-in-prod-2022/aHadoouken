import click
from sklearn.pipeline import Pipeline
import logging
import sys
import json
from src.models.model_utils import save_model

from src.utils import read_training_params
from src.data import obtain_data, load_data
from src.features import DataTransformer
from src.models import (
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
    logger.info(msg="Starting training model pipeline")
    params = read_training_params(config)

    df_train, df_test = obtain_data(params.data)
    logger.info(msg="Data was obtained")
    logger.debug(msg=f"Obtained train data:\n{df_train.head()}")
    logger.info(f"train_df.shape is {df_train.shape}")
    logger.info(f"val_df.shape is {df_test.shape}")

    X_train = df_train.drop(params.preproc.target_col, axis=1, inplace=False)
    y_train = df_train[params.preproc.target_col]
    X_test = df_test.drop(params.preproc.target_col, axis=1, inplace=False)
    y_test = df_test[params.preproc.target_col]

    data_transformer = DataTransformer(
        params.preproc.categorical_features, params.preproc.numerical_features
    )
    model = get_model(params.model)

    pipe = Pipeline(
        [("data_transformer", data_transformer), ("classifier", model)]
    )

    pipe.fit(X_train, y_train)
    metrics = eval_model(pipe, X_test, y_test)
    save_metrics(metrics, params.model.metric_path)
    logger.info(msg="Model was trained and evaluated.")
    logger.info(msg="Model metrics:")
    logger.info(msg=json.dumps(metrics, indent=6))
    save_model(pipe, params.model.output_model_path)
    logger.info(msg=f"Model was saved to: {params.model.output_model_path}")


def predict_model_pipeline(model_path, data_path, results_path):
    logger.info(msg="Starting predicting model pipeline")
    model = load_model(model_path)
    data = load_data(data_path)
    y_pred = model.predict(data)
    save_predict(y_pred, results_path)
    logger.info(msg="Prediction is done!")
    logger.info(msg=f"Results were saved to: {results_path}")


@click.command()
@click.option("--mode", required=True, type=str, help="train/predict")
@click.option("--config", type=str, help="path to config (for train mode)")
@click.option("--model", type=str, help="path to model (for predict mode)")
@click.option("--data", type=str, help="path to data (for predict mode)")
@click.option(
    "--results", type=str, help="path to predicted results (for predict mode)"
)
def main(mode, config, model, data, results):
    if mode == "train":
        if config is None:
            logger.error(msg="Config file is not specified")
            exit(0)
        if model is not None or data is not None or results is not None:
            logger.warning(
                msg="Program mode is 'train', model/data/results flags will be ignored"
            )
        train_model_pipeline(config)

    elif mode == "predict":
        if model is None or data is None or results is None:
            logger.error(msg="Not all flags were specified for predict mode")
            exit(0)
        if config is not None:
            logger.warning(
                msg="Program mode is 'predict', config flag will be ignored"
            )
        predict_model_pipeline(model, data, results)
    else:
        raise ValueError(f"Invalid mode type {mode}")


if __name__ == "__main__":
    main()
