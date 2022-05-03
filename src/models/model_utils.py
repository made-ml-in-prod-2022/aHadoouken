import logging
import os
from typing import Union
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.utils import ModelParams

logger = logging.getLogger(__name__)


def get_model(
    params: ModelParams,
) -> Union[RandomForestClassifier, LogisticRegression]:
    if params.model_type == "RandomForestClassifier":
        logger.info(msg="Choosing RandomForest model")
        model = RandomForestClassifier(**params.rf_params.__dict__)
    elif params.model_type == "LogisticRegression":
        logger.info(msg="Choosing LogisticRegression model")
        model = LogisticRegression(**params.lr_params.__dict__)
    else:
        raise ValueError(f"Invalid model type: {params.model_type}")
    return model


def save_model(model: Pipeline, path: str) -> None:
    model_path = os.path.abspath(path)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(path: str) -> Pipeline:
    model_path = os.path.abspath(path)
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def eval_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    return metrics


def save_metrics(metrics: dict, path: str) -> None:
    metrics_path = os.path.abspath(path)
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=6)


def save_predict(predict: np.ndarray, path: str) -> None:
    path = os.path.abspath(path)
    np.savetxt(path, predict)
