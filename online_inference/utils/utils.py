import os
from typing import List, Union
from enum import Enum

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from pydantic import BaseModel, conlist


class ModelType(str, Enum):
    lr = "logistic_regression"
    rf = "random_forest"


class InputDataModel(BaseModel):
    data: List[conlist(Union[float, int], min_items=13, max_items=13)]
    features_names: conlist(str, min_items=13, max_items=13)
    model: ModelType = ModelType.lr


class ModelResponse(BaseModel):
    predicted_values: List[int]


def load_model(path: str) -> Pipeline:
    """Loads model from disk"""
    model_path = os.path.abspath(path)
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def obtain_df(input_data: InputDataModel) -> pd.DataFrame:
    """Transforms data model to DataFrame"""
    data = pd.DataFrame(
        data=input_data.data,
        columns=input_data.features_names
    )
    return data
