from dataclasses import dataclass
import logging
import pprint
from typing import Union
from marshmallow_dataclass import class_schema
from marshmallow.schema import Schema
import yaml

from .data_params import DataParams
from .preproc_params import PreprocParams
from .model_params import ModelParams

logger = logging.getLogger(__name__)


@dataclass()
class TrainingParams:
    data: DataParams
    preproc: PreprocParams
    model: ModelParams


@dataclass()
class PredictParams:
    model_path: str
    data_path: str
    results_path: str


TrainingParamsSchema = class_schema(TrainingParams)
PredictParamsSchema = class_schema(PredictParams)


def read_params(
    cl_schema: Schema, path: str
) -> Union[PredictParams, TrainingParams]:
    with open(path, "r") as input_stream:
        config_params = yaml.safe_load(input_stream)
        params_str = pprint.pformat(config_params)
        logger.debug(msg=f"Config params:\n{params_str}")
        schema = cl_schema()

        return schema.load(config_params)


def read_training_params(path: str) -> TrainingParams:
    return read_params(TrainingParamsSchema, path)


def read_predict_params(path: str) -> PredictParams:
    return read_params(PredictParamsSchema, path)
