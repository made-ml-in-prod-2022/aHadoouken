from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from .data_params import DataParams
from .preproc_params import PreprocParams
from .model_params import ModelParams


@dataclass()
class TrainingParams:
    data: DataParams
    preproc: PreprocParams
    model: ModelParams


TrainingParamsSchema = class_schema(TrainingParams)


def read_training_params(path: str) -> TrainingParams:
    with open(path, "r") as input_stream:
        schema = TrainingParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
