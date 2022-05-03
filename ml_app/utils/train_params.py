from dataclasses import dataclass
import logging
import pprint
from marshmallow_dataclass import class_schema
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


TrainingParamsSchema = class_schema(TrainingParams)


def read_training_params(path: str) -> TrainingParams:
    with open(path, "r") as input_stream:
        config_params = yaml.safe_load(input_stream)
        params_str = pprint.pformat(config_params)
        logger.debug(msg=f"Config params:\n{params_str}")
        schema = TrainingParamsSchema()

        return schema.load(config_params)
