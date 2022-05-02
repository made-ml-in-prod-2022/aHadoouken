from dataclasses import dataclass, field


@dataclass
class ModelParams:
    output_model_path: str
    metric_path: str
    model_type: str = field(default="RandomForestClassifier")
