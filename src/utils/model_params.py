from dataclasses import dataclass, field

@dataclass
class RFParams:
    n_estimators: int = field(default=300)
    max_depth: int = field(default=6)
    random_state: int = field(default=42)

@dataclass
class LRParams:
    max_iter: int = field(default=1000)
    random_state: int = field(default=42)

@dataclass
class ModelParams:
    output_model_path: str
    metric_path: str
    rf_params: RFParams = field(default=RFParams())
    lr_params: LRParams = field(default=LRParams())
    model_type: str = field(default="RandomForestClassifier")
