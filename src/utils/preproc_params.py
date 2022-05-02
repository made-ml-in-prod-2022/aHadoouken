from dataclasses import dataclass, field
from typing import List

@dataclass
class SplittingParams:
    test_size: float = field(default=0.2)
    random_state: int = field(default=42)

@dataclass
class PreprocParams:
    categorical_features: List[str]
    target_col: str
    splitting_params: SplittingParams
