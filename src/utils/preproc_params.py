from dataclasses import dataclass, field
from typing import List



@dataclass
class PreprocParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str
    
