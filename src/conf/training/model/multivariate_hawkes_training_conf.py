from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class MultivariateHawkesTrainingConf:
    betas_to_train: np.ndarray
    beta_values_to_test: List[float]

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> "MultivariateHawkesTrainingConf":
        return cls(
            betas_to_train=cls.get_nparray_from_list_of_lists(conf["betas_to_train"]),
            beta_values_to_test=conf["beta_values_to_test"],
        )
    
    @classmethod
    def get_nparray_from_list_of_lists(
        cls, list_of_lists: List[List[int]]
    ) -> np.ndarray:
        return np.array(list_of_lists, dtype=int)