from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class TimePredictionModel(ABC):
    def __init__(self, params: Dict[str, Any]) -> None:
        self._parameters = params


    def predict_next_event_time_from_last_event(
        self, event_times: List[np.ndarray], **kwargs
    ) -> float:
        return self.predict_next_event_time_from_current_time(
            event_times, event_times[-1], **kwargs
        )

    @abstractmethod
    def predict_next_event_time_from_current_time(
        self, event_times: List[np.ndarray], current_time: float, **kwargs
    ) -> Dict[str, float]:
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters