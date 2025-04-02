from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from time_prediction_model.period_for_simulation import PeriodForSimulation
from time_prediction_model.time_prediction_model import TimePredictionModel


class TimePredictionTester(ABC):
    def __init__(
        self,
        trained_model: TimePredictionModel,
        period_for_simulation: PeriodForSimulation,
        warmup_time_duration: float,
    ):
        self._model = trained_model
        self._period_for_simulation = period_for_simulation
        self._warmup_time_duration = warmup_time_duration

    @abstractmethod
    def get_predicted_event_times(self) -> np.ndarray:
        raise NotImplementedError
    
    def _get_next_predicted_event_time_from_current_time(
        self,
        current_time: float,
    ) -> Dict[str, float]:
        start_warmup_time = current_time - self._warmup_time_duration

        period_for_simulation_from_warmup_to_current_time = (
            self._period_for_simulation.get_period_between_times(
                start_warmup_time,
                current_time
            )
        )

        event_type_predictions_map = self._model.predict_next_event_time_from_current_time(
            period_for_simulation_from_warmup_to_current_time,
            current_time
        )

        return {
            event_type: event_time[0] if len(event_time) > 0 else np.nan
            for event_type, event_time
            in event_type_predictions_map.items()
        }

    @abstractmethod
    def get_event_type_real_event_times_map(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError