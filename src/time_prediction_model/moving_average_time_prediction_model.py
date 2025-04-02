from typing import Any, Dict, List

import numpy as np

from time_prediction_model.period_for_simulation import PeriodForSimulation
from time_prediction_model.time_prediction_model import TimePredictionModel


class MovingAverageTimePredictionModel(TimePredictionModel):
    #def __init__(
    #    self,
    #    params: Dict[str, Any],
    #    prediction_period_duration: float,
    #) -> None:
    #    super().__init__(params)
    #    self._prediction_period_duration = prediction_period_duration

    def predict_next_event_time_from_current_time(
        self,
        period_for_simulation: PeriodForSimulation,
        current_time: float,
    ) -> Dict[str, float]:
        event_type_event_times_map_to_predict = (
            period_for_simulation.get_event_type_event_times_map_to_predict()
        )

        next_event_times = {
            event_type: [self._get_predicted_event_after_current_time(
                event_times, current_time, self._parameters["window_duration_seconds"]
            )]
            for event_type, event_times in event_type_event_times_map_to_predict.items()
        }

        return next_event_times

    def _get_predicted_event_after_current_time(
        self,
        event_times: np.ndarray,
        current_time: float,
        window_duration_seconds: float
    ) -> float:
        event_times_in_window = event_times[
            event_times >= current_time - window_duration_seconds
        ]

        mean_next_event_time_jump = self._get_mean_next_event_time_jump(
            event_times_in_window
        )

        next_event_time = event_times_in_window[-1] + mean_next_event_time_jump

        while next_event_time < current_time:
            next_event_time += mean_next_event_time_jump

        return next_event_time

    def _get_mean_next_event_time_jump(self, event_times: np.ndarray) -> float:
        return np.mean(np.diff(event_times))