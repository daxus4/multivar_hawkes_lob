from typing import Dict, List

import numpy as np

from time_prediction_model.period_for_simulation import PeriodForSimulation
from time_prediction_model.time_prediction_model import TimePredictionModel


class NaiveTimePredictionModel(TimePredictionModel):
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
                event_times[-1], current_time, self._parameters["next_event_time_jump"]
            )]
            for event_type, event_times in event_type_event_times_map_to_predict.items()
        }

        return next_event_times
    
    def _get_predicted_event_after_current_time(
        self,
        last_event_time: float,
        current_time: float,
        next_event_time_jump: float
    ) -> float:
        next_event_time = last_event_time + next_event_time_jump
        while next_event_time < current_time:
            next_event_time += next_event_time_jump
        return next_event_time