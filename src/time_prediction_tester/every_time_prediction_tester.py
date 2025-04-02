from typing import Dict

import numpy as np

from time_prediction_model.period_for_simulation import PeriodForSimulation
from time_prediction_model.time_prediction_model import TimePredictionModel
from time_prediction_tester.time_prediction_tester import TimePredictionTester


class EveryTimePredictionTester(TimePredictionTester):
    """
    When period_for_simulation contains more event types to predict than one, this class
    predicts the next event time for each event type considering as starting
    point the last event time of the event type. So it is like it does it
    for each event type separately. And also it is not optimized for this case.
    """
    def __init__(
        self,
        trained_model: TimePredictionModel,
        period_for_simulation: PeriodForSimulation,
        warmup_time_duration: float,
    ):
        super().__init__(
            trained_model, period_for_simulation, warmup_time_duration
        )
        
        self._event_type_real_event_times_map = self._get_event_type_real_event_times_map()

    def _get_event_type_real_event_times_map(self) -> Dict[str, np.ndarray]:
        period_for_simulation_from_warmup = self._period_for_simulation.get_period_from_time(
            self._warmup_time_duration
        )

        return period_for_simulation_from_warmup.get_event_type_event_times_map_to_predict()

    def _get_shifted_event_type_real_event_times_map(self) -> Dict[str, np.ndarray]:
        period_for_simulation_to_predict = (
            self._period_for_simulation.get_event_type_event_times_map_to_predict()
        )

        real_event_type, real_event_times = next(
            iter(period_for_simulation_to_predict.items())
        )

        index = np.searchsorted(
            real_event_times, self._warmup_time_duration, side="left"
        ) - 1
        result = real_event_times[index:-1]

        return {
            real_event_type: result
        }

    def get_predicted_event_times(self) -> Dict[str, np.ndarray]:
        event_type_predicted_event_times_map = self._get_empty_event_type_predicted_event_times_map()
        
        for event_type, real_event_times in self._get_shifted_event_type_real_event_times_map().items():
            for i, event_time in enumerate(real_event_times):
                predicted_event_times = self._get_next_predicted_event_time_from_current_time(
                    event_time
                )
                event_type_predicted_event_times_map[event_type][
                    i
                ] = predicted_event_times[event_type] - self._warmup_time_duration
            
            return event_type_predicted_event_times_map

    def _get_empty_event_type_predicted_event_times_map(self) -> Dict[str, np.ndarray]:
        return {
            event_type: np.zeros(len(real_event_times))
            for event_type, real_event_times
            in self._event_type_real_event_times_map.items()
        }

    
    def get_event_type_real_event_times_map(self) -> Dict[str, np.ndarray]:
        return {
            event_type: (
                self._event_type_real_event_times_map[event_type].copy()
                - self._warmup_time_duration
            )
            for event_type in self._event_type_real_event_times_map
        }