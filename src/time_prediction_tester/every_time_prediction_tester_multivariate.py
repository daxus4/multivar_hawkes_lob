from typing import Dict

import numpy as np
import pandas as pd

from time_prediction_model.hawkes_time_prediction_model import HawkesTimePredictionModel
from time_prediction_model.period_for_simulation import PeriodForSimulation
from time_prediction_tester.time_prediction_tester import TimePredictionTester


class EveryTimePredictionTesterMultivariate(TimePredictionTester):
    def __init__(
        self,
        hawkes_trained_model: HawkesTimePredictionModel,
        period_for_simulation: PeriodForSimulation,
        warmup_time_duration: float,
    ):
        super().__init__(
            hawkes_trained_model, period_for_simulation, warmup_time_duration
        )

        self._simulation_event_dfs = period_for_simulation.get_simulation_dataframes(
            warmup_time_duration
        )

    def get_predicted_event_times(self) -> Dict[str, np.ndarray]:
        return None

    def _get_empty_event_type_predicted_event_times_map(self) -> Dict[str, np.ndarray]:
        return {
            event_type: np.zeros(len(real_event_times))
            for event_type, real_event_times in self._event_type_real_event_times_map.items()
        }

    def get_event_type_real_event_times_map(self) -> Dict[str, np.ndarray]:
        return None

    def get_multivariate_predictions(self) -> pd.DataFrame:
        prediction_df = {
            "Real Event Time": [],
            "Real Event Type": [],
            "Predicted MP UP Time": [],
            "Predicted MP DOWN Time": [],
            "Real Time Previous Event": [],
            "Real Type Previous Event": [],
        }

        for simulation_df in self._simulation_event_dfs:
            simulation_df = simulation_df.copy()

            warmup_df = simulation_df.iloc[:-1]
            current_event_type = simulation_df.iloc[-1]["Event Type"]
            current_time = simulation_df.iloc[-1]["Time"]
            warmup_start_time = warmup_df.iloc[0]["Time"]
            real_time_previous_event = warmup_df.iloc[-2]["Time"]
            real_type_previous_event = warmup_df.iloc[-2]["Event Type"]

            period_for_simulation = (
                self._period_for_simulation.get_period_between_times(
                    warmup_start_time, current_time
                )
            )

            predictions = self._model.predict_next_event_time_from_current_time(
                period_for_simulation, current_time
            )

            prediction_df["Real Event Time"].append(current_time)
            prediction_df["Real Event Type"].append(current_event_type)
            prediction_df["Predicted MP UP Time"].append(predictions["MID_PRICE_UP"][0])
            prediction_df["Predicted MP DOWN Time"].append(
                predictions["MID_PRICE_DOWN"][0]
            )
            prediction_df["Real Time Previous Event"].append(real_time_previous_event)
            prediction_df["Real Type Previous Event"].append(real_type_previous_event)

        prediction_df = pd.DataFrame(prediction_df).sort_values("Real Event Time")
        prediction_df = prediction_df[prediction_df["Real Event Time"] > self._warmup_time_duration].reset_index(drop=True)

        return prediction_df
