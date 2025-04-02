from typing import Dict, List

import numpy as np
import pandas as pd


class PeriodForSimulation:
    def __init__(
        self,
        event_type_event_times_map: Dict[str, np.ndarray],
        event_types_to_predict: List[str],
        event_types_order: List[str],
    ) -> None:
        self._event_type_event_times_map = event_type_event_times_map
        self._event_types_to_predict = event_types_to_predict
        self._event_types_order = sorted(event_types_order)

        self._dataframe = self._get_dataframe()

    def _get_dataframe(self) -> pd.DataFrame:
        data = [
            (event, timestamp)
            for event, times in self._event_type_event_times_map.items()
            for timestamp in times
        ]

        df = pd.DataFrame(data, columns=["Event Type", "Time"])
        df = df.sort_values(by="Time").reset_index(drop=True)

        return df

    def get_simulation_dataframes(
        self, warmup_window_size: float
    ) -> List[pd.DataFrame]:
        results = []

        start_time_initial = self._dataframe["Time"].iloc[0] + warmup_window_size

        for event_type in self._event_types_to_predict:
            event_rows = self._dataframe[
                (self._dataframe["Event Type"] == event_type)
                & (self._dataframe["Time"] >= start_time_initial)
            ]

            for _, event in event_rows.iterrows():
                start_time = event["Time"] - warmup_window_size
                time_window_df = self._dataframe[
                    (self._dataframe["Time"] >= start_time)
                    & (self._dataframe["Time"] <= event["Time"])
                ]

                # count how many events there are at the time of the event
                event_count = time_window_df[time_window_df["Time"] == event["Time"]].shape[0]
                if event_count > 1:
                    # keep only the event between them at the correct type
                    time_window_df = time_window_df[
                        (time_window_df["Event Type"] == event_type)
                        | (time_window_df["Time"] < event["Time"])
                    ]
                results.append(time_window_df)

        return results

    @property
    def event_types_order(self) -> List[str]:
        return self._event_types_order.copy()

    def get_ordered_event_times(self) -> List[np.ndarray]:
        return [
            self._event_type_event_times_map[event_type]
            for event_type in self._event_types_order
        ]

    def get_ordered_event_times(self, event_types_order: List[str]) -> List[np.ndarray]:
        return [
            self._event_type_event_times_map[event_type]
            for event_type in event_types_order
        ]

    def get_period_from_time(self, time: float) -> "PeriodForSimulation":
        return PeriodForSimulation(
            {
                event_type: event_times[event_times >= time]
                for event_type, event_times in self._event_type_event_times_map.items()
            },
            self._event_types_to_predict,
            self._event_types_order,
        )

    def get_period_to_time(self, time: float) -> "PeriodForSimulation":
        return PeriodForSimulation(
            {
                event_type: event_times[event_times <= time]
                for event_type, event_times in self._event_type_event_times_map.items()
            },
            self._event_types_to_predict,
            self._event_types_order,
        )

    def get_period_between_times(
        self,
        start_time: float,
        end_time: float,
    ) -> "PeriodForSimulation":
        return PeriodForSimulation(
            {
                event_type: event_times[
                    (event_times >= start_time) & (event_times <= end_time)
                ]
                for event_type, event_times in self._event_type_event_times_map.items()
            },
            self._event_types_to_predict,
            self._event_types_order,
        )

    def get_event_type_event_times_map_to_predict(self) -> Dict[str, np.ndarray]:
        return {
            event_type: self._event_type_event_times_map[event_type]
            for event_type in self._event_types_to_predict
        }

    def get_sorted_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                event_type: self._event_type_event_times_map[event_type]
                for event_type in self._event_types_order
            }
        )
