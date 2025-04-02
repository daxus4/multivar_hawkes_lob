from typing import Any, Dict, List

import numpy as np
import tick.hawkes as hk

from time_prediction_model.period_for_simulation import PeriodForSimulation
from time_prediction_model.time_prediction_model import TimePredictionModel


class HawkesTimePredictionModel(TimePredictionModel):
    def __init__(
        self,
        params: Dict[str, Any],
        prediction_period_duration: float,
        seed: int = 1039,
    ) -> None:
        super().__init__(params)
        # self._parameters["alpha"] = self._parameters["alpha"] / self._parameters["beta"]
        self._prediction_period_duration = prediction_period_duration
        self._seed = seed

    def predict_next_event_time_from_current_time(
        self,
        period_for_simulation: PeriodForSimulation,
        current_time: float,
    ) -> Dict[str, float]:
        event_times = period_for_simulation.get_ordered_event_times(
            self._parameters["event_types_order"]
        )

        simulated_hawkes = self._get_hawkes_simulation(
            event_times, current_time, self._seed
        )

        predicted_timestamps = self._get_new_simulated_timestamps(
            simulated_hawkes,
            current_time,
        )

        return {
            event_type: predicted_timestamps[i]
            for i, event_type in enumerate(self._parameters["event_types_order"])
        }

    def _get_hawkes_simulation(
        self,
        event_times: List[np.ndarray],
        current_time: float,
        seed: int,
    ) -> hk.SimuHawkesExpKernels:
        sim_hawkes = hk.SimuHawkesExpKernels(
            adjacency=self._get_rho_converted_for_simulation(),
            decays=self._get_beta_converted_for_simulation(),
            baseline=self._get_mu_converted_for_simulation(),
            end_time=current_time,
            seed=seed,
            verbose=False,
        )
        sim_hawkes.track_intensity(1)

        sim_hawkes.set_timestamps(event_times, current_time)
        sim_hawkes.end_time = current_time + self._prediction_period_duration

        sim_hawkes.simulate()

        return sim_hawkes

    def _get_new_simulated_timestamps(
        self,
        simulated_hawkes: hk.SimuHawkesExpKernels,
        current_time: float,
    ) -> np.ndarray:
        new_simulated_timestamps = [
            np.array([t for t in timestamps if t > current_time])
            for timestamps in simulated_hawkes.timestamps
        ]

        # add max_simulated_time to timestamps that are empty
        for i, timestamps in enumerate(new_simulated_timestamps):
            if len(timestamps) == 0:
                new_simulated_timestamps[i] = np.array(
                    [current_time + self._prediction_period_duration]
                )

        return new_simulated_timestamps

    def _get_rho_converted_for_simulation(self) -> np.ndarray:
        return self._parameters["rho"]

    def _get_beta_converted_for_simulation(self) -> np.ndarray:
        return self._parameters["beta"]

    def _get_mu_converted_for_simulation(self) -> float:
        return self._parameters["mu"]
