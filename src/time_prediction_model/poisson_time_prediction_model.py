import numpy as np

from time_prediction_model.hawkes_time_prediction_model import HawkesTimePredictionModel


class PoissonTimePredictionModel(HawkesTimePredictionModel):
    def _get_rho_converted_for_simulation(self) -> np.ndarray:
        return np.array([[0]])

    def _get_beta_converted_for_simulation(self) -> np.ndarray:
        return np.array([[0]])
