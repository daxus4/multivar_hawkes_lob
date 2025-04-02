from typing import List

import numpy as np
from tick.hawkes import HawkesExpKern


class MultivariateHawkesTrainerWithGreedyBetaSearch:
    def __init__(
        self, event_types_periods: List[List[np.ndarray]], betas_to_train: np.ndarray
    ) -> None:
        self._event_types_periods = event_types_periods
        self._betas_to_train = betas_to_train

    def get_trained_kernel(
        self, beta_values_to_test: List[float], beta_for_null_alpha: float = 1000
    ) -> HawkesExpKern:
        current_betas_matrix = self._get_starting_betas_matrix(
            beta_values_to_test[0], beta_for_null_alpha
        )

        current_hawkes_kernel = self._get_hawkes_exp_kernel(
            current_betas_matrix, self._event_types_periods[0]
        )
        best_score = current_hawkes_kernel.score()
        improvement = True

        while improvement:
            improvement = False

            for i in range(self._betas_to_train.shape[0]):
                for j in range(self._betas_to_train.shape[1]):
                    current_beta = current_betas_matrix[i, j]

                    if current_beta < beta_for_null_alpha:
                        best_local_decay = current_beta
                        best_local_score = best_score
                        best_local_kernel = current_hawkes_kernel
                        for decay in beta_values_to_test:
                            current_betas_matrix[i, j] = decay
                            current_hawkes_kernel = self._get_hawkes_exp_kernel(
                                current_betas_matrix, self._event_types_periods
                            )
                            score = current_hawkes_kernel.score()
                            if score > best_local_score:
                                best_local_score = score
                                best_local_kernel = current_hawkes_kernel
                                best_local_decay = decay

                        current_betas_matrix[i, j] = best_local_decay
                        if best_local_score > best_score:
                            best_score = best_local_score
                            best_hawkes_kernel = best_local_kernel
                            improvement = True

        return best_hawkes_kernel

    def _get_hawkes_exp_kernel(
        self, beta_matrix: np.ndarray, timestamps: List[List[np.ndarray]]
    ) -> HawkesExpKern:
        kernel = HawkesExpKern(beta_matrix, gofit="likelihood", penalty="l1")
        kernel.fit(timestamps)

        return kernel

    def _get_starting_betas_matrix(
        self, starting_beta: float, beta_for_null_alpha: float = 1000
    ) -> np.ndarray:
        decays_matrix = np.zeros(self._betas_to_train.shape)

        for i in range(self._betas_to_train.shape[0]):
            for j in range(self._betas_to_train.shape[1]):
                decays_matrix[i][j] = (
                    beta_for_null_alpha
                    if self._betas_to_train[i][j] == 0
                    else starting_beta
                )

        return decays_matrix
