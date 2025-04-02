from typing import List

import numpy as np

from time_prediction_model.time_prediction_model_factory.time_prediction_model_params.poisson_time_prediction_model_params import (
    PoissonTimePredictionModelParams,
)


class HawkesTimePredictionModelParams(PoissonTimePredictionModelParams):
    @classmethod
    def from_files_indications(
        cls,
        folder_path: str,
        file_start_registration_time: int,
        file_start_simulation_time: int,
    ) -> "HawkesTimePredictionModelParams":
        order_events = cls._get_order_events(folder_path)

        prefix_filename = cls._get_correct_parameter_file_prefix(
            folder_path, file_start_registration_time, file_start_simulation_time
        )

        params_dict = cls._get_poisson_params_map(folder_path, prefix_filename)

        rho = cls._get_param_from_file(folder_path, prefix_filename, "rho")

        if rho.shape == ():
            rho = np.reshape(rho, (1, 1))

        beta = cls._get_param_from_file(folder_path, prefix_filename, "beta")

        if beta.shape == ():
            beta = np.reshape(beta, (1, 1))

        params_dict.update(
            {"event_types_order": order_events, "rho": rho, "beta": beta}
        )

        return cls(params_dict)

    @classmethod
    def _get_strings_from_file(cls, file_path: str) -> List[str]:
        with open(file_path, "r") as file:
            lines = file.readlines()  # Read all lines into a list
            lines = [
                line.strip() for line in lines
            ]  # Remove trailing newlines or spaces
        return lines
