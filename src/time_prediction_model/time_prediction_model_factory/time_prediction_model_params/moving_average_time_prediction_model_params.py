import os

from time_prediction_model.time_prediction_model_factory.time_prediction_model_params.time_prediction_model_params import (
    TimePredictionModelParams,
)


class MovingAverageTimePredictionModelParams(TimePredictionModelParams):
    @classmethod
    def from_files_indications(
        cls,
        folder_path: str,
        file_start_registration_time: int,
        file_start_simulation_time: int
    ) -> 'MovingAverageTimePredictionModelParams':
        prefix_filename = cls._get_correct_parameter_file_prefix(
            folder_path,
            file_start_registration_time,
            file_start_simulation_time
        )

        window_duration_seconds = cls.get_float_from_file(
            os.path.join(folder_path, prefix_filename + "_window_duration_seconds.txt")
        )

        return cls({
            "window_duration_seconds": window_duration_seconds
        })
        
    @classmethod
    def get_float_from_file(
        cls, file_path: str
    ) -> float:
        with open(file_path, 'r') as file:
            content = file.read().strip()  # Read the content and remove any extra spaces/newlines
            return float(content)
