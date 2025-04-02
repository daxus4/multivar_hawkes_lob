import os

from time_prediction_model.time_prediction_model_factory.time_prediction_model_params.time_prediction_model_params import (
    TimePredictionModelParams,
)


class NaiveTimePredictionModelParams(TimePredictionModelParams):
    @classmethod
    def from_files_indications(
        cls,
        folder_path: str,
        file_start_registration_time: int,
        file_start_simulation_time: int
    ) -> 'NaiveTimePredictionModelParams':
        prefix_filename = cls._get_correct_parameter_file_prefix(
            folder_path,
            file_start_registration_time,
            file_start_simulation_time
        )

        next_event_time_jump = cls.get_float_from_file(
            os.path.join(folder_path, prefix_filename + "_next_event_time_jump.txt")
        )

        return cls({
            "next_event_time_jump": next_event_time_jump
        })
        
    @classmethod
    def get_float_from_file(
        cls, file_path: str
    ) -> float:
        with open(file_path, 'r') as file:
            content = file.read().strip()  # Read the content and remove any extra spaces/newlines
            return float(content)
