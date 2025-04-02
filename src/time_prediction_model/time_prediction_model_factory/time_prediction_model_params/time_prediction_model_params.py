import os
from typing import Any, Dict

from src.constants import ORDERBOOK_CHANGES_FILE_PREFIX


class TimePredictionModelParams:
    def __init__(self, model_params: Dict[str, Any]):
        self._model_params = model_params

    @classmethod
    def _get_correct_parameter_file_prefix(
        cls,
        folder_path: str, 
        file_start_registration_time: int,
        file_start_simulation_time: int
    ) -> str:
        prefix_without_interrupted = (
            ORDERBOOK_CHANGES_FILE_PREFIX +
            str(file_start_registration_time) + "_" +
            str(file_start_simulation_time)
        )

        for file_name in os.listdir(folder_path):
            if file_name.startswith(prefix_without_interrupted):
                return prefix_without_interrupted
            
        prefix_with_interrupted = (
            ORDERBOOK_CHANGES_FILE_PREFIX +
            str(file_start_registration_time) + "_interrupted_" +
            str(file_start_simulation_time)
        )

        for file_name in os.listdir(folder_path):
            if file_name.startswith(prefix_with_interrupted):
                return prefix_with_interrupted
            
        raise ValueError(
            f"Cannot find file with prefix {prefix_without_interrupted} or {prefix_with_interrupted}"
        )