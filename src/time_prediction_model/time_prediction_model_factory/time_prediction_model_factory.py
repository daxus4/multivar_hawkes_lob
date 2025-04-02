import src.constants as CONST
from time_prediction_model.hawkes_time_prediction_model import HawkesTimePredictionModel
from time_prediction_model.moving_average_time_prediction_model import (
    MovingAverageTimePredictionModel,
)
from time_prediction_model.naive_time_prediction_model import NaiveTimePredictionModel
from time_prediction_model.poisson_time_prediction_model import (
    PoissonTimePredictionModel,
)
from time_prediction_model.time_prediction_model import TimePredictionModel
from time_prediction_model.time_prediction_model_factory.time_prediction_model_params.hawkes_time_prediction_model_params import (
    HawkesTimePredictionModelParams,
)
from time_prediction_model.time_prediction_model_factory.time_prediction_model_params.moving_average_time_prediction_model_params import (
    MovingAverageTimePredictionModelParams,
)
from time_prediction_model.time_prediction_model_factory.time_prediction_model_params.naive_time_prediction_model_params import (
    NaiveTimePredictionModelParams,
)
from time_prediction_model.time_prediction_model_factory.time_prediction_model_params.poisson_time_prediction_model_params import (
    PoissonTimePredictionModelParams,
)


class TimePredictionModelFactory:
    def __init__(
        self,
        model_type: str,
        prediction_period_duration: int,
        parameters_dir_path: str,
        file_start_registration_time: int,
        file_start_simulation_time: int
    ) -> None:
        self._model_type = model_type
        self._prediction_period_duration = prediction_period_duration
        self._parameters_dir_path = parameters_dir_path
        self._file_start_registration_time = file_start_registration_time
        self._file_start_simulation_time = file_start_simulation_time

    def get_model(self) -> TimePredictionModel:
        if self._model_type in (CONST.MULTIVARIATE_HAWKES, CONST.UNIVARIATE_HAWKES):
            params = HawkesTimePredictionModelParams.from_files_indications(
                self._parameters_dir_path,
                self._file_start_registration_time,
                self._file_start_simulation_time
            )
            model = HawkesTimePredictionModel(
                params._model_params,
                self._prediction_period_duration
            )
            return model
        
        elif self._model_type == CONST.POISSON:
            params = PoissonTimePredictionModelParams.from_files_indications(
                self._parameters_dir_path,
                self._file_start_registration_time,
                self._file_start_simulation_time
            )
            model = PoissonTimePredictionModel(
                params._model_params,
                self._prediction_period_duration
            )
            return model

        elif self._model_type == CONST.MOVING_AVERAGE:
            params = MovingAverageTimePredictionModelParams.from_files_indications(
                self._parameters_dir_path,
                self._file_start_registration_time,
                self._file_start_simulation_time
            )
            model = MovingAverageTimePredictionModel(
                params._model_params,
            )
            return model

        elif self._model_type == CONST.NAIVE:
            params = NaiveTimePredictionModelParams.from_files_indications(
                self._parameters_dir_path,
                self._file_start_registration_time,
                self._file_start_simulation_time
            )
            model = NaiveTimePredictionModel(
                params._model_params,
            )
            return model
        
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")