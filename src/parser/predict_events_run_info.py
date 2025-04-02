from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class PredictEventsRunInfo:
    testing_conf_filename: str
    events_conf_filename: str
    model_name: str

    @classmethod
    def from_namedtuple(cls, tuple_info: NamedTuple) -> "PredictEventsRunInfo":
        return cls(
            testing_conf_filename=tuple_info.testing_conf_filename,
            events_conf_filename=tuple_info.events_conf_filename,
            model_name=tuple_info.model_name,
        )