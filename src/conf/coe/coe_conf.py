from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CoeConf:
    pairs: List[str]
    bi_levels: List[int]
    training_times_seconds: List[int]
    simulation_periods_seconds: List[int]

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> "CoeConf":
        return cls(
            pairs=conf["pairs"],
            bi_levels=conf["bi_levels"],
            training_times_seconds=conf["training_times_seconds"],
            simulation_periods_seconds=conf["simulation_periods_seconds"],
        )
