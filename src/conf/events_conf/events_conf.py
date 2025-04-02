from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EventsConf:
    num_levels_for_which_save_events: int
    num_levels_in_a_side: int
    combined_event_types_map: Dict[str, List[str]]
    events_to_compute: List[str]

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> "EventsConf":
        return cls(
            num_levels_for_which_save_events=conf["num_levels_for_which_save_events"],
            num_levels_in_a_side=conf["num_levels_in_a_side"],
            combined_event_types_map=conf["combined_event_types_map"],
            events_to_compute=conf["events_to_compute"],
        )
    