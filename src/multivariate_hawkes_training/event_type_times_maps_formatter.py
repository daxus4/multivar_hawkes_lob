from typing import Dict, List, Optional

import numpy as np


class EventTypeTimesMapsFormatter:

    def get_events_types_periods(
        self,
        event_type_times_maps: List[Dict[str, np.ndarray]],
        event_types_ordered: Optional[List[str]] = None,
    ) -> List[List[np.ndarray]]:
        if event_types_ordered is None:
            event_types_ordered = list(event_type_times_maps[0].keys())

        event_types_periods = list()

        for event_type_times_map in event_type_times_maps:
            events_types_period = list()

            for event_type in event_types_ordered:
                events_types_period.append(event_type_times_map[event_type])

            event_types_periods.append(events_types_period)

        return event_types_periods
