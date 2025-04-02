from typing import Dict

import numpy as np
import pandas as pd

from src.events_extractor.lob_events_extractor import LOBEventsExtractor
from src.lob_event_type_indicator.lob_event_type import LOBEventType
from src.lob_event_type_indicator.lob_event_type_indicator import LOBEventTypeIndicator


class MultivariateLOBEventsExtractor(LOBEventsExtractor):
    def __init__(
        self,
        lob_dataframe: pd.DataFrame,
        num_levels_in_a_side: int,
        num_levels_for_which_save_events: int,
    ) -> None:
        super().__init__(lob_dataframe)

        self._num_levels_in_a_side = num_levels_in_a_side
        self._num_levels_for_which_save_events = num_levels_for_which_save_events

    def get_events(self) -> Dict[LOBEventType, np.ndarray]:
        lob_type_event_indicator = LOBEventTypeIndicator(
            self._lob_df,
            self._num_levels_in_a_side,
            self._num_levels_for_which_save_events,
        )
        event_type_times_map = lob_type_event_indicator.get_lob_event_type_times_map()

        return event_type_times_map
