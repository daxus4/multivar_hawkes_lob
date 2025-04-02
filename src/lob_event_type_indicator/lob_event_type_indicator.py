from decimal import Decimal
from typing import Dict, List, Type

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

from src.lob_event_type_indicator.lob_event_type import LOBEventType
from src.lob_event_type_indicator.lob_event_type_reconstructor import (
    LOBEventTypeReconstructor,
)
from src.lob_event_type_indicator.lob_snapshot import LOBSnapshotFactory


class LOBEventTypeIndicator:
    def __init__(
        self,
        lob_dataframe: pd.DataFrame,
        num_levels_in_a_side: int,
        num_levels_for_which_save_events: int,
    ):
        self._num_levels_in_a_side = num_levels_in_a_side
        self._num_levels_for_which_save_events = num_levels_for_which_save_events

        self._lob_event_type_reconstructor = LOBEventTypeReconstructor(
            self._num_levels_in_a_side, self._num_levels_for_which_save_events
        )

        self._lob_snapshot_factory = LOBSnapshotFactory(
            lob_dataframe, self._num_levels_in_a_side
        )

    def get_lob_event_type_times_map(self) -> Dict[LOBEventType, np.ndarray]:
        event_type_times_map = dict()

        lob_snapshot_iterator = self._lob_snapshot_factory.get_lob_snapshots_iterator()
        prev_lob_snapshot = next(lob_snapshot_iterator)

        for curr_lob_snapshot in lob_snapshot_iterator:
            lob_event_types_container = (
                self._lob_event_type_reconstructor.get_happened_event(
                    prev_lob_snapshot, curr_lob_snapshot
                )
            )

            self._update_event_type_times_map(
                event_type_times_map,
                lob_event_types_container.price_bid_event_map,
                lob_event_types_container.curr_timestamp,
            )

            self._update_event_type_times_map(
                event_type_times_map,
                lob_event_types_container.price_ask_event_map,
                lob_event_types_container.curr_timestamp,
            )

            prev_lob_snapshot = curr_lob_snapshot

        event_type_times_map = {k: np.array(v) for k, v in event_type_times_map.items()}

        return event_type_times_map

    def _update_event_type_times_map(
        self,
        event_type_times_map: Dict[LOBEventType, List[int]],
        price_event_map: Dict[Decimal, Type[LOBEventType]],
        curr_timestamp: int,
    ) -> None:
        for bid_event in price_event_map.values():
            if bid_event not in event_type_times_map:
                event_type_times_map[bid_event] = []
            event_type_times_map[bid_event].append(curr_timestamp)
