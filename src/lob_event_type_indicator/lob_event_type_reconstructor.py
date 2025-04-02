from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Type

from sortedcontainers import SortedDict

from src.lob_event_type_indicator.lob_event_type import LOBEventType
from src.lob_event_type_indicator.lob_snapshot import LOBSnapshot


@dataclass(frozen=True)
class LOBEventTypesContainer:
    price_bid_event_map: Dict[Decimal, Type[LOBEventType]]
    price_ask_event_map: Dict[Decimal, Type[LOBEventType]]
    curr_timestamp: Optional[int]


class LOBEventTypeReconstructor:
    def __init__(
        self, num_levels_considered: int, num_levels_for_which_save_events: int
    ):
        """
        Reconstruct the LOB events types occurred between two snapshots.
        num_levels_considered is the level number to which you want to truncate the orderbook
        num_levels_for_which_save_events is the number of level that you want to check if
        events are happened. This is because you cannot be sure to what happens when
        a level is not tracked anymore in these num_levels_considered. So this algorithm
        works well for example when num_levels_for_which_save_events = 4 and
        num_levels_for_which_save_events = 20, but we cannot be sure because if
        there are 20 new limits with aggressive prices we will lose all the information.
        The expectation is that such movements are very rare.
        """
        self._num_levels_considered = num_levels_considered
        self._num_levels_for_which_save_events = num_levels_for_which_save_events
        self._snapshot_levels_sorted_map_factory = (
            LOBSideSnapshotsLevelsSortedMapFactory(self._num_levels_considered)
        )

    def get_happened_event(
        self, prev_lob_snapshot: LOBSnapshot, curr_lob_snapshot: LOBSnapshot
    ) -> LOBEventTypesContainer:
        price_bid_event_map = self.get_happened_event_types_for_side(
            prev_lob_snapshot.bids_from_best, curr_lob_snapshot.bids_from_best, True
        )

        price_ask_event_map = self.get_happened_event_types_for_side(
            prev_lob_snapshot.asks_from_best, curr_lob_snapshot.asks_from_best, False
        )

        return LOBEventTypesContainer(
            price_bid_event_map, price_ask_event_map, curr_lob_snapshot.timestamp
        )

    def get_happened_event_types_for_side(
        self,
        prev_side_levels_from_best: List[Tuple[float, float]],
        curr_side_levels_from_best: List[Tuple[float, float]],
        is_bid: bool,
    ) -> Dict[Decimal, Type[LOBEventType]]:

        snapshot_levels_sorted_map = (
            self._snapshot_levels_sorted_map_factory.get_snapshots_levels_sorted_map(
                prev_side_levels_from_best, curr_side_levels_from_best, is_bid
            )
        )

        # last_level_cancelled_from_market_order is needed to manage a market
        # order that delete the first level and hit the successives
        last_level_cancelled_from_market_order = False
        lob_event_types = SortedDict()
        for i, price_level in enumerate(
            snapshot_levels_sorted_map.get_sorted_price_levels_from_best_iterator()
        ):
            if i < self._num_levels_for_which_save_events:
                is_best_level = i == 0

                if snapshot_levels_sorted_map.is_cancelled(price_level):
                    lob_event_types[price_level] = (
                        LOBEventType.get_value_from_conditions(
                            is_bid,
                            False,
                            is_best_level,
                            # Here I consider all removed liquidity at best level
                            # as market order. Also successive levels hitted when
                            # best level is deleted from market order
                            is_best_level or last_level_cancelled_from_market_order,
                        )
                    )

                    last_level_cancelled_from_market_order = is_best_level

                elif snapshot_levels_sorted_map.is_added(price_level):
                    lob_event_types[price_level] = (
                        LOBEventType.get_value_from_conditions(
                            is_bid, True, is_best_level, False
                        )
                    )

                    last_level_cancelled_from_market_order = False

                elif snapshot_levels_sorted_map.is_size_decreased(price_level):
                    lob_event_types[price_level] = (
                        LOBEventType.get_value_from_conditions(
                            is_bid,
                            False,
                            False,
                            # Here I consider all removed liquidity at best level
                            # as market order. Also successive levels hitted when
                            # best level is deleted from market order
                            is_best_level or last_level_cancelled_from_market_order,
                        )
                    )

                    last_level_cancelled_from_market_order = False

                elif snapshot_levels_sorted_map.is_size_increased(price_level):
                    lob_event_types[price_level] = (
                        LOBEventType.get_value_from_conditions(
                            is_bid, True, False, False
                        )
                    )

                    last_level_cancelled_from_market_order = False

        return lob_event_types


class LOBSideSnapshotsLevelsSortedMap:
    _PREV_SNAPSHOT_INDEX = 0
    _CURR_SNAPSHOT_INDEX = 1

    def __init__(
        self,
        snapshots_levels_sorted_map: Dict[Decimal, Tuple[Decimal, Decimal]],
        is_bid: bool,
    ):
        self._snapshots_levels_sorted_map = snapshots_levels_sorted_map
        self._is_bid = is_bid

    def get_size(self, price: Decimal, snapshot_index: int) -> Decimal:
        return self._snapshots_levels_sorted_map[price][snapshot_index]

    def is_cancelled(self, price: Decimal) -> bool:
        return (
            self.get_size(price, self._CURR_SNAPSHOT_INDEX).is_zero()
            and not self.get_size(price, self._PREV_SNAPSHOT_INDEX).is_zero()
        )

    def is_added(self, price: Decimal) -> bool:
        return (
            self.get_size(price, self._PREV_SNAPSHOT_INDEX).is_zero()
            and not self.get_size(price, self._CURR_SNAPSHOT_INDEX).is_zero()
        )

    def is_size_decreased(self, price: Decimal) -> bool:
        return self.get_size(price, self._PREV_SNAPSHOT_INDEX) > self.get_size(
            price, self._CURR_SNAPSHOT_INDEX
        )

    def is_size_increased(self, price: Decimal) -> bool:
        return self.get_size(price, self._PREV_SNAPSHOT_INDEX) < self.get_size(
            price, self._CURR_SNAPSHOT_INDEX
        )

    def get_sorted_price_levels_from_best_iterator(self):
        return (
            reversed(self._snapshots_levels_sorted_map.keys())
            if self._is_bid
            else iter(self._snapshots_levels_sorted_map.keys())
        )


class LOBSideSnapshotsLevelsSortedMapFactory:
    def __init__(self, num_levels_considered: int) -> None:
        self._num_levels_considered = num_levels_considered

    def get_snapshots_levels_sorted_map(
        self,
        prev_side_levels_from_best: List[Tuple[Decimal, Decimal]],
        curr_side_levels_from_best: List[Tuple[Decimal, Decimal]],
        is_bid: bool,
    ) -> LOBSideSnapshotsLevelsSortedMap:
        return LOBSideSnapshotsLevelsSortedMap(
            self._get_snapshots_levels_sorted_map_input(
                prev_side_levels_from_best, curr_side_levels_from_best
            ),
            is_bid,
        )

    def _get_snapshots_levels_sorted_map_input(
        self,
        prev_side_levels_from_best: List[Tuple[Decimal, Decimal]],
        curr_side_levels_from_best: List[Tuple[Decimal, Decimal]],
    ) -> Dict[Decimal, Tuple[Decimal, Decimal]]:

        prev_side_levels_from_best = prev_side_levels_from_best[
            : self._num_levels_considered
        ]
        curr_side_levels_from_best = curr_side_levels_from_best[
            : self._num_levels_considered
        ]

        snapshots_levels_sorted_map = self._get_snapshot_level_map_for_previous(
            prev_side_levels_from_best
        )

        for curr_side_level in curr_side_levels_from_best:
            curr_side_level_price = curr_side_level[LOBSnapshot.PRICE_INDEX]
            curr_side_level_size = curr_side_level[LOBSnapshot.SIZE_INDEX]

            if curr_side_level_price in snapshots_levels_sorted_map.keys():
                snapshots_levels_sorted_map[curr_side_level_price] = (
                    self._get_updated_snapshot_level_map_entry_with_current_size(
                        snapshots_levels_sorted_map,
                        curr_side_level_price,
                        curr_side_level_size,
                    )
                )

            elif not (curr_side_level_price.is_nan() or curr_side_level_size.is_nan()):
                snapshots_levels_sorted_map[curr_side_level_price] = (
                    self._get_new_snapshot_level_map_entry_for_current(
                        curr_side_level_size
                    )
                )

        return snapshots_levels_sorted_map

    def _get_snapshot_level_map_for_previous(
        self, prev_side_levels_from_best: List[Tuple[Decimal, Decimal]]
    ) -> Dict[Decimal, Tuple[Decimal, Decimal]]:
        snapshots_levels_sorted_map = SortedDict()

        for prev_side_level in prev_side_levels_from_best:
            prev_side_level_price = prev_side_level[LOBSnapshot.PRICE_INDEX]
            prev_side_level_size = prev_side_level[LOBSnapshot.SIZE_INDEX]

            if not (prev_side_level_price.is_nan() or prev_side_level_size.is_nan()):
                snapshots_levels_sorted_map[prev_side_level_price] = (
                    self._get_new_snapshot_level_map_entry_for_previous(
                        prev_side_level_size
                    )
                )

        return snapshots_levels_sorted_map

    def _get_updated_snapshot_level_map_entry_with_current_size(
        self,
        snapshots_levels_sorted_map: Dict[Decimal, Tuple[Decimal, Decimal]],
        curr_side_level_price: Decimal,
        curr_side_level_size: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        return (
            snapshots_levels_sorted_map[curr_side_level_price][0],
            curr_side_level_size,
        )

    def _get_new_snapshot_level_map_entry_for_current(
        self, size: Decimal
    ) -> Tuple[Decimal, Decimal]:
        return (Decimal(0), size)

    def _get_new_snapshot_level_map_entry_for_previous(
        self, size: Decimal
    ) -> Tuple[Decimal, Decimal]:
        return (size, Decimal(0))
