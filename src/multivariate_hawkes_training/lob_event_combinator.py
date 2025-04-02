from typing import Dict, List

import numpy as np


class LOBEventCombinator:
    def __init__(self, event_type_times_maps: List[Dict[str, np.ndarray]]) -> None:
        self._event_type_times_maps = event_type_times_maps

    @property
    def event_type_times_maps(self) -> List[Dict[str, np.ndarray]]:
        return self._event_type_times_maps
    
    @event_type_times_maps.setter
    def event_type_times_maps(self, event_type_times_maps: List[Dict[str, np.ndarray]]) -> None:
        self._event_type_times_maps = event_type_times_maps

    def get_event_type_times_maps_with_new_combination(
        self,
        lob_event_types_to_combine: List[str],
        combination_name: str = "combination",
    ) -> List[Dict[str, np.ndarray]]:
        event_type_times_combination = self.get_event_type_times_combination(
            lob_event_types_to_combine
        )

        event_type_times_maps_with_new_combination = list()

        for event_type_times_map in self._event_type_times_maps:
            event_type_times_map_with_new_combination = dict(event_type_times_map)
            event_type_times_map_with_new_combination[combination_name] = (
                event_type_times_combination.pop(0)
            )

            event_type_times_maps_with_new_combination.append(
                event_type_times_map_with_new_combination
            )

        return event_type_times_maps_with_new_combination

    def get_event_type_times_combination(
        self, lob_event_types_to_combine: List[str]
    ) -> List[np.ndarray]:
        event_type_times_combination = list()

        for event_type_times_map in self._event_type_times_maps:
            event_type_times_combination.append(
                self.sorted_array_from_keys(
                    event_type_times_map, lob_event_types_to_combine
                )
            )

        return event_type_times_combination

    def sorted_array_from_keys(
        self,
        event_type_times_map: Dict[str, np.ndarray],
        lob_event_types_to_combine: List[str],
    ) -> np.ndarray:
        arrays = [
            event_type_times_map[key]
            for key in lob_event_types_to_combine
            if key in event_type_times_map
        ]
        combined_array = np.concatenate(arrays)
        sorted_array = np.unique(np.sort(combined_array))

        return sorted_array
