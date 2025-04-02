import numpy as np


class TimeSeries:
    def __init__(self, time_events: np.ndarray) -> None:
        self._time_events = time_events
        
    def get_nearest_value(self, value: float) -> float:
        idx = (np.abs(self._time_events - value)).argmin()
        return self._time_events[idx]

    def get_last_value(self, limit_value: float) -> float:
        return self._time_events[self._time_events <= limit_value][-1]
    
    def get_next_value(self, this_value: float) -> float:
        return self._time_events[self._time_events > this_value][0]
    
    def get_index_of_last_value_lower_than(self, value: float) -> int:
        return np.searchsorted(self._time_events, value, side='right') - 1
    
    def get_time_events(self) -> np.ndarray:
        return self._time_events.copy()

        
