import pandas as pd

from src.lob_period.time_series import TimeSeries


class LOBPeriod:
    def __init__(self, lob_df: pd.DataFrame, time_series: TimeSeries) -> None:
        self._lob_df = lob_df
        self._time_series = time_series

    def get_nearest_value(self, value: float) -> float:
        return self._time_series.get_nearest_value(value)

    def get_last_value(self, limit_value: float) -> float:
        return self._time_series.get_last_value(limit_value)
    
    def get_next_value(self, this_value: float) -> float:
        return self._time_series.get_next_value(this_value)
    
    def get_base_imbalance(self, time: float) -> float:
        time_index = self._time_series.get_index_of_last_value_lower_than(time)
        return self._lob_df['BaseImbalance'][time_index]
    
    def get_lob_df_with_timestamp_column(self) -> pd.DataFrame:
        lob_df = self._lob_df.copy()
        lob_df['Timestamp'] = self._time_series.get_time_events()
        return lob_df
