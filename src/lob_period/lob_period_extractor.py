import pandas as pd

from src.lob_period.lob_period import LOBPeriod
from src.lob_period.time_series import TimeSeries


class LOBPeriodExtractor:
    def __init__(self, lob_df: pd.DataFrame) -> None:
        self._lob_df = lob_df

    def get_lob_period(self, start_time: float, end_time: float) -> LOBPeriod:
        lob_df = self._get_cutted_lob_df(start_time, end_time)
        time_series = self._get_time_series(start_time, lob_df)

        return LOBPeriod(lob_df, time_series)

    def _get_time_series(self, start_time: float, lob_df: pd.DataFrame) -> TimeSeries:
        time_series = lob_df['Timestamp'].values
        time_series = time_series - start_time

        time_series = TimeSeries(time_series)
        return time_series

    def _get_cutted_lob_df(self, start_time: float, end_time: float) -> pd.DataFrame:
        lob_df = self._lob_df[
            (self._lob_df['Timestamp'] >= start_time) & (self._lob_df['Timestamp'] < end_time)
        ].copy()
        lob_df.reset_index(drop=True, inplace=True)

        return lob_df