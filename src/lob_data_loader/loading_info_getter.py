import os
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class LoadingInfo:
    path: str
    start_registration_time: int
    start_times: List[float]

class LoadingInfoGetter:

    def __init__(self, period_df: pd.DataFrame) -> None:
        self._period_df = self._get_period_df_grouped(period_df)

    def _get_period_df_grouped(self, period_df: pd.DataFrame) -> pd.DataFrame:
        return period_df[['timestamp', 'timestamp_density']].groupby('timestamp').agg(
            {'timestamp_density': list}
        ).reset_index()


    def get_loading_info(
        self, lob_df_folder_path: str, lob_df_prefix: str
    ) -> List[LoadingInfo]:
        loading_infos = list()

        for df_timestamp, starting_time_periods in zip(
            self._period_df['timestamp'], self._period_df['timestamp_density']
        ):
            path = self._get_lob_df_path(lob_df_folder_path, lob_df_prefix, df_timestamp)
            loading_info = LoadingInfo(path, df_timestamp, starting_time_periods)
            loading_infos.append(loading_info)

        return loading_infos
    
    def _get_lob_df_path(
        self, lob_df_folder_path: str, lob_df_prefix: str, df_timestamp: float
    ) -> str:
        lob_files = os.listdir(lob_df_folder_path)
        lob_files = [
            file
            for file in lob_files
            if file.startswith(lob_df_prefix) and str(df_timestamp) in file
        ]

        if len(lob_files) != 1:
            raise ValueError(f"Found {len(lob_files)} files for timestamp {df_timestamp}")
        
        return os.path.join(lob_df_folder_path, lob_files[0])