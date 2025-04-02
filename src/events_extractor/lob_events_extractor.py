from abc import ABC
from typing import Any

import pandas as pd


class LOBEventsExtractor(ABC):
    def __init__(self, lob_dataframe: pd.DataFrame) -> None:
        self._lob_df = lob_dataframe

    def get_events(self) -> Any:
        raise NotImplementedError
