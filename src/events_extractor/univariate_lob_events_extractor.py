import numpy as np

from src.events_extractor.lob_events_extractor import LOBEventsExtractor


class UnivariateLOBEventsExtractor(LOBEventsExtractor):
    def get_events(self) -> np.ndarray:
        raise NotImplementedError
