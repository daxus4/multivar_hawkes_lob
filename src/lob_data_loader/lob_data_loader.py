import pandas as pd


class LOBDataLoader:

    def get_lob_dataframe(self, path: str, base_imbalance_level: int) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t')
        df = self._get_preprocessed_df(df, base_imbalance_level)
        
        return df

    def _get_preprocessed_df(
        self, df: pd.DataFrame, base_imbalance_orderbook_level: int
    ) -> pd.DataFrame:
        df["Timestamp"] = df["Timestamp"] / 1000
        df['MidPrice'] = (df["AskPrice1"]+df["BidPrice1"])/2
        df['Return'] = (-df["MidPrice"]+df["MidPrice"].shift(-1)) / df["MidPrice"]

        pbid = df["BidPrice1"] - df[f"BidPrice{base_imbalance_orderbook_level}"]
        pask = df[f"AskPrice{base_imbalance_orderbook_level}"] - df["AskPrice1"]
        df["BaseImbalance"] = (pbid-pask)/(pbid+pask)

        return df
