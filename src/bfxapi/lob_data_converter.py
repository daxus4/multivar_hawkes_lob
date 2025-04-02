import json
import os
from typing import Dict, List, Tuple
from sortedcontainers import SortedDict
import pandas as pd

class OrderBook:
    def __init__(self):
        self.levels = 25
        self._book = {"bids": SortedDict(), "asks": SortedDict()}

    def insert_snapshot(self, snapshot: Dict):
        self._insert_side_snapshot(snapshot["bids"], is_bid=True)
        self._insert_side_snapshot(snapshot["asks"], is_bid=False)

    def _insert_side_snapshot(self, snapshot: Dict, is_bid: bool):
        side = self.get_side_str(is_bid)

        for order in snapshot.values():
            self._insert_level(order, side)

    def get_side_str(self, is_bid):
        return "bids" if is_bid else "asks"

    def _insert_level(self, order: Dict, side: str):
        price = order["p"]
        amount = order["a"]

        self._book[side][price] = amount if side == "bids" else -amount

    def insert_updates_for_timestamp(self, update: Dict[str, List[float]]):
        if not (len(update["p"]) == len(update["a"]) == len(update["c"])):
            raise Exception("Update is not valid")

        for price, amount, count in zip(update["p"], update["a"], update["c"]):
            self.insert_update(price, amount, count)

    def insert_update(self, price, amount, count):
        side = self.get_side_str_for_update(amount)
        if count > 0:
            self._book[side][price] = amount if side == "bids" else -amount

        elif count == 0:
            if price in self._book[side]:
                self._book[side].pop(price)

    def get_side_str_for_update(self, amount: int):
        return "bids" if amount > 0 else "asks"

    def get_column_names(self, side: str, info_type: str):
        side = "Bid" if side == "bids" else "Ask"

        return [f"{side}{info_type}{i}" for i in range(1, self.levels + 1)]

    def get_best_bid_price(self) -> float:
        return self._book["bids"].peekitem(index=-1)[0]

    def get_best_ask(self) -> float:
        return self._book["asks"].peekitem(index=0)[0]

    def get_mid_price(self) -> float:
        return (self.get_best_bid_price() + self.get_best_ask()) / 2

    def get_row_book(self) -> Dict[str, float]:
        price_size_bid_map = self._book["bids"]
        price_size_ask_map = self._book["asks"]

        # Create a new dictionary with custom keys
        row_dict = {}

        # Iterate over the items in the ask_order_book
        for i, (price, size) in enumerate(price_size_ask_map.items(), start=1):
            row_dict[f"AskPrice{i}"] = price
            row_dict[f"AskSize{i}"] = size

        for i, (price, size) in enumerate(
            reversed(price_size_bid_map.items()), start=1
        ):
            row_dict[f"BidPrice{i}"] = price
            row_dict[f"BidSize{i}"] = size

        return row_dict


class LobDataConverter:
    def __init__(self, prefix_path: str, subdirectory: str):
        self.prefix_path = prefix_path
        self.subdirectory = subdirectory

    def get_changes_orderbook_df(
        self,
        timestamp_snapshot: int,
        snapshot_dict: Dict[int, Dict],
        timestamp_updates_map: SortedDict,
    ) -> pd.DataFrame:
        row_dicts = []

        order_book = OrderBook()
        order_book.insert_snapshot(snapshot_dict)

        # last_mid_price = order_book.get_mid_price()

        self._append_new_row(timestamp_snapshot, order_book, row_dicts)

        for timestamp, update in timestamp_updates_map.items():
            order_book.insert_updates_for_timestamp(update)

            # current_mid_price = order_book.get_mid_price()
            # if current_mid_price != last_mid_price:
            self._append_new_row(timestamp, order_book, row_dicts)
            # last_mid_price = current_mid_price

        df = pd.DataFrame(row_dicts)

        return df


    def _append_new_row(self, timestamp: int, order_book: OrderBook, row_dicts: List[Dict]):
        row_dict = order_book.get_row_book()
        row_dict["Timestamp"] = timestamp
        row_dicts.append(row_dict)


    def read_orderbook_json(self, path: str) -> SortedDict:
        with open(path, "r") as f:
            json_dict = json.load(f)

        json_dict = SortedDict({int(key): value for key, value in json_dict.items()})
        return json_dict


    def pop_final_snapshot(self, orderbook_json: SortedDict) -> Dict:
        if -1 in orderbook_json.keys():
            return orderbook_json.pop(-1)
        else:
            return None


    def pop_first_timestamp_and_snapshot(
        self, orderbook_json: SortedDict
    ) -> Tuple[int, Dict]:
        return orderbook_json.popitem(index=0)


    def get_orderbook_changes_df(self, orderbook_json: SortedDict) -> pd.DataFrame:
        timestamp_snapshot, initial_snapshot = self.pop_first_timestamp_and_snapshot(
            orderbook_json
        )

        return self.get_changes_orderbook_df(
            timestamp_snapshot, initial_snapshot, orderbook_json
        )

    def is_final_snapshot_correct(self, final_snapshot_row, orderbook_changes_df):
        last_row = orderbook_changes_df.iloc[-1].to_dict()
        changed_keys = [key for key, value in final_snapshot_row.items() if value != last_row[key]]

        return len(changed_keys) == 0

    def is_data_collection_in_file_interrupted(self, filename: str) -> bool:
        return filename.endswith("interrupted.json")

    def get_file_timestamp(self, filename: str) -> int:
        return int(filename.split(".")[0].split("_")[1])

    def get_file_timestamp_for_processed_file(self, filename: str) -> int:
        return int(filename.split(".")[0].split("_")[2])

    def get_orderbook_changes_filename(self, prefix_path: str, timestamp: int, is_interrupted: bool) -> str:
        prefix = os.path.join(prefix_path, "orderbook_changes_")
        return f"{prefix}{timestamp}{'_interrupted' if is_interrupted else ''}.tsv"

    def save_orderbook_changes_df(self, orderbook_changes_df: pd.DataFrame, timestamp: int, is_interrupted: bool, directory: str):
        orderbook_changes_df_filename = self.get_orderbook_changes_filename(
            directory, timestamp, is_interrupted
        )
        orderbook_changes_df.to_csv(orderbook_changes_df_filename, index=False, sep="\t")

    def get_final_snapshot_row_from_json_data(self, orderbook_data):
        final_snapshot = self.pop_final_snapshot(orderbook_data)
        final_snapshot_orderbook = OrderBook()
        final_snapshot_orderbook.insert_snapshot(final_snapshot)
        final_snapshot_row = final_snapshot_orderbook.get_row_book()
        return final_snapshot_row

    def get_files(self, directory: str) -> List[str]:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def get_timestamp_of_already_processed_files(self, directory: str) -> List[int]:
        files = self.get_files(directory)
        return [self.get_file_timestamp_for_processed_file(f) for f in files]

    def get_json_files_to_process(self, main_directory: str, processed_directory: str) -> List[str]:
        already_processed_timestamps = self.get_timestamp_of_already_processed_files(processed_directory)
        files = self.get_files(main_directory)
        return [f for f in files if self.get_file_timestamp(f) not in already_processed_timestamps]

    def convert_files(self):
        full_path_subdirectory_orderbook_changes = os.path.join(self.prefix_path, self.subdirectory)
        for filename in self.get_json_files_to_process(self.prefix_path, full_path_subdirectory_orderbook_changes):
            try:
                print(filename)

                timestamp = self.get_file_timestamp(filename)
                is_interrupted = self.is_data_collection_in_file_interrupted(filename)

                full_path_json = os.path.join(self.prefix_path, filename)

                orderbook_data = self.read_orderbook_json(full_path_json)
                if not is_interrupted:
                    final_snapshot_row = self.get_final_snapshot_row_from_json_data(orderbook_data)

                orderbook_changes_df = self.get_orderbook_changes_df(orderbook_data)
                if not is_interrupted:
                    if not self.is_final_snapshot_correct(final_snapshot_row, orderbook_changes_df):
                        raise Exception("Final snapshot is not correct")

                self.save_orderbook_changes_df(
                    orderbook_changes_df, timestamp, is_interrupted, full_path_subdirectory_orderbook_changes
                )
            except Exception as e:
                print(f"{filename}: {e}")
                continue
