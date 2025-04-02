# python -c "import examples.websocket.public.order_book"

import json
import os
import zlib
from collections import OrderedDict
from copy import deepcopy
from decimal import Decimal
from math import floor, log10
from typing import Any, Dict, List, cast

from bfxapi import Client
from bfxapi.types import TradingPairBook
from bfxapi.websocket.subscriptions import Book


def _format_float(value: float) -> str:
    """
    Format float numbers into a string compatible with the Bitfinex API.
    """

    def _find_exp(number: float) -> int:
        base10 = log10(abs(number))

        return floor(base10)

    if _find_exp(value) >= -6:
        return format(Decimal(repr(value)), "f")

    return str(value).replace("e-0", "e-")


class OrderBook:
    def __init__(self):
        self.__order_book = {"bids": OrderedDict(), "asks": OrderedDict()}

    def update(self, data: TradingPairBook) -> None:
        price, count, amount = data.price, data.count, data.amount

        kind = "bids" if amount > 0 else "asks"

        if count > 0:
            self.__order_book[kind][price] = {
                "p": price,
                "c": count,
                "a": amount,
            }

        if count == 0:
            if price in self.__order_book[kind]:
                del self.__order_book[kind][price]

    def verify(self, checksum: int) -> bool:
        values: List[int] = []

        bids = sorted(
            [
                (data["p"], data["c"], data["a"])
                for _, data in self.__order_book["bids"].items()
            ],
            key=lambda data: -data[0],
        )

        asks = sorted(
            [
                (data["p"], data["c"], data["a"])
                for _, data in self.__order_book["asks"].items()
            ],
            key=lambda data: data[0],
        )

        if len(bids) < 25 or len(asks) < 25:
            raise AssertionError("Not enough bids or asks (need at least 25).")

        for _i in range(25):
            bid, ask = bids[_i], asks[_i]
            values.extend([bid[0], bid[2]])
            values.extend([ask[0], ask[2]])

        local = ":".join(_format_float(value) for value in values)

        crc32 = zlib.crc32(local.encode("UTF-8"))

        return crc32 == checksum

    def is_verifiable(self) -> bool:
        return (
            len(self.__order_book["bids"]) >= 25
            and len(self.__order_book["asks"]) >= 25
        )

    def clear(self) -> None:
        self.__order_book = {"bids": OrderedDict(), "asks": OrderedDict()}


class LobRecorder():
    def __init__(self, symbol: str, recording_duration: int = 7200, saving_path: str = "data"):
        self.symbol = symbol
        self.recording_duration = recording_duration
        self.order_book = OrderBook()
        self.timestamp_orderbook_update_map = dict()
        self.last_checksum_timestamp = 0
        self.starting_timestamp = 0
        self.ending_timestamp = 0
        self.are_new_messages_to_drop = False
        self.bfx = Client()
        
        self.saving_path = saving_path
        if not os.path.exists(f"{self.saving_path}"):
            os.makedirs(f"{self.saving_path}")

        self._init_client_events()

    def _init_client_events(self):
        self.bfx.wss.on("open")(self.on_open)
        self.bfx.wss.on("subscribed")(self.on_subscribed)
        self.bfx.wss.on("t_book_snapshot")(self.on_t_book_snapshot)
        self.bfx.wss.on("t_book_update")(self.on_t_book_update)
        self.bfx.wss.on("checksum")(self.on_checksum)


    async def on_open(self):
        await self.bfx.wss.subscribe("book", symbol=self.symbol, prec="P0", len="25")

    async def on_subscribed(self, subscription):
        print(f"Subscription successful for symbol <{subscription['symbol']}>")

    async def on_t_book_snapshot(
        self, subscription: Book, snapshot: List[TradingPairBook], timestamp: int
    ):
        self.starting_timestamp = timestamp
        self.ending_timestamp = self.starting_timestamp + int(self.recording_duration * 1e3)
        self.are_new_messages_to_drop = False

        for data in snapshot:
            self.order_book.update(data)
        self.timestamp_orderbook_update_map[timestamp] = deepcopy(
            self.order_book._OrderBook__order_book
        )

    async def on_t_book_update(self, subscription: Book, data: TradingPairBook, timestamp: int):
        if self.are_new_messages_to_drop:
            return
        
        self.order_book.update(data)

        if timestamp in self.timestamp_orderbook_update_map:
            self.timestamp_orderbook_update_map[timestamp]["p"].append(data.price)
            self.timestamp_orderbook_update_map[timestamp]["a"].append(data.amount)
            self.timestamp_orderbook_update_map[timestamp]["c"].append(data.count)
        else:
            self.timestamp_orderbook_update_map[timestamp] = {
                "p": [data.price],
                "a": [data.amount],
                "c": [data.count],
            }
        if timestamp >= self.ending_timestamp:
            self.timestamp_orderbook_update_map[-1] = deepcopy(self.order_book._OrderBook__order_book)

            self.save_orderbook_messages_to_file(timestamp)

            await self._reset_connection(subscription)

    def save_orderbook_messages_to_file(self, timestamp: int, is_interrupted: bool = False):
        path = (
            f"{self.saving_path}/data_{timestamp}_interrupted.json"
            if is_interrupted
            else f"{self.saving_path}/data_{timestamp}.json"
        )

        with open(path, "w") as json_file:
            json.dump(self.timestamp_orderbook_update_map, json_file)


    async def on_checksum(self, subscription: Book, value: int, timestamp: int):
        if self.order_book.is_verifiable():
            if not self.order_book.verify(value):
                print(
                    "Mismatch between local and remote checksums: " + f"restarting book..."
                )

                self.timestamp_orderbook_update_map = self._get_orderbook_updates_until(
                    self.timestamp_orderbook_update_map, self.last_checksum_timestamp
                )

                self.save_orderbook_messages_to_file(
                    self.last_checksum_timestamp, is_interrupted=True
                )

                await self._reset_connection(subscription)
            else:
                self.last_checksum_timestamp = timestamp
        else:
            print("Order book not verifiable.")

    def _get_orderbook_updates_until(self, timestamp_orderbook_update_map: Dict, timestamp: int):
        return {k: v for k, v in timestamp_orderbook_update_map.items() if k <= timestamp}
    
    async def _reset_connection(self, subscription):
        self.are_new_messages_to_drop = True

        _subscription = cast(Dict[str, Any], subscription.copy())

        self.timestamp_orderbook_update_map.clear()

        await self.bfx.wss.unsubscribe(sub_id=_subscription.pop("sub_id"))
        print("I am subscribing ...")
        await self.bfx.wss.subscribe(**_subscription)
        self.order_book.clear()

    def run(self):
        self.bfx.wss.run()
