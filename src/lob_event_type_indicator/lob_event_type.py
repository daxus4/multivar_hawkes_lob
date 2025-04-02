from enum import Enum


class LOBEventType(Enum):
    # CHANGER means that the order changed the best bid/ask price
    # REMOVE_LIQUIDITY means that the order is a market order or is cancelled

    BID_REMOVE_LIQUIDITY_ORDER_CHANGER = 1
    ASK_REMOVE_LIQUIDITY_ORDER_CHANGER = 2
    BID_REMOVE_LIQUIDITY_ORDER_NOT_CHANGER = 3
    ASK_REMOVE_LIQUIDITY_ORDER_NOT_CHANGER = 4
    BID_LIMIT_ORDER_CHANGER = 5
    ASK_LIMIT_ORDER_CHANGER = 6
    BID_LIMIT_ORDER_NOT_CHANGER = 7
    ASK_LIMIT_ORDER_NOT_CHANGER = 8
    BID_MARKET_ORDER_CHANGER = 9
    ASK_MARKET_ORDER_CHANGER = 10
    BID_MARKET_ORDER_NOT_CHANGER = 11
    ASK_MARKET_ORDER_NOT_CHANGER = 12
    BID_CANCELLED_ORDER_CHANGER = 13
    ASK_CANCELLED_ORDER_CHANGER = 14
    BID_CANCELLED_ORDER_NOT_CHANGER = 15
    ASK_CANCELLED_ORDER_NOT_CHANGER = 16

    @classmethod
    def get_side_prefix(cls, is_bid: bool) -> str:
        return "BID" if is_bid else "ASK"

    @classmethod
    def get_order_action_type_string(
        cls, is_liquidity_added: bool, is_market_order: bool = False
    ) -> str:
        if is_liquidity_added:
            if is_market_order:
                raise Exception("A market order cannot add liquidity!")
            return "LIMIT_ORDER"

        return "MARKET_ORDER" if is_market_order else "CANCELLED_ORDER"

    @classmethod
    def get_best_price_impact_string(cls, is_changer: bool) -> str:
        return "CHANGER" if is_changer else "NOT_CHANGER"

    @classmethod
    def get_value_from_conditions(
        cls,
        is_bid: bool,
        is_liquidity_added: bool,
        is_changer: bool,
        is_market_order: bool = False,
    ) -> "LOBEventType":
        return getattr(
            cls,
            "_".join(
                [
                    LOBEventType.get_side_prefix(is_bid),
                    LOBEventType.get_order_action_type_string(
                        is_liquidity_added, is_market_order
                    ),
                    LOBEventType.get_best_price_impact_string(is_changer),
                ]
            ),
        )
