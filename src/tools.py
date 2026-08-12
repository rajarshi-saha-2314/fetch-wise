"""
tools.py — Mock tools the FetchWise agent can call.

Currently one tool: check_order_status(order_id). In a real system this would
hit an orders database / shipping-carrier API; here it's a small in-memory
mock so the project can be run and evaluated end-to-end with no live backend.

The return shape (OrderStatusResult) is what a real implementation would also
return, so swapping the mock for a real API call wouldn't require changing
agent.py's tool-calling logic.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

ORDER_ID_PATTERN = re.compile(r"^FCH-\d{5}$")

# Mock "orders database" — one order per status mentioned in
# data/docs/03-order-tracking.md, so the eval set can exercise every branch.
_MOCK_ORDERS = {
    "FCH-10234": {
        "status": "Shipped",
        "carrier": "UPS",
        "expected_delivery": "2026-08-15",
        "items": ["Salmon & Sweet Potato Dry Food (12lb)"],
    },
    "FCH-10391": {
        "status": "Processing",
        "carrier": None,
        "expected_delivery": "2026-08-17",
        "items": ["Fetch Box - Monthly Treats Assortment"],
    },
    "FCH-10556": {
        "status": "Out for Delivery",
        "carrier": "USPS",
        "expected_delivery": "2026-08-12",
        "items": ["Stainless Steel Slow-Feed Bowl"],
    },
    "FCH-10777": {
        "status": "Delivered",
        "carrier": "FedEx",
        "expected_delivery": "2026-08-08",
        "items": ["Orthopedic Pet Bed (Large)", "Chew-Resistant Rope Toy"],
    },
    "FCH-10982": {
        "status": "Delayed",
        "carrier": "UPS",
        "expected_delivery": "2026-08-20",
        "items": ["Grain-Free Puppy Food (30lb)"],
    },
}


@dataclass
class OrderStatusResult:
    found: bool
    order_id: str
    status: Optional[str] = None
    carrier: Optional[str] = None
    expected_delivery: Optional[str] = None
    items: Optional[list[str]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def check_order_status(order_id: str) -> OrderStatusResult:
    """Mock tool: look up an order's status by order ID.

    Returns found=False with a human-readable `error` for malformed or
    unknown order IDs rather than raising, so the agent can relay a clean
    "couldn't find that order" message instead of crashing.
    """
    order_id = order_id.strip().upper()

    if not ORDER_ID_PATTERN.match(order_id):
        return OrderStatusResult(
            found=False,
            order_id=order_id,
            error=f"'{order_id}' doesn't look like a valid Fetchly order ID (expected format: FCH-XXXXX).",
        )

    order = _MOCK_ORDERS.get(order_id)
    if order is None:
        return OrderStatusResult(
            found=False,
            order_id=order_id,
            error=f"No order found with ID {order_id}.",
        )

    return OrderStatusResult(
        found=True,
        order_id=order_id,
        status=order["status"],
        carrier=order["carrier"],
        expected_delivery=order["expected_delivery"],
        items=order["items"],
    )


# --- Tool schema -------------------------------------------------------------
# OpenAI-compatible function-calling format (this is what Groq's API expects,
# and what agent.py passes as the `tools=` argument on chat completions).
# check_order_status() itself stays provider-agnostic — only this schema's
# shape would need to change if we ever swapped providers.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": (
                "Look up the current status, carrier, expected delivery date, and items "
                "for a Fetchly order, given its order ID (format FCH-XXXXX). Use this "
                "whenever a customer asks about the status, tracking, or delivery date "
                "of a specific order and provides (or can provide) an order ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The Fetchly order ID, e.g. 'FCH-10234'.",
                    }
                },
                "required": ["order_id"],
            },
        },
    }
]


if __name__ == "__main__":
    # Quick manual sanity check covering: valid order, another status,
    # well-formed-but-unknown ID, and malformed ID.
    for oid in ["FCH-10234", "FCH-10982", "FCH-99999", "not-an-id"]:
        result = check_order_status(oid)
        print(oid, "->", result.to_dict())
