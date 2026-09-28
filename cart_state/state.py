"""Cart-State: a deterministic shopping-cart state object with exact ground truth.

The state has four fields:
    counts   :: dict[item -> nonneg int]
    prices   :: dict[item -> float (2 dp)]
    flags    :: dict[flag -> bool]
    discount :: float in [0, 1]   (1.0 = no discount)

Derived:
    total = round(discount * sum_i counts[i] * prices[i], 2)

The total is recomputed from first principles after every instruction so that
no incremental floating-point drift can corrupt ground truth.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any


def _round_money(x: float) -> float:
    """Round to two decimal places (banker's rounding suppressed)."""
    # round() in Python uses banker's rounding; we want commercial rounding.
    return float(f"{x + 1e-9:.2f}") if x >= 0 else -float(f"{-x + 1e-9:.2f}")


@dataclass
class CartState:
    counts: dict[str, int] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    discount: float = 1.0

    # ---------------------------------------------------------------- factory
    @classmethod
    def initial(
        cls,
        items: list[str],
        flags: list[str],
        prices: dict[str, float] | None = None,
    ) -> "CartState":
        prices = prices or {it: 1.00 for it in items}
        return cls(
            counts={it: 0 for it in items},
            prices={it: _round_money(prices[it]) for it in items},
            flags={fl: False for fl in flags},
            discount=1.0,
        )

    # ---------------------------------------------------------------- derived
    @property
    def total(self) -> float:
        raw = sum(self.counts[it] * self.prices[it] for it in self.counts)
        return _round_money(self.discount * raw)

    # ---------------------------------------------------------------- ser/de
    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "prices": {k: _round_money(v) for k, v in self.prices.items()},
            "flags": dict(self.flags),
            "discount": round(self.discount, 4),
            "total": self.total,
        }

    def to_json(self, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), sort_keys=sort_keys, separators=(", ", ": "))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CartState":
        return cls(
            counts=dict(d["counts"]),
            prices={k: float(v) for k, v in d["prices"].items()},
            flags=dict(d["flags"]),
            discount=float(d.get("discount", 1.0)),
        )

    # ---------------------------------------------------------------- copy
    def copy(self) -> "CartState":
        return copy.deepcopy(self)

    # ---------------------------------------------------------------- eq
    def equals(self, other: "CartState | dict[str, Any]") -> bool:
        if isinstance(other, dict):
            other = CartState.from_dict(other)
        return self.to_dict() == other.to_dict()
