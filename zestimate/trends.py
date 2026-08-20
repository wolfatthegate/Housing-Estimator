"""Local price trend from repeat sales, used to restate old prices as today's.

The forest has no time dimension: a 2016 sale and a 2026 sale are two rows that
differ only in their features, so an old price is learned as if it closed this
morning. Rather than discard old sales, we measure how much the neighborhood
moved and mark them to market first.

The measurement uses matched pairs. When one home sells twice, the ratio of the
two prices is appreciation with the house held constant -- same lot, same
street, largely the same structure -- so it needs no hedonic model to interpret.
The median pair rate across the neighborhood is the index. This is the idea
behind the Case-Shiller and FHFA repeat-sales indices, at neighborhood scale.

What it cannot see: a home renovated between sales looks like appreciation, and
a distressed sale followed by a flip looks like a boom. Both are filtered by
holding period and rate caps rather than modeled, and a thin sample is refused
outright instead of reported quietly.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone

from . import config


def _parse(text) -> date | None:
    try:
        return datetime.strptime(str(text)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


@dataclass
class PriceTrend:
    """A neighborhood's annual appreciation, as a continuous log rate."""

    annual_rate: float
    n_pairs: int
    as_of: date

    @property
    def annual_pct(self) -> float:
        """The rate as the percentage a reader expects: 0.05 -> 5.1%."""
        return math.exp(self.annual_rate) - 1.0

    def factor_for(self, sold: date) -> float:
        """Multiplier carrying a sale on `sold` forward to `as_of`."""
        years = (self.as_of - sold).days / 365.25
        if years <= 0:
            return 1.0
        factor = math.exp(self.annual_rate * years)
        cap = max(1.0, config.TIME_ADJUST_MAX_FACTOR)
        return min(max(factor, 1.0 / cap), cap)


def matched_pairs(properties: list) -> list:
    """Annualized log rates from consecutive sales of the same home."""
    min_hold = config.TIME_ADJUST_MIN_HOLD_YEARS
    max_rate = config.TIME_ADJUST_MAX_ANNUAL_RATE
    rates = []

    for prop in properties:
        sales = []
        for entry in getattr(prop, "sale_history", None) or []:
            # Listing events move with the market too, but only a closed sale is
            # a transaction; mixing asks into the pairs would bias the index.
            if "sale" not in str(entry.get("event", "sale")).lower():
                continue
            when, price = _parse(entry.get("date")), entry.get("price")
            if when and isinstance(price, (int, float)) and price > 0:
                sales.append((when, float(price)))
        if len(sales) < 2:
            continue
        sales.sort()

        for (d1, p1), (d2, p2) in zip(sales, sales[1:]):
            years = (d2 - d1).days / 365.25
            if years < min_hold:
                continue  # a flip, or the same transaction recorded twice
            rate = math.log(p2 / p1) / years
            if abs(rate) > max_rate:
                continue  # renovation, teardown, or a bad record
            rates.append(rate)
    return rates


def build_index(properties: list, as_of: date | None = None) -> PriceTrend | None:
    """Median matched-pair rate, or None when the evidence is too thin."""
    if not config.TIME_ADJUST_ENABLED:
        return None

    rates = matched_pairs(properties)
    if len(rates) < max(2, config.TIME_ADJUST_MIN_PAIRS):
        return None

    rates.sort()
    mid = len(rates) // 2
    median = rates[mid] if len(rates) % 2 else (rates[mid - 1] + rates[mid]) / 2.0
    return PriceTrend(
        annual_rate=float(median),
        n_pairs=len(rates),
        as_of=as_of or datetime.now(timezone.utc).date(),
    )


def apply(properties: list, trend: PriceTrend | None) -> list:
    """Restate each sold price in `as_of` dollars. Mutates and returns the list."""
    if trend is None:
        return properties

    for prop in properties:
        sold = _parse(prop.sold_date)
        if not sold or not prop.price or prop.price_basis != "sold":
            continue  # an asking price is already current
        factor = trend.factor_for(sold)
        if abs(factor - 1.0) < 1e-9:
            continue
        prop.original_price = prop.price
        prop.time_adjust_factor = factor
        prop.price = prop.price * factor
    return properties


def describe(trend: PriceTrend | None) -> str:
    """One sentence for the UI, or empty when nothing was adjusted."""
    if trend is None:
        return ""
    direction = "rising" if trend.annual_rate > 0 else "falling"
    return (
        f"Older sale prices were restated in today's dollars using a local "
        f"{direction} trend of {trend.annual_pct * 100:.1f}%/year, measured from "
        f"{trend.n_pairs} homes that sold twice."
    )
