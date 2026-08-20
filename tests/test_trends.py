"""Repeat-sales index: pair extraction, filtering, and price restatement."""

import math
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ZESTIMATE_PROVIDER", "synthetic")

from zestimate import config, model, trends  # noqa: E402
from zestimate.providers.base import Property  # noqa: E402

AS_OF = date(2026, 1, 1)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(config, "TIME_ADJUST_ENABLED", True)
    monkeypatch.setattr(config, "TIME_ADJUST_MIN_PAIRS", 3)


def home(*sales, price=None, sold_date="", basis="sold") -> Property:
    """`sales` are (date, price) pairs, oldest first."""
    return Property(
        address="x", lat=39.7, lon=-104.9,
        price=price, sold_date=sold_date, price_basis=basis,
        sale_history=[
            {"date": d, "price": float(p), "event": "Sale"} for d, p in reversed(sales)
        ],
    )


def doubling_homes(n=4) -> list:
    """Each home doubles over 10 years -> ~6.93% continuous annual rate."""
    return [home(("2010-01-01", 200000), ("2020-01-01", 400000)) for _ in range(n)]


# --- pair extraction -------------------------------------------------------

def test_pairs_come_from_consecutive_sales_of_one_home():
    rates = trends.matched_pairs([home(("2010-01-01", 200000), ("2020-01-01", 400000))])
    assert len(rates) == 1
    assert rates[0] == pytest.approx(math.log(2) / 10, rel=1e-3)


def test_three_sales_yield_two_pairs():
    h = home(("2010-01-01", 100000), ("2015-01-01", 150000), ("2020-01-01", 200000))
    assert len(trends.matched_pairs([h])) == 2


def test_a_single_sale_yields_nothing():
    assert trends.matched_pairs([home(("2020-01-01", 300000))]) == []


def test_quick_flips_are_dropped(monkeypatch):
    monkeypatch.setattr(config, "TIME_ADJUST_MIN_HOLD_YEARS", 0.75)
    flip = home(("2020-01-01", 200000), ("2020-04-01", 260000))  # 3 months
    assert trends.matched_pairs([flip]) == []


def test_implausible_rates_are_dropped(monkeypatch):
    """A gut renovation reads as appreciation; the cap is what excludes it."""
    monkeypatch.setattr(config, "TIME_ADJUST_MAX_ANNUAL_RATE", 0.35)
    reno = home(("2018-01-01", 100000), ("2020-01-01", 900000))
    assert trends.matched_pairs([reno]) == []


def test_only_sale_events_count():
    h = Property(
        address="x", lat=39.7, lon=-104.9,
        sale_history=[
            {"date": "2020-01-01", "price": 400000, "event": "Listing"},
            {"date": "2010-01-01", "price": 200000, "event": "Sale"},
        ],
    )
    assert trends.matched_pairs([h]) == []


# --- index -----------------------------------------------------------------

def test_index_is_the_median_pair_rate():
    trend = trends.build_index(doubling_homes(4), as_of=AS_OF)
    assert trend is not None
    assert trend.annual_rate == pytest.approx(math.log(2) / 10, rel=1e-3)
    assert trend.annual_pct == pytest.approx(0.0718, abs=1e-3)
    assert trend.n_pairs == 4


def test_thin_evidence_refuses_to_produce_an_index(monkeypatch):
    monkeypatch.setattr(config, "TIME_ADJUST_MIN_PAIRS", 8)
    assert trends.build_index(doubling_homes(4), as_of=AS_OF) is None


def test_disabled_flag_short_circuits(monkeypatch):
    monkeypatch.setattr(config, "TIME_ADJUST_ENABLED", False)
    assert trends.build_index(doubling_homes(20), as_of=AS_OF) is None


def test_median_ignores_a_wild_outlier():
    homes = doubling_homes(5) + [home(("2019-01-01", 100000), ("2020-01-01", 130000))]
    trend = trends.build_index(homes, as_of=AS_OF)
    assert trend.annual_rate == pytest.approx(math.log(2) / 10, rel=1e-2)


# --- applying the index ----------------------------------------------------

def test_old_sale_is_carried_forward():
    trend = trends.PriceTrend(annual_rate=math.log(2) / 10, n_pairs=9, as_of=AS_OF)
    c = home(price=300000.0, sold_date="2021-01-01")
    trends.apply([c], trend)
    assert c.original_price == 300000.0
    assert c.price == pytest.approx(300000 * 2 ** 0.5, rel=1e-3)  # 5 years
    assert c.is_time_adjusted


def test_asking_prices_are_left_alone():
    trend = trends.PriceTrend(annual_rate=0.07, n_pairs=9, as_of=AS_OF)
    c = home(price=300000.0, sold_date="2021-01-01", basis="listed")
    trends.apply([c], trend)
    assert c.price == 300000.0 and not c.is_time_adjusted


def test_no_trend_leaves_everything_untouched():
    c = home(price=300000.0, sold_date="2015-01-01")
    trends.apply([c], None)
    assert c.price == 300000.0 and c.original_price is None


def test_adjustment_is_capped(monkeypatch):
    """However old the sale, one price cannot be scaled without limit."""
    monkeypatch.setattr(config, "TIME_ADJUST_MAX_FACTOR", 1.5)
    trend = trends.PriceTrend(annual_rate=0.30, n_pairs=9, as_of=AS_OF)
    c = home(price=100000.0, sold_date="2006-01-01")  # 20 years
    trends.apply([c], trend)
    assert c.price == pytest.approx(150000.0)


def test_describe_mentions_direction_and_sample():
    trend = trends.PriceTrend(annual_rate=math.log(1.05), n_pairs=12, as_of=AS_OF)
    note = trends.describe(trend)
    assert "rising" in note and "5.0%/year" in note and "12 homes" in note
    assert trends.describe(None) == ""


# --- model interaction -----------------------------------------------------

def test_restated_prices_are_trusted_less_than_reported_ones():
    reported = home(price=400000.0, sold_date="2025-06-01")
    restated = home(price=400000.0, sold_date="2016-06-01")
    restated.original_price, restated.time_adjust_factor = 250000.0, 1.6
    for c in (reported, restated):
        c.distance_mi = 1.0
    w = model.sample_weights([reported, restated])
    assert w[0] > w[1]


def test_untouched_comps_weigh_exactly_as_before():
    a = home(price=400000.0, sold_date="2025-06-01")
    a.distance_mi = 1.0
    assert model._restatement_weight(a) == 1.0
