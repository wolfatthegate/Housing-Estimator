"""RentCast provider: parsing, recency filtering, and the off-market subject.

No network. Each test hands the provider a canned /properties payload.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ZESTIMATE_PROVIDER", "synthetic")

from zestimate.geo import GeoPoint  # noqa: E402
from zestimate.providers.rentcast import RentCastProvider, _home_type, _iso_date  # noqa: E402

POINT = GeoPoint(lat=39.74, lon=-104.99, display_name="Denver, CO")


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%dT00:00:00.000Z")


def _record(**over) -> dict:
    base = {
        "id": "1234-Elm-St",
        "formattedAddress": "1234 Elm St, Denver, CO 80202",
        "latitude": 39.7405,
        "longitude": -104.9905,
        "propertyType": "Single Family",
        "bedrooms": 3,
        "bathrooms": 2.5,
        "squareFootage": 1840,
        "lotSize": 6250,
        "yearBuilt": 1998,
        "lastSalePrice": 615000,
        "lastSaleDate": _days_ago(200),
    }
    base.update(over)
    return base


class _FakeProvider(RentCastProvider):
    """Skips __init__'s key check and serves a fixed payload instead of HTTP."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def _get(self, path, params, retries=2):
        self.calls.append((path, params))
        return self.payload


# --- helpers ---------------------------------------------------------------

def test_iso_date_trims_timestamp_and_rejects_junk():
    assert _iso_date("2024-11-18T00:00:00.000Z") == "2024-11-18"
    assert _iso_date("not a date") == ""
    assert _iso_date(None) == ""


def test_home_type_maps_to_the_strings_the_model_greps_for():
    assert _home_type("Single Family") == "SINGLE_FAMILY"
    assert _home_type("Condo") == "CONDO"
    assert _home_type("Townhouse") == "TOWNHOUSE"
    assert _home_type("Houseboat") == "UNKNOWN"


# --- fetch_nearby ----------------------------------------------------------

def test_parses_a_record_into_a_sold_comp():
    p = _FakeProvider([_record()]).fetch_nearby(POINT, 2.0, 10)
    assert len(p) == 1
    c = p[0]
    assert c.price == 615000
    assert c.price_basis == "sold"          # model weights this above "listed"
    assert c.sold_date == _days_ago(200)[:10]
    assert (c.beds, c.baths, c.sqft, c.lot_sqft) == (3.0, 2.5, 1840.0, 6250.0)
    assert c.year_built == 1998
    assert c.home_type == "SINGLE_FAMILY"
    assert c.distance_mi < 0.1


def test_stale_sales_are_dropped():
    """The forest has no time feature, so an old price must not become a row."""
    payload = [
        _record(id="fresh", lastSaleDate=_days_ago(100)),
        _record(id="stale", lastSaleDate=_days_ago(4000)),
    ]
    comps = _FakeProvider(payload).fetch_nearby(POINT, 2.0, 10)
    assert [c.zpid for c in comps] == ["fresh"]


def test_records_without_a_price_or_date_are_skipped():
    payload = [
        _record(id="no-price", lastSalePrice=None),
        _record(id="no-date", lastSaleDate=None),
        _record(id="no-coords", latitude=None, longitude=None),
        _record(id="good"),
    ]
    comps = _FakeProvider(payload).fetch_nearby(POINT, 2.0, 10)
    assert [c.zpid for c in comps] == ["good"]


def test_results_outside_the_radius_are_filtered():
    far = _record(id="far", latitude=40.9, longitude=-104.99)  # ~80 mi north
    comps = _FakeProvider([far, _record(id="near")]).fetch_nearby(POINT, 2.0, 10)
    assert [c.zpid for c in comps] == ["near"]


def test_duplicate_ids_collapse():
    comps = _FakeProvider([_record(), _record()]).fetch_nearby(POINT, 2.0, 10)
    assert len(comps) == 1


def test_radius_and_limit_are_sent_to_the_api():
    prov = _FakeProvider([])
    prov.fetch_nearby(POINT, 1.5, 40)
    path, params = prov.calls[0]
    assert path == "/properties"
    assert params["radius"] == 1.5
    assert params["latitude"] == round(POINT.lat, 6)
    assert params["limit"] <= 500 and params["limit"] >= 120  # over-fetches


def test_limit_is_respected():
    payload = [_record(id=f"h{i}", latitude=39.7405 + i * 0.0005) for i in range(30)]
    assert len(_FakeProvider(payload).fetch_nearby(POINT, 2.0, 5)) == 5


def test_wrapped_payload_is_tolerated():
    comps = _FakeProvider({"properties": [_record()]}).fetch_nearby(POINT, 2.0, 10)
    assert len(comps) == 1


# --- fetch_subject ---------------------------------------------------------

def test_subject_keeps_facts_but_never_a_price():
    """A last-sale price is history; service.py reads `price` as 'on the market'."""
    s = _FakeProvider([_record()]).fetch_subject("1234 Elm St, Denver, CO", POINT)
    assert s is not None
    assert s.price is None
    assert s.price_basis == "off-market"
    assert s.sqft == 1840.0
    assert s.zpid == "1234-Elm-St"


def test_subject_is_none_when_unknown():
    assert _FakeProvider([]).fetch_subject("nowhere", POINT) is None


def test_subject_is_none_without_usable_facts():
    bare = {"id": "x", "formattedAddress": "a", "latitude": 39.74, "longitude": -104.99}
    assert _FakeProvider([bare]).fetch_subject("a", POINT) is None


# --- construction ----------------------------------------------------------

def test_missing_key_raises_a_readable_error(monkeypatch):
    monkeypatch.setattr("zestimate.config.RENTCAST_API_KEY", "")
    with pytest.raises(RuntimeError, match="RENTCAST_API_KEY"):
        RentCastProvider()


def test_key_is_sent_as_a_header():
    prov = RentCastProvider(api_key="test-key-123")
    assert prov._session.headers["X-Api-Key"] == "test-key-123"
    assert prov.base_url == "https://api.rentcast.io/v1"


# --- sale history ----------------------------------------------------------

HISTORY = {
    "2017-10-19": {"event": "Sale", "date": "2017-10-19T00:00:00.000Z", "price": 185000},
    "2024-11-18": {"event": "Sale", "date": "2024-11-18T00:00:00.000Z", "price": 310000},
}


def test_history_is_parsed_newest_first():
    comps = _FakeProvider([_record(history=HISTORY)]).fetch_nearby(POINT, 2.0, 10)
    hist = comps[0].sale_history
    assert [h["date"] for h in hist] == ["2024-11-18", "2017-10-19"]
    assert hist[0]["price"] == 310000
    assert hist[0]["event"] == "Sale"


def test_missing_or_malformed_history_is_harmless():
    for bad in (None, [], "nope", {"2020-01-01": "not a dict"}):
        comps = _FakeProvider([_record(history=bad)]).fetch_nearby(POINT, 2.0, 10)
        assert comps[0].sale_history == []


def test_history_entries_without_a_price_are_skipped():
    payload = {"2019-01-01": {"event": "Sale", "date": "2019-01-01", "price": None}}
    comps = _FakeProvider([_record(history=payload)]).fetch_nearby(POINT, 2.0, 10)
    assert comps[0].sale_history == []


def test_subject_carries_history_too():
    s = _FakeProvider([_record(history=HISTORY)]).fetch_subject("1234 Elm St", POINT)
    assert len(s.sale_history) == 2


def test_time_adjustment_widens_the_sale_window(monkeypatch):
    """Old sales are kept once trends.apply can mark them to market."""
    from zestimate import config

    old = _record(id="old", lastSaleDate=_days_ago(2500))  # ~7 years
    monkeypatch.setattr(config, "TIME_ADJUST_ENABLED", False)
    assert _FakeProvider([old]).fetch_nearby(POINT, 2.0, 10) == []

    monkeypatch.setattr(config, "TIME_ADJUST_ENABLED", True)
    monkeypatch.setattr(config, "TIME_ADJUST_MAX_SALE_AGE_DAYS", 3650)
    assert len(_FakeProvider([old]).fetch_nearby(POINT, 2.0, 10)) == 1
