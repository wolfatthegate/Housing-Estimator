"""End-to-end and unit checks. Run with: venv/bin/python -m pytest -q"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ZESTIMATE_PROVIDER", "synthetic")

from zestimate import model, usps  # noqa: E402
from zestimate.comps import select_comps, similarity  # noqa: E402
from zestimate.geo import GeoPoint, _parse_locality, haversine_mi  # noqa: E402
from zestimate.providers import get_provider  # noqa: E402
from zestimate.providers.base import Property  # noqa: E402
from zestimate.service import value_address  # noqa: E402
from zestimate import service  # noqa: E402
from zestimate.usps import AddressNotFoundError  # noqa: E402
from zestimate.viz import build_map_points, build_price_bars  # noqa: E402

ADDRESS = "742 Evergreen Terrace, Springfield, IL 62704"


@pytest.fixture(scope="module")
def report():
    return value_address(ADDRESS)


# --- geo -------------------------------------------------------------------

def test_haversine_known_distance():
    # Seattle -> Portland is ~145 miles.
    d = haversine_mi(47.6062, -122.3321, 45.5152, -122.6784)
    assert 140 < d < 150


def test_haversine_zero():
    assert haversine_mi(40.0, -80.0, 40.0, -80.0) == pytest.approx(0.0)


@pytest.mark.parametrize("raw,expected", [
    ("742 Evergreen Terrace, Springfield, IL 62704", ("Springfield", "IL", "62704")),
    ("1 Main St, Boise, ID", ("Boise", "ID", "")),
    ("5 Elm St, Austin, 78701", ("Austin", "", "78701")),
])
def test_parse_locality(raw, expected):
    assert _parse_locality(raw) == expected


# --- usps --------------------------------------------------------------

def test_verify_address_skips_without_credentials(monkeypatch):
    monkeypatch.setattr(usps.config, "USPS_CLIENT_ID", "")
    monkeypatch.setattr(usps.config, "USPS_CLIENT_SECRET", "")
    assert usps.verify_address(ADDRESS) is True


def test_verify_address_rejects_blank_street(monkeypatch):
    monkeypatch.setattr(usps.config, "USPS_CLIENT_ID", "test-id")
    monkeypatch.setattr(usps.config, "USPS_CLIENT_SECRET", "test-secret")
    assert usps.verify_address("   ") is False


def test_value_address_rejects_when_usps_says_invalid(monkeypatch):
    monkeypatch.setattr(service, "verify_address", lambda addr: False)
    with pytest.raises(AddressNotFoundError, match="Address not valid"):
        value_address(ADDRESS)


# --- provider --------------------------------------------------------------

def test_synthetic_provider_is_deterministic():
    point = GeoPoint(lat=47.62, lon=-122.33, display_name="x")
    a = get_provider("synthetic").fetch_nearby(point, 2.0, 50)
    b = get_provider("synthetic").fetch_nearby(point, 2.0, 50)
    assert len(a) > 10
    assert [p.price for p in a] == [p.price for p in b]


def test_comps_lie_within_requested_radius():
    point = GeoPoint(lat=39.74, lon=-104.99, display_name="x")
    comps = get_provider("synthetic").fetch_nearby(point, 1.5, 60)
    assert all(c.distance_mi <= 1.5 + 1e-6 for c in comps)


def test_subject_is_off_market_in_synthetic_mode():
    point = GeoPoint(lat=39.74, lon=-104.99, display_name="x")
    assert get_provider("synthetic").fetch_subject("anywhere", point) is None


# --- comps -----------------------------------------------------------------

def test_similarity_peaks_for_identical_home():
    point = GeoPoint(lat=40.0, lon=-80.0, display_name="x")
    subject = Property(address="a", lat=40.0, lon=-80.0, sqft=2000, beds=3,
                       baths=2, year_built=2000, home_type="SINGLE_FAMILY")
    twin = Property(address="b", lat=40.0, lon=-80.0, sqft=2000, beds=3, baths=2,
                    year_built=2000, home_type="SINGLE_FAMILY", price=500000)
    twin.set_distance_from(point)
    far = Property(address="c", lat=40.02, lon=-80.02, sqft=5000, beds=6, baths=5,
                   year_built=1930, home_type="CONDO", price=900000)
    far.set_distance_from(point)
    assert similarity(subject, twin, 2.0) > similarity(subject, far, 2.0)


def test_select_comps_respects_count_and_drops_unpriced():
    point = GeoPoint(lat=37.77, lon=-122.42, display_name="x")
    pool = get_provider("synthetic").fetch_nearby(point, 2.0, 80)
    pool[0].price = None
    subject = model.synthesize_subject("s", point, [p for p in pool if p.price])
    chosen = select_comps(subject, pool, 2.0, n=8)
    assert 5 <= len(chosen) <= 10
    assert all(c.price for c in chosen)
    assert all(0.0 <= c.similarity <= 1.0 for c in chosen)
    # Ranked by descending similarity.
    assert chosen == sorted(chosen, key=lambda c: -c.similarity)


# --- model -----------------------------------------------------------------

def test_estimate_is_positive_and_bracketed(report):
    e = report.estimate
    assert e.point_estimate > 0
    assert e.low <= e.point_estimate <= e.high
    assert e.n_training >= 3


def test_estimate_tracks_the_ppsf_benchmark(report):
    """The forest should land in the same ballpark as median $/sqft."""
    e = report.estimate
    assert 0.5 < e.point_estimate / e.baseline_ppsf_estimate < 2.0


def test_importances_sum_to_one(report):
    total = sum(f["weight"] for f in report.estimate.importances)
    assert 0.0 < total <= 1.0 + 1e-6


def test_bigger_home_is_worth_more():
    """Monotonicity in living area — the strongest sanity check we have."""
    small = value_address(ADDRESS, overrides={"sqft": 1200}).estimate.point_estimate
    large = value_address(ADDRESS, overrides={"sqft": 3200}).estimate.point_estimate
    assert large > small


def test_pipeline_is_reproducible():
    a = value_address(ADDRESS).estimate.point_estimate
    b = value_address(ADDRESS).estimate.point_estimate
    assert a == pytest.approx(b, rel=1e-9)


def test_off_market_subject_is_imputed(report):
    assert report.is_off_market
    assert "sqft" in report.subject.imputed_fields
    assert any("off-market" in w for w in report.estimate.warnings)


def test_overrides_clear_the_imputed_flag():
    r = value_address(ADDRESS, overrides={"sqft": 2400})
    assert r.subject.sqft == 2400
    assert "sqft" not in r.subject.imputed_fields


def test_subject_excluded_from_training(report):
    assert all(c.distance_mi > 0 for c in report.comps)


def test_too_few_comps_raises():
    point = GeoPoint(lat=1.0, lon=1.0, display_name="x")
    subject = Property(address="a", lat=1.0, lon=1.0, sqft=2000)
    with pytest.raises(model.NotEnoughData):
        model.estimate_price(subject, [], point, 2.0)


# --- viz -------------------------------------------------------------------

def test_map_points_stay_inside_viewbox(report):
    data = build_map_points(report.subject, report.comps)
    assert data["subject"] is not None
    for c in data["comps"]:
        assert 0 <= c["x"] <= 100 and 0 <= c["y"] <= 100


def test_price_bars_are_within_bounds(report):
    data = build_price_bars(report.estimate, report.comps)
    assert len(data["bars"]) == len(report.comps)
    assert all(0 <= b["width"] <= 100 for b in data["bars"])
    assert 0 <= data["estimate_x"] <= 100
    assert data["low_x"] <= data["high_x"]


def test_report_shape(report):
    assert 5 <= len(report.comps) <= 10
    assert report.summary["count"] == len(report.comps)
    assert report.summary["price_min"] <= report.summary["price_max"]
    assert not math.isnan(report.estimate.point_estimate)
