"""End-to-end and unit checks. Run with: venv/bin/python -m pytest -q"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ZESTIMATE_PROVIDER", "synthetic")

from zestimate import model  # noqa: E402
from zestimate.comps import select_comps, similarity  # noqa: E402
from zestimate.geo import GeoPoint, _parse_locality, haversine_mi  # noqa: E402
from zestimate.providers import get_provider  # noqa: E402
from zestimate.providers.base import Property  # noqa: E402
from zestimate import config  # noqa: E402
from zestimate.service import clamp_radius, value_address  # noqa: E402
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


# --- radius control --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (0.01, 0.1),      # below the floor
    (0.1, 0.1),       # at the floor
    (0.75, 0.75),     # inside the band
    (2.0, 2.0),       # at the ceiling
    (9.0, 2.0),       # above the ceiling
    ("nonsense", config.SEARCH_RADIUS_MI),
])
def test_clamp_radius(raw, expected):
    assert clamp_radius(raw) == pytest.approx(expected)


def test_radius_bounds_are_the_requested_range():
    assert config.MIN_USER_RADIUS_MI == pytest.approx(0.1)
    assert config.MAX_USER_RADIUS_MI == pytest.approx(2.0)


@pytest.mark.parametrize("radius", [0.1, 0.25, 0.5, 1.0, 2.0])
def test_pinned_radius_is_honored_exactly(radius):
    """The whole point of the control: no silent auto-widening."""
    r = value_address(ADDRESS, radius_mi=radius)
    assert r.radius_pinned is True
    assert r.estimate.radius_mi == pytest.approx(radius)
    assert all(c.distance_mi <= radius + 1e-6 for c in r.comps)


def test_out_of_range_radius_is_clamped_not_rejected():
    assert value_address(ADDRESS, radius_mi=50).estimate.radius_mi == pytest.approx(2.0)
    assert value_address(ADDRESS, radius_mi=0.001).estimate.radius_mi == pytest.approx(0.1)


def test_auto_radius_is_not_marked_pinned():
    assert value_address(ADDRESS).radius_pinned is False


def test_tighter_radius_yields_fewer_training_homes():
    tight = value_address(ADDRESS, radius_mi=0.25).estimate.n_training
    loose = value_address(ADDRESS, radius_mi=1.0).estimate.n_training
    assert tight < loose


def test_radius_changes_the_estimate():
    """A control that never changes the answer is not a control."""
    values = {
        value_address(ADDRESS, radius_mi=r).estimate.point_estimate
        for r in (0.25, 0.5, 1.0, 2.0)
    }
    assert len(values) == 4


def test_narrow_search_is_a_subset_of_wide_search():
    point = GeoPoint(lat=47.62, lon=-122.33, display_name="x")
    provider = get_provider("synthetic")
    wide = {h.zpid for h in provider.fetch_nearby(point, 2.0, 5000)}
    narrow = {h.zpid for h in provider.fetch_nearby(point, 0.5, 5000)}
    assert narrow and narrow < wide


def test_comp_count_scales_with_area():
    """Doubling the radius should roughly quadruple the pool, not hold flat."""
    point = GeoPoint(lat=47.62, lon=-122.33, display_name="x")
    provider = get_provider("synthetic")
    small = len(provider.fetch_nearby(point, 0.25, 100000))
    big = len(provider.fetch_nearby(point, 0.5, 100000))
    assert big > 2.5 * small


# --- home type -------------------------------------------------------------

def test_townhouse_is_valued_below_single_family():
    """The reported bug: a townhome priced as a single-family home."""
    sfh = value_address(ADDRESS, overrides={"home_type": "SINGLE_FAMILY"},
                        radius_mi=0.5).estimate.point_estimate
    town = value_address(ADDRESS, overrides={"home_type": "TOWNHOUSE"},
                         radius_mi=0.5).estimate.point_estimate
    condo = value_address(ADDRESS, overrides={"home_type": "CONDO"},
                          radius_mi=0.5).estimate.point_estimate
    assert town < sfh
    assert condo < town


def test_home_type_override_pulls_size_and_lot_with_it():
    sfh = value_address(ADDRESS, overrides={"home_type": "SINGLE_FAMILY"},
                        radius_mi=0.5).subject
    town = value_address(ADDRESS, overrides={"home_type": "TOWNHOUSE"},
                         radius_mi=0.5).subject
    assert town.sqft < sfh.sqft
    assert town.lot_sqft < sfh.lot_sqft


def test_explicit_size_survives_a_type_override():
    """Re-imputation must not clobber a fact the user actually supplied."""
    r = value_address(ADDRESS, overrides={"home_type": "TOWNHOUSE", "sqft": 1650},
                      radius_mi=0.5)
    assert r.subject.sqft == 1650


def test_type_fallback_is_disclosed_not_hidden():
    """A tight radius can hold too few same-type homes; say so."""
    r = value_address(ADDRESS, overrides={"home_type": "TOWNHOUSE"}, radius_mi=0.1)
    assert any("Too few townhouse" in w for w in r.estimate.warnings)


def test_no_type_fallback_warning_when_enough_same_type():
    r = value_address(ADDRESS, overrides={"home_type": "TOWNHOUSE"}, radius_mi=0.5)
    assert not any("Too few" in w for w in r.estimate.warnings)


def test_dominant_home_type_prefers_the_local_majority():
    point = GeoPoint(lat=40.0, lon=-80.0, display_name="x")
    def make(kind, dist):
        p = Property(address=kind, lat=40.0, lon=-80.0, price=1, home_type=kind)
        p.distance_mi = dist
        return p
    comps = [make("CONDO", 0.1), make("CONDO", 0.2), make("SINGLE_FAMILY", 0.3)]
    assert model.dominant_home_type(comps) == "CONDO"


def test_dominant_home_type_falls_back_when_unknown():
    assert model.dominant_home_type([]) == "SINGLE_FAMILY"


def test_off_market_type_is_flagged_as_imputed(report):
    assert "home_type" in report.subject.imputed_fields


def test_applied_overrides_records_only_user_input():
    r = value_address(ADDRESS, overrides={"sqft": 2000})
    assert r.applied_overrides == {"sqft": 2000}
    assert value_address(ADDRESS).applied_overrides == {}


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
