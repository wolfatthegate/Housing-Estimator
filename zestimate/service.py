"""Orchestration: address in, valuation report out."""

from dataclasses import dataclass, field

from . import config, model
from .comps import comps_summary, select_comps
from .geo import GeoPoint, geocode
from .providers import get_provider
from .providers.base import Property


@dataclass
class ValuationReport:
    address: str
    location: GeoPoint
    subject: Property
    estimate: model.Estimate
    comps: list
    summary: dict
    provider_name: str
    provider_disclosure: str
    is_off_market: bool
    #: True when the user pinned the radius, so no auto-widening happened.
    radius_pinned: bool = False
    #: Only the facts the user explicitly supplied, so the radius form can
    #: carry them forward without freezing values that should be re-imputed.
    applied_overrides: dict = field(default_factory=dict)


def clamp_radius(value) -> float:
    """Hold a user-supplied radius inside the supported band."""
    try:
        radius = float(value)
    except (TypeError, ValueError):
        return config.SEARCH_RADIUS_MI
    return max(config.MIN_USER_RADIUS_MI, min(config.MAX_USER_RADIUS_MI, radius))


def _priced_within(provider, point: GeoPoint, radius_mi: float) -> list:
    found = provider.fetch_nearby(point, radius_mi, config.MAX_TRAINING_ROWS)
    return [p for p in found if p.price and p.price > 0]


def _widen_until_enough(provider, point: GeoPoint) -> tuple:
    """Grow the search radius until we have enough priced homes to model."""
    radius = config.SEARCH_RADIUS_MI
    best = []
    while radius <= config.MAX_SEARCH_RADIUS_MI:
        found = provider.fetch_nearby(point, radius, config.MAX_TRAINING_ROWS)
        priced = [p for p in found if p.price and p.price > 0]
        if len(priced) > len(best):
            best = priced
        if len(priced) >= config.MIN_TRAINING_ROWS:
            return priced, radius
        radius *= 2
    return best, min(radius, config.MAX_SEARCH_RADIUS_MI)


def value_address(address: str, overrides: dict | None = None,
                  provider_name: str = "",
                  radius_mi: float | None = None) -> ValuationReport:
    """Full pipeline for one address.

    `overrides` lets the results page refine imputed facts (sqft, beds, ...)
    without changing the single-textbox home page. `radius_mi` pins the comp
    search radius; when given it is honored exactly rather than being widened
    to reach a target sample size, because silently ignoring the user's radius
    would make the control meaningless.
    """
    point = geocode(address)
    provider = get_provider(provider_name)

    pinned = radius_mi is not None
    if pinned:
        radius = clamp_radius(radius_mi)
        candidates = _priced_within(provider, point, radius)
        if len(candidates) < 3:
            raise model.NotEnoughData(
                f"Only {len(candidates)} priced home"
                f"{'' if len(candidates) == 1 else 's'} within "
                f"{radius:g} miles of this address — too few to model. "
                f"Widen the search radius and try again."
            )
    else:
        candidates, radius = _widen_until_enough(provider, point)

    if not candidates:
        raise model.NotEnoughData(
            "No nearby listings were returned for this area. Try a nearby "
            "address, or a more complete one including city, state and ZIP."
        )

    subject = provider.fetch_subject(address, point)
    is_off_market = subject is None or not subject.price

    if subject is None:
        subject = model.synthesize_subject(address, point, candidates)
    else:
        subject.address = subject.address or address
        subject = model.fill_missing(subject, candidates)

    applied = {}
    for key, value in (overrides or {}).items():
        if value in (None, ""):
            continue
        if hasattr(subject, key):
            setattr(subject, key, value)
            applied[key] = value
            if key in subject.imputed_fields:
                subject.imputed_fields.remove(key)

    # Pinning the type has to pull size and lot with it, or a townhouse keeps
    # the single-family footprint it inherited from the neighborhood medians.
    type_matched = True
    if "home_type" in applied:
        subject, type_matched = model.reimpute_for_type(subject, candidates)

    # Never let the subject itself leak into its own training data.
    training = [
        c for c in candidates
        if not (subject.zpid and c.zpid == subject.zpid) and c.distance_mi > 1e-4
    ]

    estimate = model.estimate_price(subject, training, point, radius)

    if not type_matched and subject.imputed_fields:
        pretty = (subject.home_type or "").replace("_", " ").lower()
        estimate.warnings.append(
            f"Too few {pretty} homes within {radius:g} mi to size this home from, "
            f"so its remaining details come from all nearby homes — which may not "
            f"look like a {pretty}. Widen the radius, or enter the real square "
            f"footage and lot size below."
        )

    shown = select_comps(subject, training, radius, n=config.N_COMPS_SHOWN)

    return ValuationReport(
        address=address,
        location=point,
        subject=subject,
        estimate=estimate,
        comps=shown,
        summary=comps_summary(shown),
        provider_name=provider.name,
        provider_disclosure=provider.disclosure,
        is_off_market=is_off_market,
        radius_pinned=pinned,
        applied_overrides=applied,
    )
