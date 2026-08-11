"""Orchestration: address in, valuation report out."""

from dataclasses import dataclass

from . import config, model
from .comps import comps_summary, select_comps
from .geo import GeoPoint, geocode
from .providers import get_provider
from .providers.base import Property
from .usps import AddressNotFoundError, verify_address


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
                  provider_name: str = "") -> ValuationReport:
    """Full pipeline for one address.

    `overrides` lets the results page refine imputed facts (sqft, beds, ...)
    without changing the single-textbox home page.
    """
    if not verify_address(address):
        raise AddressNotFoundError("Address not valid. Please enter valid address")

    point = geocode(address)
    provider = get_provider(provider_name)

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

    for key, value in (overrides or {}).items():
        if value in (None, ""):
            continue
        if hasattr(subject, key):
            setattr(subject, key, value)
            if key in subject.imputed_fields:
                subject.imputed_fields.remove(key)

    # Never let the subject itself leak into its own training data.
    training = [
        c for c in candidates
        if not (subject.zpid and c.zpid == subject.zpid) and c.distance_mi > 1e-4
    ]

    estimate = model.estimate_price(subject, training, point, radius)
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
    )
