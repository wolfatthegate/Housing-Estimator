"""Provider registry."""

from .. import config
from .base import Property, PropertyProvider
from .synthetic import SyntheticProvider

__all__ = ["Property", "PropertyProvider", "SyntheticProvider", "get_provider"]

_cached = {}


def _rentcast():
    from .rentcast import RentCastProvider

    return RentCastProvider()


def _rapidapi():
    from .rapidapi import RapidApiZillowProvider

    return RapidApiZillowProvider()


def get_provider(name: str = "") -> PropertyProvider:
    """Build the configured provider, falling back to synthetic when unusable."""
    name = (name or config.PROVIDER or "auto").lower()

    if name in _cached:
        return _cached[name]

    if name == "synthetic":
        provider = SyntheticProvider()
    elif name == "rentcast":
        provider = _rentcast()
    elif name in ("rapidapi", "zillow"):
        provider = _rapidapi()
    elif name == "auto":
        # Prefer recorded sales over asking prices, then whatever is configured.
        provider = None
        for build in (_rentcast, _rapidapi):
            try:
                provider = build()
                break
            except Exception:
                continue
        provider = provider or SyntheticProvider()
    else:
        raise ValueError(f"Unknown provider {name!r}")

    _cached[name] = provider
    return provider
