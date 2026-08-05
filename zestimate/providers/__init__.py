"""Provider registry."""

from .. import config
from .base import Property, PropertyProvider
from .synthetic import SyntheticProvider

__all__ = ["Property", "PropertyProvider", "SyntheticProvider", "get_provider"]

_cached = {}


def get_provider(name: str = "") -> PropertyProvider:
    """Build the configured provider, falling back to synthetic when unusable."""
    name = (name or config.PROVIDER or "auto").lower()

    if name in _cached:
        return _cached[name]

    if name == "synthetic":
        provider = SyntheticProvider()
    elif name in ("rapidapi", "zillow", "auto"):
        try:
            from .rapidapi import RapidApiZillowProvider

            provider = RapidApiZillowProvider()
        except Exception:
            if name != "auto":
                raise
            provider = SyntheticProvider()
    else:
        raise ValueError(f"Unknown provider {name!r}")

    _cached[name] = provider
    return provider
