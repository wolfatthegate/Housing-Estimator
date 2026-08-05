"""Geocoding and distance helpers."""

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass

import requests

from . import config

EARTH_RADIUS_MI = 3958.7613


def haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def miles_to_deg_lat(miles: float) -> float:
    return miles / 69.0


def miles_to_deg_lon(miles: float, at_lat: float) -> float:
    return miles / max(69.0 * math.cos(math.radians(at_lat)), 1e-6)


def normalize_address(raw: str) -> str:
    """Collapse whitespace/punctuation so the same address caches consistently."""
    s = (raw or "").strip().lower()
    s = re.sub(r"[.,]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@dataclass
class GeoPoint:
    lat: float
    lon: float
    display_name: str
    city: str = ""
    state: str = ""
    zipcode: str = ""
    exact: bool = True  # False when we fell back to a deterministic stand-in

    @property
    def short_locality(self) -> str:
        bits = [b for b in (self.city, self.state) if b]
        return ", ".join(bits) if bits else self.display_name


class GeocodeError(RuntimeError):
    pass


def _cache_path(key: str) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(config.CACHE_DIR, f"geo_{digest}.json")


_last_call = 0.0


def _throttle(min_interval: float = 1.05) -> None:
    """Nominatim's usage policy caps us at roughly one request per second."""
    global _last_call
    wait = min_interval - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


_STATE_ZIP = re.compile(r"^\s*([A-Za-z]{2})?\s*(\d{5})(?:-\d{4})?\s*$")


def _parse_locality(raw: str) -> tuple:
    """Best-effort city/state/ZIP from a comma-separated address string."""
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    city = state = zipcode = ""
    if len(parts) >= 2:
        tail = parts[-1]
        match = _STATE_ZIP.match(tail)
        if match:  # "IL 62704" or "62704"
            state, zipcode = (match.group(1) or "").upper(), match.group(2)
            city = parts[-2]
        elif len(tail) == 2 and tail.isalpha():  # "..., Springfield, IL"
            state, city = tail.upper(), parts[-2]
        else:
            city = tail
    return city.title(), state, zipcode


def _synthetic_point(raw_address: str, key: str) -> GeoPoint:
    """Deterministic stand-in coordinates when the geocoder finds nothing.

    Keeps the app usable for unlisted or fictional addresses: the same input
    always lands on the same point, so comps and estimates stay stable.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    lat = 25.0 + (int.from_bytes(digest[0:4], "big") / 2**32) * 24.0  # 25..49 N
    lon = -124.0 + (int.from_bytes(digest[4:8], "big") / 2**32) * 57.0  # -124..-67
    city, state, zipcode = _parse_locality(raw_address)
    return GeoPoint(
        lat=round(lat, 6),
        lon=round(lon, 6),
        display_name=raw_address.strip(),
        city=city,
        state=state,
        zipcode=zipcode,
        exact=False,
    )


def geocode(address: str, allow_synthetic: bool = True) -> GeoPoint:
    """Resolve a free-text address to coordinates, with an on-disk cache."""
    key = normalize_address(address)
    if not key:
        raise GeocodeError("Please enter a street address.")

    path = _cache_path(key)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return GeoPoint(**json.load(fh))
        except (OSError, ValueError, TypeError):
            pass  # corrupt cache entry: fall through and re-fetch

    point = None
    try:
        _throttle()
        resp = requests.get(
            config.GEOCODER_URL,
            params={"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 1},
            headers={"User-Agent": config.GEOCODER_USER_AGENT},
            timeout=config.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        hits = resp.json()
        if hits:
            hit = hits[0]
            addr = hit.get("address", {}) or {}
            point = GeoPoint(
                lat=float(hit["lat"]),
                lon=float(hit["lon"]),
                display_name=hit.get("display_name", address),
                city=addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("hamlet")
                or addr.get("county", ""),
                state=addr.get("state", ""),
                zipcode=addr.get("postcode", ""),
                exact=True,
            )
    except (requests.RequestException, ValueError, KeyError):
        point = None

    if point is None:
        if not allow_synthetic:
            raise GeocodeError(
                f"Could not locate {address!r}. Try adding city, state, and ZIP."
            )
        point = _synthetic_point(address, key)

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(point.__dict__, fh)
    except OSError:
        pass
    return point
