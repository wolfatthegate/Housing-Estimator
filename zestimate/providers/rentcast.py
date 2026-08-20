"""Live provider backed by RentCast's property-records API.

RentCast is built on public records and tax assessor filings rather than
listings, so it answers with *recorded sale prices* for homes that were never
advertised. That suits this app twice over: off-market homes are the whole
premise, and `model.build_features` weights a sold price above an asking price.

Docs: https://developers.rentcast.io/reference/property-records
"""

import time
from datetime import datetime, timedelta, timezone

import requests

from .. import config
from ..geo import GeoPoint
from .base import Property, PropertyProvider

#: RentCast's propertyType vocabulary -> the strings model.py greps for.
_HOME_TYPES = {
    "single family": "SINGLE_FAMILY",
    "condo": "CONDO",
    "townhouse": "TOWNHOUSE",
    "manufactured": "MANUFACTURED",
    "multi-family": "MULTI_FAMILY",
    "apartment": "APARTMENT",
    "land": "LAND",
}


def _num(value):
    """Coerce loose JSON numerics to float, or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _iso_date(value) -> str:
    """'2024-11-18T00:00:00.000Z' -> '2024-11-18'. Junk in, empty string out."""
    text = str(value or "").strip()
    return text[:10] if len(text) >= 10 and text[4] == "-" and text[7] == "-" else ""


def _home_type(value) -> str:
    return _HOME_TYPES.get(str(value or "").strip().lower(), "UNKNOWN")


def _sale_history(raw: dict) -> list:
    """RentCast's `history` object -> a newest-first list of transactions.

    Shape is {"2017-10-19": {"event": "Sale", "date": ..., "price": 185000}}.
    Every event type is kept for display; trends.matched_pairs picks out sales.
    """
    hist = raw.get("history")
    if not isinstance(hist, dict):
        return []
    out = []
    for key, entry in hist.items():
        if not isinstance(entry, dict):
            continue
        when = _iso_date(entry.get("date")) or _iso_date(key)
        price = _num(entry.get("price"))
        if not when or not price:
            continue
        out.append({
            "date": when,
            "price": price,
            "event": str(entry.get("event") or "Sale").strip() or "Sale",
        })
    out.sort(key=lambda h: h["date"], reverse=True)
    return out


class RentCastProvider(PropertyProvider):
    name = "rentcast"
    disclosure = "Recorded sale prices and property records via RentCast."

    def __init__(self, api_key: str = "", base_url: str = ""):
        self.api_key = api_key or config.RENTCAST_API_KEY
        self.base_url = (base_url or config.RENTCAST_BASE_URL).rstrip("/")
        if not self.api_key:
            raise RuntimeError(
                "RENTCAST_API_KEY is not set. Set it in .env, or run with "
                "ZESTIMATE_PROVIDER=synthetic for offline demo data."
            )
        self._session = requests.Session()
        self._session.headers.update(
            {"X-Api-Key": self.api_key, "Accept": "application/json"}
        )

    # -- HTTP ---------------------------------------------------------------
    def _get(self, path: str, params: dict, retries: int = 2):
        url = f"{self.base_url}{path}"
        for attempt in range(retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=config.HTTP_TIMEOUT)
            except requests.RequestException:
                if attempt == retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code == 404:
                return []  # "no record here" is an answer, not a failure
            if resp.status_code == 429 and attempt < retries:
                time.sleep(2.0 * (attempt + 1))  # free tier is 50 calls/month
                continue
            resp.raise_for_status()
            return resp.json()
        return None

    @staticmethod
    def _records(data) -> list:
        """/properties answers with a bare array; tolerate a wrapped object."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("properties", "results", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data] if data.get("id") or data.get("formattedAddress") else []
        return []

    # -- Parsing ------------------------------------------------------------
    def _facts(self, raw: dict, point: GeoPoint) -> Property | None:
        """Everything except the price, which the two callers treat differently."""
        lat, lon = _num(raw.get("latitude")), _num(raw.get("longitude"))
        if lat is None or lon is None:
            return None
        return Property(
            address=str(raw.get("formattedAddress") or "").strip() or "Address unavailable",
            lat=lat,
            lon=lon,
            price=None,
            beds=_num(raw.get("bedrooms")),
            baths=_num(raw.get("bathrooms")),
            sqft=_num(raw.get("squareFootage")),
            lot_sqft=_num(raw.get("lotSize")),  # already square feet
            year_built=int(_num(raw.get("yearBuilt")) or 0) or None,
            home_type=_home_type(raw.get("propertyType")),
            zpid=str(raw.get("id") or ""),
            sale_history=_sale_history(raw),
        ).set_distance_from(point)

    def _to_comp(self, raw: dict, point: GeoPoint, cutoff: str) -> Property | None:
        """A training row: needs a price, and a sale recent enough to trust."""
        price = _num(raw.get("lastSalePrice"))
        sold_date = _iso_date(raw.get("lastSaleDate"))
        if not price or not sold_date:
            return None
        # The forest has no time dimension (see model.FEATURE_NAMES), so a 2009
        # sale would be learned as if it closed today. Drop anything stale.
        if cutoff and sold_date < cutoff:
            return None
        prop = self._facts(raw, point)
        if prop is None:
            return None
        prop.price = price
        prop.price_basis = "sold"
        prop.sold_date = sold_date
        return prop

    # -- Interface ----------------------------------------------------------
    def fetch_subject(self, address: str, point: GeoPoint) -> Property | None:
        try:
            data = self._get("/properties", {"address": address})
        except requests.RequestException:
            return None
        records = self._records(data)
        if not records:
            return None

        prop = self._facts(records[0], point)
        if prop is None or (not prop.beds and not prop.sqft):
            return None
        # A last-sale price is history, not an asking price. Leaving price unset
        # keeps service.value_address treating this as the off-market case.
        prop.price_basis = "off-market"
        prop.address = prop.address if prop.address != "Address unavailable" else address
        return prop

    def fetch_nearby(self, point: GeoPoint, radius_mi: float, limit: int) -> list:
        # With time adjustment on, old sales are an asset rather than a hazard:
        # trends.apply marks them to market, so keep a much longer window.
        max_age = (
            config.TIME_ADJUST_MAX_SALE_AGE_DAYS
            if config.TIME_ADJUST_ENABLED
            else config.RENTCAST_MAX_SALE_AGE_DAYS
        )
        cutoff = ""
        if max_age > 0:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=max_age)
            ).strftime("%Y-%m-%d")

        # Most records carry no recent sale, so ask for more than we need.
        # RentCast caps `limit` at 500.
        try:
            data = self._get(
                "/properties",
                {
                    "latitude": round(point.lat, 6),
                    "longitude": round(point.lon, 6),
                    "radius": round(max(radius_mi, 0.1), 3),
                    "limit": min(500, max(limit * 3, 100)),
                },
            )
        except requests.RequestException:
            return []

        seen, results = set(), []
        for item in self._records(data):
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or "")
            if key and key in seen:
                continue
            seen.add(key)
            prop = self._to_comp(item, point, cutoff)
            if prop and prop.distance_mi <= radius_mi:
                results.append(prop)

        results.sort(key=lambda p: (p.distance_mi, -(p.price or 0)))
        return results[:limit]
