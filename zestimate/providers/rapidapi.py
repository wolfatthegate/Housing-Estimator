"""Live provider backed by a licensed third-party Zillow data API.

Uses the widely available `zillow-com1` RapidAPI service. We deliberately do not
scrape zillow.com: it violates their terms of service and is bot-blocked. Point
RAPIDAPI_HOST at any equivalent vendor whose JSON matches these shapes, or
subclass and override `_search` for a different one.
"""

import time

import requests

from .. import config
from ..geo import GeoPoint, haversine_mi, miles_to_deg_lat, miles_to_deg_lon
from .base import Property, PropertyProvider

SQFT_PER_ACRE = 43560.0


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


def _lot_to_sqft(value, unit) -> float | None:
    v = _num(value)
    if v is None:
        return None
    if str(unit).lower() in ("acres", "acre"):
        return v * SQFT_PER_ACRE
    return v


def _flatten_address(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        parts = [
            raw.get("streetAddress", ""),
            raw.get("city", ""),
            raw.get("state", ""),
            raw.get("zipcode", ""),
        ]
        return ", ".join(p for p in parts if p)
    return ""


class RapidApiZillowProvider(PropertyProvider):
    name = "rapidapi-zillow"
    disclosure = "Live comparable data via a licensed Zillow data API."

    def __init__(self, api_key: str = "", host: str = ""):
        self.api_key = api_key or config.RAPIDAPI_KEY
        self.host = host or config.RAPIDAPI_HOST
        if not self.api_key:
            raise RuntimeError(
                "RAPIDAPI_KEY is not set. Set it in .env, or run with "
                "ZESTIMATE_PROVIDER=synthetic for offline demo data."
            )
        self._session = requests.Session()
        self._session.headers.update(
            {"X-RapidAPI-Key": self.api_key, "X-RapidAPI-Host": self.host}
        )

    # -- HTTP ---------------------------------------------------------------
    def _get(self, path: str, params: dict, retries: int = 2):
        url = f"https://{self.host}{path}"
        for attempt in range(retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=config.HTTP_TIMEOUT)
            except requests.RequestException:
                if attempt == retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code == 429 and attempt < retries:
                time.sleep(2.0 * (attempt + 1))  # rate limited: back off
                continue
            resp.raise_for_status()
            return resp.json()
        return None

    # -- Parsing ------------------------------------------------------------
    def _to_property(self, raw: dict, point: GeoPoint) -> Property | None:
        lat = _num(raw.get("latitude") or raw.get("lat"))
        lon = _num(raw.get("longitude") or raw.get("lng") or raw.get("lon"))
        price = _num(raw.get("price")) or _num(raw.get("zestimate"))
        if lat is None or lon is None or not price:
            return None

        status = str(raw.get("listingStatus") or raw.get("homeStatus") or "").upper()
        if "SOLD" in status or raw.get("dateSold"):
            basis = "sold"
        elif raw.get("price"):
            basis = "listed"
        else:
            basis = "estimate"

        sold_date = raw.get("dateSold") or ""
        if isinstance(sold_date, (int, float)) and sold_date > 10**11:
            sold_date = time.strftime("%Y-%m-%d", time.gmtime(sold_date / 1000))

        zpid = str(raw.get("zpid", "") or "")
        return Property(
            address=_flatten_address(raw.get("address")) or "Address unavailable",
            lat=lat,
            lon=lon,
            price=price,
            beds=_num(raw.get("bedrooms")),
            baths=_num(raw.get("bathrooms")),
            sqft=_num(raw.get("livingArea") or raw.get("livingAreaValue")),
            lot_sqft=_lot_to_sqft(
                raw.get("lotAreaValue") or raw.get("lotSize"),
                raw.get("lotAreaUnit") or "sqft",
            ),
            year_built=int(_num(raw.get("yearBuilt")) or 0) or None,
            home_type=str(raw.get("propertyType") or raw.get("homeType") or "UNKNOWN"),
            price_basis=basis,
            sold_date=str(sold_date),
            zpid=zpid,
            url=f"https://www.zillow.com/homedetails/{zpid}_zpid/" if zpid else "",
        ).set_distance_from(point)

    # -- Interface ----------------------------------------------------------
    def fetch_subject(self, address: str, point: GeoPoint) -> Property | None:
        try:
            data = self._get("/property", {"address": address})
        except requests.RequestException:
            return None
        if not isinstance(data, dict) or not data:
            return None
        prop = self._to_property(data, point)
        if prop is None:
            # Off-market homes often have facts but no price. Keep the facts.
            lat = _num(data.get("latitude")) or point.lat
            lon = _num(data.get("longitude")) or point.lon
            if not data.get("bedrooms") and not data.get("livingArea"):
                return None
            prop = Property(
                address=_flatten_address(data.get("address")) or address,
                lat=lat,
                lon=lon,
                price=None,
                beds=_num(data.get("bedrooms")),
                baths=_num(data.get("bathrooms")),
                sqft=_num(data.get("livingArea")),
                lot_sqft=_lot_to_sqft(data.get("lotSize"), data.get("lotAreaUnit") or "sqft"),
                year_built=int(_num(data.get("yearBuilt")) or 0) or None,
                home_type=str(data.get("homeType") or "UNKNOWN"),
                price_basis="off-market",
            ).set_distance_from(point)
        return prop

    def _search(self, location: str, status_type: str, pages: int = 2) -> list:
        out = []
        for page in range(1, pages + 1):
            try:
                data = self._get(
                    "/propertyExtendedSearch",
                    {"location": location, "status_type": status_type, "page": page},
                )
            except requests.RequestException:
                break
            props = (data or {}).get("props") or []
            if not props:
                break
            out.extend(props)
            if page >= int((data or {}).get("totalPages") or 1):
                break
        return out

    def fetch_nearby(self, point: GeoPoint, radius_mi: float, limit: int) -> list:
        # The vendor searches by place name, so query the locality then filter
        # to a true radius ourselves.
        locality = point.zipcode or point.short_locality
        if not locality:
            return []

        raw = self._search(locality, "RecentlySold")
        raw += self._search(locality, "ForSale")

        seen, results = set(), []
        for item in raw:
            zpid = str(item.get("zpid", ""))
            if zpid and zpid in seen:
                continue
            seen.add(zpid)
            prop = self._to_property(item, point)
            if prop and prop.distance_mi <= radius_mi:
                results.append(prop)

        results.sort(key=lambda p: (p.price_basis != "sold", p.distance_mi))
        return results[:limit]
