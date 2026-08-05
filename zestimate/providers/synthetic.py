"""Offline provider that fabricates a plausible, deterministic neighborhood.

This exists so the app runs with no API key and no network. Every number it
returns is *simulated* — the UI labels it as such. The generator is seeded from
the subject's coordinates, so a given address always yields the same market.
"""

import hashlib
import math
import random

from ..geo import GeoPoint, miles_to_deg_lat, miles_to_deg_lon
from .base import Property, PropertyProvider

HOME_TYPES = ["SINGLE_FAMILY", "SINGLE_FAMILY", "SINGLE_FAMILY", "TOWNHOUSE", "CONDO"]

#: The pool is generated once over this radius and then filtered down, which is
#: what makes a narrower search a strict subset of a wider one.
REFERENCE_RADIUS_MI = 2.0
#: Pool size over that disc — about 250 priced homes per square mile, tuned so
#: a 0.1 mi search still returns a usable handful and 2.0 mi saturates the
#: training cap.
POOL_SIZE = 3200

STREETS = [
    "Maple Ave", "Oak St", "Cedar Ln", "Birch Ct", "Willow Dr", "Chestnut St",
    "Sycamore Way", "Juniper Rd", "Aspen Ct", "Magnolia Dr", "Hawthorn Ln",
    "Laurel St", "Poplar Ave", "Spruce Ct", "Dogwood Dr", "Elmwood Rd",
]


def _seed_from(point: GeoPoint) -> int:
    key = f"{point.lat:.4f},{point.lon:.4f}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _base_ppsf(point: GeoPoint) -> float:
    """A smooth, coordinate-driven price surface, loosely US-shaped.

    Coastal longitudes and mid latitudes read as pricier. Purely cosmetic —
    it just gives each address a stable, non-uniform market level.
    """
    coastal_west = math.exp(-((point.lon + 121.5) ** 2) / 18.0)
    coastal_east = math.exp(-((point.lon + 73.5) ** 2) / 22.0)
    lat_pull = math.exp(-((point.lat - 38.0) ** 2) / 90.0)
    level = 130 + 520 * coastal_west + 380 * coastal_east + 90 * lat_pull
    return level


class SyntheticProvider(PropertyProvider):
    name = "synthetic"
    disclosure = (
        "Simulated market data — no Zillow connection. Set RAPIDAPI_KEY to pull "
        "live comparable sales."
    )

    def _market(self, point: GeoPoint):
        rng = random.Random(_seed_from(point))
        ppsf = _base_ppsf(point) * rng.uniform(0.85, 1.15)
        return rng, ppsf

    def _make_home(self, rng, point, ppsf, idx, radius_mi):
        # Uniform-in-area sampling so homes aren't clumped at the center.
        # Positions are drawn against the fixed reference radius, never the
        # requested one, so narrowing the search returns a strict subset of the
        # same homes rather than a freshly scattered set.
        r = radius_mi * math.sqrt(rng.random())
        theta = rng.uniform(0, 2 * math.pi)
        lat = point.lat + miles_to_deg_lat(r * math.sin(theta))
        lon = point.lon + miles_to_deg_lon(r * math.cos(theta), point.lat)

        home_type = rng.choice(HOME_TYPES)
        if home_type == "CONDO":
            sqft = rng.gauss(1150, 260)
            beds = rng.choice([1, 2, 2, 3])
            lot = 0.0
        elif home_type == "TOWNHOUSE":
            sqft = rng.gauss(1650, 320)
            beds = rng.choice([2, 3, 3, 4])
            lot = rng.gauss(2400, 700)
        else:
            sqft = rng.gauss(2150, 620)
            beds = rng.choice([3, 3, 4, 4, 5])
            lot = rng.gauss(9000, 3800)

        sqft = float(max(520, round(sqft / 10) * 10))
        lot = float(max(0, round(lot / 100) * 100))
        baths = max(1.0, round((beds * 0.72 + rng.uniform(-0.4, 0.9)) * 2) / 2)
        year = int(min(2025, max(1900, rng.gauss(1988, 24))))

        # Price model the forest has to recover from the features.
        quality = rng.gauss(1.0, 0.11)
        age_factor = 1.0 - min(0.26, (2026 - year) * 0.0032)
        type_factor = {"CONDO": 1.16, "TOWNHOUSE": 1.04, "SINGLE_FAMILY": 1.0}[home_type]
        size_factor = (sqft / 2000.0) ** -0.14  # $/sqft falls as homes get bigger
        lot_bonus = 1.0 + min(0.11, lot / 90000.0)
        micro = 1.0 + 0.05 * math.sin(lat * 900) + 0.05 * math.cos(lon * 900)

        price = (
            sqft * ppsf * quality * age_factor * type_factor
            * size_factor * lot_bonus * micro
        )
        price = float(round(price / 1000) * 1000)

        sold = rng.random() < 0.62
        street = STREETS[idx % len(STREETS)]
        number = 100 + (idx * 37) % 8900
        city = point.city or "Springfield"
        state = point.state or ""
        addr = f"{number} {street}, {city}{', ' + state if state else ''}"

        return Property(
            address=addr,
            lat=lat,
            lon=lon,
            price=price,
            beds=float(beds),
            baths=float(baths),
            sqft=sqft,
            lot_sqft=lot,
            year_built=year,
            home_type=home_type,
            price_basis="sold" if sold else "listed",
            sold_date=f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}" if sold else "",
            zpid=f"SIM{_seed_from(point) % 10**6:06d}{idx:03d}",
            url="",
        ).set_distance_from(point)

    def fetch_subject(self, address: str, point: GeoPoint):
        """Simulates the off-market case: the subject is never on the market."""
        return None

    def fetch_nearby(self, point: GeoPoint, radius_mi: float, limit: int) -> list:
        """Homes within `radius_mi`, drawn from one fixed neighborhood pool.

        The pool is always generated over REFERENCE_RADIUS_MI and then filtered,
        so the comp count falls with area the way a real market does and a
        tighter radius is a subset of a looser one.
        """
        rng, ppsf = self._market(point)
        reference = max(REFERENCE_RADIUS_MI, radius_mi)
        pool = [
            self._make_home(rng, point, ppsf, i, reference)
            for i in range(POOL_SIZE)
        ]
        within = [h for h in pool if h.distance_mi <= radius_mi]
        within.sort(key=lambda h: h.distance_mi)
        if len(within) <= limit:
            return within
        # Over the cap, take an even stride through the distance-sorted list
        # rather than the closest `limit`. Truncating to the nearest would make
        # every radius above ~0.9 mi collapse to the same comp set, silently
        # ignoring the radius the user asked for.
        stride = len(within) / limit
        return [within[int(i * stride)] for i in range(limit)]
