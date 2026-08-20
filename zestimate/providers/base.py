"""Provider interface and the property record every provider must emit."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..geo import GeoPoint, haversine_mi


@dataclass
class Property:
    """One home. `price` is what the model learns from or predicts."""

    address: str
    lat: float
    lon: float
    price: Optional[float] = None
    beds: Optional[float] = None
    baths: Optional[float] = None
    sqft: Optional[float] = None
    lot_sqft: Optional[float] = None
    year_built: Optional[int] = None
    home_type: str = "SINGLE_FAMILY"
    # "sold" (best signal), "listed" (asking price), "estimate" (provider AVM)
    price_basis: str = "listed"
    sold_date: str = ""
    zpid: str = ""
    url: str = ""
    distance_mi: float = 0.0
    #: 0-1 similarity to the subject; set by comps.select_comps.
    similarity: float = 0.0
    # Names of fields imputed rather than reported by the source.
    imputed_fields: list = field(default_factory=list)
    #: Prior transactions, newest first. Each: {"date", "price", "event"}.
    #: Providers that expose no history simply leave this empty.
    sale_history: list = field(default_factory=list)
    #: What `price` was before trends.apply restated it in today's dollars.
    #: None means the price is as-reported.
    original_price: Optional[float] = None
    #: The multiplier that restatement applied; 1.0 when untouched.
    time_adjust_factor: float = 1.0

    @property
    def price_per_sqft(self) -> Optional[float]:
        if self.price and self.sqft and self.sqft > 0:
            return self.price / self.sqft
        return None

    @property
    def age(self) -> Optional[int]:
        if self.year_built and self.year_built > 1500:
            return max(0, 2026 - int(self.year_built))
        return None

    @property
    def is_time_adjusted(self) -> bool:
        return self.original_price is not None

    @property
    def earlier_sales(self) -> list:
        """History entries before the sale that set `price`, for display."""
        return [h for h in self.sale_history if h.get("date", "") < (self.sold_date or "9999")]

    def set_distance_from(self, point: GeoPoint) -> "Property":
        self.distance_mi = haversine_mi(point.lat, point.lon, self.lat, self.lon)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["price_per_sqft"] = self.price_per_sqft
        d["age"] = self.age
        d["is_time_adjusted"] = self.is_time_adjusted
        return d


class PropertyProvider(ABC):
    """Source of property data. Swap implementations without touching the model."""

    name = "base"
    #: Shown in the UI so the user always knows where numbers came from.
    disclosure = ""

    @abstractmethod
    def fetch_subject(self, address: str, point: GeoPoint) -> Optional[Property]:
        """Facts for the queried home, or None if it is not in the source.

        Returning None is the expected path for a genuinely off-market home.
        """

    @abstractmethod
    def fetch_nearby(self, point: GeoPoint, radius_mi: float, limit: int) -> list:
        """Comparable homes near `point`, each a :class:`Property`."""
