"""Random-forest valuation trained on the subject home's own neighborhood.

We fit a fresh model per query rather than shipping a global one: a few dozen
nearby homes carry more signal about one address than a national model does, and
it keeps the explanation local ("these are the homes that drove the number").
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from . import config
from .geo import GeoPoint
from .providers.base import Property

# Order matters: it is reused for feature importances in the UI.
FEATURE_NAMES = [
    "sqft",
    "beds",
    "baths",
    "lot_sqft",
    "age",
    "lat_offset_mi",
    "lon_offset_mi",
    "is_condo",
    "is_townhouse",
    "is_sold",
]

PRETTY_FEATURES = {
    "sqft": "Living area",
    "beds": "Bedrooms",
    "baths": "Bathrooms",
    "lot_sqft": "Lot size",
    "age": "Age of home",
    "lat_offset_mi": "Location (N–S)",
    "lon_offset_mi": "Location (E–W)",
    "is_condo": "Condo",
    "is_townhouse": "Townhouse",
    "is_sold": "Sold vs. listed",
}


#: Human-readable names for subject fields we may have to impute.
FIELD_LABELS = {
    "sqft": "living area",
    "beds": "bedroom count",
    "baths": "bathroom count",
    "lot_sqft": "lot size",
    "year_built": "year built",
}


def _join_english(items: list) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


class NotEnoughData(RuntimeError):
    """Raised when the neighborhood has too few priced homes to model."""


@dataclass
class Estimate:
    point_estimate: float
    low: float
    high: float
    price_per_sqft: float
    baseline_ppsf_estimate: float
    n_training: int
    radius_mi: float
    cv_mape: Optional[float]
    cv_mae: Optional[float]
    r2_oob: Optional[float]
    importances: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def confidence(self) -> str:
        """Coarse label driven by sample size and cross-validated error."""
        if self.n_training < config.MIN_TRAINING_ROWS:
            return "Low"
        if self.cv_mape is None:
            return "Moderate"
        if self.cv_mape <= 0.10 and self.n_training >= 30:
            return "High"
        if self.cv_mape <= 0.18:
            return "Moderate"
        return "Low"

    @property
    def spread_pct(self) -> float:
        if not self.point_estimate:
            return 0.0
        return (self.high - self.low) / self.point_estimate


# --- Feature construction --------------------------------------------------


def _median(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(vals)) if vals else None


def neighborhood_profile(comps: list) -> dict:
    """Median attributes, used to fill in an unknown off-market subject."""
    return {
        "sqft": _median([c.sqft for c in comps]),
        "beds": _median([c.beds for c in comps]),
        "baths": _median([c.baths for c in comps]),
        "lot_sqft": _median([c.lot_sqft for c in comps]),
        "year_built": _median([c.year_built for c in comps]),
        "ppsf": _median([c.price_per_sqft for c in comps]),
    }


def synthesize_subject(address: str, point: GeoPoint, comps: list) -> Property:
    """Build a stand-in subject from neighborhood medians.

    This is the off-market path: we know where the home is but not what it is,
    so we value a *typical* home at that location and say so plainly.
    """
    prof = neighborhood_profile(comps)
    subject = Property(
        address=address,
        lat=point.lat,
        lon=point.lon,
        price=None,
        beds=prof["beds"],
        baths=prof["baths"],
        sqft=prof["sqft"],
        lot_sqft=prof["lot_sqft"],
        year_built=int(prof["year_built"]) if prof["year_built"] else None,
        home_type="SINGLE_FAMILY",
        price_basis="off-market",
    )
    subject.imputed_fields = [
        f for f in ("beds", "baths", "sqft", "lot_sqft", "year_built")
        if getattr(subject, f) is not None
    ]
    return subject


def fill_missing(subject: Property, comps: list) -> Property:
    """Patch individual gaps in a partially known subject."""
    prof = neighborhood_profile(comps)
    for field_name, key in (
        ("beds", "beds"),
        ("baths", "baths"),
        ("sqft", "sqft"),
        ("lot_sqft", "lot_sqft"),
    ):
        if getattr(subject, field_name) in (None, 0) and prof[key] is not None:
            setattr(subject, field_name, prof[key])
            subject.imputed_fields.append(field_name)
    if not subject.year_built and prof["year_built"]:
        subject.year_built = int(prof["year_built"])
        subject.imputed_fields.append("year_built")
    return subject


def _row(prop: Property, origin: GeoPoint, defaults: dict) -> list:
    home_type = (prop.home_type or "").upper()
    sqft = prop.sqft or defaults.get("sqft") or 1800.0
    year = prop.year_built or defaults.get("year_built") or 1985
    return [
        float(sqft),
        float(prop.beds if prop.beds is not None else defaults.get("beds") or 3),
        float(prop.baths if prop.baths is not None else defaults.get("baths") or 2),
        float(prop.lot_sqft if prop.lot_sqft is not None else defaults.get("lot_sqft") or 0),
        float(max(0, 2026 - int(year))),
        (prop.lat - origin.lat) * 69.0,
        (prop.lon - origin.lon) * 69.0 * math.cos(math.radians(origin.lat)),
        1.0 if "CONDO" in home_type else 0.0,
        1.0 if "TOWNHOUSE" in home_type or "TOWNHOME" in home_type else 0.0,
        1.0 if prop.price_basis == "sold" else 0.0,
    ]


def build_matrix(comps: list, origin: GeoPoint) -> tuple:
    defaults = neighborhood_profile(comps)
    X = np.array([_row(c, origin, defaults) for c in comps], dtype=float)
    y = np.array([float(c.price) for c in comps], dtype=float)
    return X, y


def sample_weights(comps: list) -> np.ndarray:
    """Closer homes and actual closings carry more weight than distant asks."""
    w = []
    for c in comps:
        proximity = 1.0 / (1.0 + c.distance_mi) ** 1.5
        basis = {"sold": 1.0, "listed": 0.72, "estimate": 0.5}.get(c.price_basis, 0.6)
        w.append(proximity * basis)
    arr = np.array(w, dtype=float)
    return arr / arr.mean() if arr.mean() > 0 else np.ones(len(comps))


# --- Fit and predict -------------------------------------------------------


def _cross_validate(X, y_log, weights) -> tuple:
    """K-fold MAPE/MAE in dollar space. None when the sample is too small."""
    n = len(y_log)
    if n < 15:
        return None, None
    k = 5 if n >= 40 else 3
    kf = KFold(n_splits=k, shuffle=True, random_state=config.RF_RANDOM_STATE)
    errs, abs_errs = [], []
    for train_idx, test_idx in kf.split(X):
        m = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
        )
        m.fit(X[train_idx], y_log[train_idx], sample_weight=weights[train_idx])
        pred = np.exp(m.predict(X[test_idx]))
        actual = np.exp(y_log[test_idx])
        errs.extend(np.abs(pred - actual) / actual)
        abs_errs.extend(np.abs(pred - actual))
    return float(np.mean(errs)), float(np.mean(abs_errs))


def estimate_price(subject: Property, comps: list, origin: GeoPoint,
                   radius_mi: float) -> Estimate:
    """Fit a random forest on `comps` and value `subject` with it."""
    priced = [c for c in comps if c.price and c.price > 0 and c.sqft]
    warnings = []

    if len(priced) < 3:
        raise NotEnoughData(
            f"Only {len(priced)} priced homes found near this address — "
            "not enough to build an estimate."
        )
    if len(priced) < config.MIN_TRAINING_ROWS:
        warnings.append(
            f"Only {len(priced)} nearby homes had usable data; treat this "
            "estimate as a rough indication."
        )

    X, y = build_matrix(priced, origin)
    y_log = np.log(y)
    weights = sample_weights(priced)

    forest = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS,
        min_samples_leaf=min(config.RF_MIN_SAMPLES_LEAF, max(1, len(priced) // 8)),
        max_features=min(1.0, 0.7),
        oob_score=len(priced) >= 20,
        bootstrap=True,
        random_state=config.RF_RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(X, y_log, sample_weight=weights)

    defaults = neighborhood_profile(priced)
    x_subject = np.array([_row(subject, origin, defaults)], dtype=float)

    # Per-tree predictions give an empirical spread, not a formal CI.
    tree_preds = np.array([t.predict(x_subject)[0] for t in forest.estimators_])
    point = float(np.exp(np.mean(tree_preds)))
    low = float(np.exp(np.percentile(tree_preds, 10)))
    high = float(np.exp(np.percentile(tree_preds, 90)))

    # Independent sanity check: neighborhood median $/sqft.
    ppsf = defaults.get("ppsf") or 0.0
    baseline = float(ppsf * (subject.sqft or defaults.get("sqft") or 0))
    if baseline > 0 and (point > baseline * 2.0 or point < baseline * 0.5):
        warnings.append(
            "The model diverges sharply from a simple $/sqft benchmark — the "
            "nearby homes may be too dissimilar to this one."
        )

    cv_mape, cv_mae = _cross_validate(X, y_log, weights)
    oob = float(forest.oob_score_) if getattr(forest, "oob_score_", None) is not None else None

    importances = sorted(
        (
            {"feature": PRETTY_FEATURES.get(n, n), "weight": float(v)}
            for n, v in zip(FEATURE_NAMES, forest.feature_importances_)
        ),
        key=lambda d: d["weight"],
        reverse=True,
    )

    if subject.imputed_fields:
        labels = [
            FIELD_LABELS.get(f, f.replace("_", " "))
            for f in sorted(set(subject.imputed_fields))
        ]
        warnings.append(
            "This home is off-market, so its "
            + _join_english(labels)
            + " were filled in from neighborhood medians. Correcting them below "
            "will sharpen the estimate."
        )

    return Estimate(
        point_estimate=point,
        low=low,
        high=high,
        price_per_sqft=point / subject.sqft if subject.sqft else 0.0,
        baseline_ppsf_estimate=baseline,
        n_training=len(priced),
        radius_mi=radius_mi,
        cv_mape=cv_mape,
        cv_mae=cv_mae,
        r2_oob=oob,
        importances=importances[:6],
        warnings=warnings,
    )
