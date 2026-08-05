"""Choosing which nearby homes to show as comparables.

The forest trains on every usable home in the radius; this module picks the
handful a human should actually look at, ranked by how similar they are to the
subject rather than by distance alone.
"""

from .providers.base import Property

WEIGHTS = {
    "distance": 0.34,
    "sqft": 0.28,
    "beds": 0.12,
    "baths": 0.08,
    "age": 0.10,
    "type": 0.08,
}


def _ratio_score(a, b) -> float:
    """1.0 when equal, decaying toward 0 as the two diverge."""
    if not a or not b or a <= 0 or b <= 0:
        return 0.5  # unknown: neither reward nor punish
    return min(a, b) / max(a, b)


def similarity(subject: Property, comp: Property, radius_mi: float) -> float:
    """0–1 similarity blending size, layout, age, type, and proximity."""
    dist = max(0.0, 1.0 - (comp.distance_mi / max(radius_mi, 0.1)))

    sqft = _ratio_score(subject.sqft, comp.sqft)

    if subject.beds and comp.beds:
        beds = max(0.0, 1.0 - abs(subject.beds - comp.beds) / 3.0)
    else:
        beds = 0.5

    if subject.baths and comp.baths:
        baths = max(0.0, 1.0 - abs(subject.baths - comp.baths) / 3.0)
    else:
        baths = 0.5

    if subject.age is not None and comp.age is not None:
        age = max(0.0, 1.0 - abs(subject.age - comp.age) / 45.0)
    else:
        age = 0.5

    s_type = (subject.home_type or "").upper()
    c_type = (comp.home_type or "").upper()
    same_type = 1.0 if s_type and s_type == c_type else 0.35

    score = (
        WEIGHTS["distance"] * dist
        + WEIGHTS["sqft"] * sqft
        + WEIGHTS["beds"] * beds
        + WEIGHTS["baths"] * baths
        + WEIGHTS["age"] * age
        + WEIGHTS["type"] * same_type
    )
    # Recent closings beat active asking prices as evidence of value.
    if comp.price_basis == "sold":
        score *= 1.06
    return min(1.0, score)


def select_comps(subject: Property, candidates: list, radius_mi: float,
                 n: int = 8) -> list:
    """Return the `n` most comparable homes, nearest first among equals."""
    scored = []
    for comp in candidates:
        if not comp.price or comp.price <= 0:
            continue
        # Drop homes that are a different animal entirely (>2.5x size gap).
        if subject.sqft and comp.sqft:
            ratio = max(subject.sqft, comp.sqft) / min(subject.sqft, comp.sqft)
            if ratio > 2.5:
                continue
        scored.append((similarity(subject, comp, radius_mi), comp))

    scored.sort(key=lambda pair: (-pair[0], pair[1].distance_mi))
    out = []
    for score, comp in scored[:n]:
        comp.similarity = round(score, 3)
        out.append(comp)
    return out


def comps_summary(comps: list) -> dict:
    """Headline stats for the comp set, for the results page."""
    if not comps:
        return {}
    prices = [c.price for c in comps if c.price]
    ppsfs = [c.price_per_sqft for c in comps if c.price_per_sqft]
    sqfts = [c.sqft for c in comps if c.sqft]
    return {
        "count": len(comps),
        "price_min": min(prices) if prices else 0,
        "price_max": max(prices) if prices else 0,
        "price_median": sorted(prices)[len(prices) // 2] if prices else 0,
        "ppsf_median": sorted(ppsfs)[len(ppsfs) // 2] if ppsfs else 0,
        "sqft_median": sorted(sqfts)[len(sqfts) // 2] if sqfts else 0,
        "sold_count": sum(1 for c in comps if c.price_basis == "sold"),
        "max_distance": max((c.distance_mi for c in comps), default=0.0),
    }
