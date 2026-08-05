"""Precomputed geometry for the inline SVG charts.

Everything is rendered as plain SVG in the template, so the page needs no
JavaScript, no CDN, and works offline.
"""

import math


def build_map_points(subject, comps, size: float = 100.0, pad: float = 10.0) -> dict:
    """Project subject + comps into a square viewbox, north-up."""
    if not comps:
        return {"subject": None, "comps": [], "scale_mi": 0.0}

    lat0 = subject.lat
    cos_lat = math.cos(math.radians(lat0)) or 1e-6

    def to_mi(p):
        return (
            (p.lon - subject.lon) * 69.0 * cos_lat,  # east
            (p.lat - subject.lat) * 69.0,            # north
        )

    pts = [to_mi(c) for c in comps]
    span = max((max(abs(x), abs(y)) for x, y in pts), default=0.5) or 0.5
    span *= 1.15
    inner = size - 2 * pad

    def project(x_mi, y_mi):
        x = pad + inner * (0.5 + x_mi / (2 * span))
        y = pad + inner * (0.5 - y_mi / (2 * span))  # SVG y grows downward
        return round(x, 2), round(y, 2)

    out = []
    for comp, (x_mi, y_mi) in zip(comps, pts):
        x, y = project(x_mi, y_mi)
        out.append({
            "x": x,
            "y": y,
            "address": comp.address,
            "price": comp.price,
            "distance_mi": comp.distance_mi,
            "sold": comp.price_basis == "sold",
        })

    sx, sy = project(0.0, 0.0)
    return {
        "subject": {"x": sx, "y": sy},
        "comps": out,
        "scale_mi": round(span, 2),
        "size": size,
    }


def build_price_bars(estimate, comps, width: float = 100.0) -> dict:
    """Horizontal bars comparing each comp's price to the estimate."""
    prices = [c.price for c in comps if c.price] + [estimate.point_estimate]
    if not prices:
        return {"bars": [], "estimate_x": 0.0}
    lo, hi = min(prices), max(prices)
    rng = (hi - lo) or hi or 1.0

    def project(value):
        return round(4 + 92 * (value - lo) / rng, 2)

    bars = [
        {
            "address": c.address,
            "price": c.price,
            "ppsf": c.price_per_sqft,
            "width": project(c.price),
            "sold": c.price_basis == "sold",
            "similarity": c.similarity,
        }
        for c in comps if c.price
    ]
    return {
        "bars": bars,
        "estimate_x": project(estimate.point_estimate),
        "low_x": project(max(lo, estimate.low)),
        "high_x": project(min(hi, estimate.high)),
        "min_price": lo,
        "max_price": hi,
    }
