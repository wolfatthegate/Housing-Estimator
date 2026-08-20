"""Flask front end: one text box in, a comp-backed valuation out."""

from datetime import datetime

from flask import Flask, jsonify, render_template, request

from zestimate import config, trends
from zestimate.geo import GeocodeError
from zestimate.model import NotEnoughData
from zestimate.service import value_address
from zestimate.usps import AddressNotFoundError
from zestimate.viz import build_map_points, build_price_bars

app = Flask(__name__)

NUMERIC_OVERRIDES = {
    "sqft": float,
    "beds": float,
    "baths": float,
    "lot_sqft": float,
    "year_built": int,
}
 

def _collect_overrides(source) -> dict:
    """Pull optional refinement fields off the results-page form."""
    out = {} 
    for key, cast in NUMERIC_OVERRIDES.items():
        raw = (source.get(key) or "").strip() if hasattr(source, "get") else ""
        if not raw:
            continue
        try:
            value = cast(float(raw))
        except (TypeError, ValueError):
            continue
        if value > 0:
            out[key] = value
    home_type = (source.get("home_type") or "").strip()
    if home_type:
        out["home_type"] = home_type
    return out


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/estimate", methods=["GET", "POST"])
def estimate():
    source = request.form if request.method == "POST" else request.args
    address = (source.get("address") or "").strip()

    if not address:
        return render_template("index.html", error="Enter an address to get started."), 400

    try:
        report = value_address(address, overrides=_collect_overrides(source))
    except AddressNotFoundError as exc:
        return render_template("index.html", error=str(exc), address=address), 400
    except GeocodeError as exc:
        return render_template("index.html", error=str(exc), address=address), 400
    except NotEnoughData as exc:
        return render_template("index.html", error=str(exc), address=address), 404
    except Exception as exc:  # noqa: BLE001 - surface a readable message, log the rest
        app.logger.exception("Valuation failed for %r", address)
        return render_template(
            "index.html",
            error=f"Something went wrong while valuing that address: {exc}",
            address=address,
        ), 500

    return render_template(
        "result.html",
        r=report,
        e=report.estimate,
        s=report.subject,
        comps=report.comps,
        summary=report.summary,
        map_data=build_map_points(report.subject, report.comps),
        bar_data=build_price_bars(report.estimate, report.comps),
        n_trees=config.RF_N_ESTIMATORS,
        trend_note=trends.describe(report.price_trend),
    )


@app.route("/api/estimate", methods=["GET"])
def api_estimate():
    """JSON version of the same pipeline."""
    address = (request.args.get("address") or "").strip()
    if not address:
        return jsonify({"error": "address is required"}), 400
    try:
        report = value_address(address, overrides=_collect_overrides(request.args))
    except AddressNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except (GeocodeError, NotEnoughData) as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("API valuation failed for %r", address)
        return jsonify({"error": str(exc)}), 500

    e = report.estimate
    return jsonify({
        "address": report.address,
        "resolved_address": report.location.display_name,
        "coordinates": {"lat": report.location.lat, "lon": report.location.lon},
        "off_market": report.is_off_market,
        "subject": report.subject.to_dict(),
        "estimate": {
            "point": round(e.point_estimate),
            "low": round(e.low),
            "high": round(e.high),
            "price_per_sqft": round(e.price_per_sqft, 2),
            "baseline_ppsf_estimate": round(e.baseline_ppsf_estimate),
            "confidence": e.confidence,
            "cv_mape": e.cv_mape,
            "cv_mae": e.cv_mae,
            "oob_r2": e.r2_oob,
            "training_homes": e.n_training,
            "search_radius_mi": e.radius_mi,
            "feature_importances": e.importances,
            "warnings": e.warnings,
        },
        "comparables": [c.to_dict() for c in report.comps],
        "comp_summary": report.summary,
        "data_source": {
            "provider": report.provider_name,
            "disclosure": report.provider_disclosure,
        },
        "price_trend": {
            "annual_pct": round(report.price_trend.annual_pct, 4),
            "matched_pairs": report.price_trend.n_pairs,
            "as_of": report.price_trend.as_of.isoformat(),
        } if report.price_trend else None,
    })


@app.context_processor
def inject_year():
    """The footer copyright year, so it never goes stale in the template."""
    return {"year": datetime.now().year}


@app.template_filter("money")
def money(value):
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


@app.template_filter("num")
def num(value, digits=0):
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


@app.template_filter("pct")
def pct(value, digits=1):
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
