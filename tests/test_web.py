"""Flask route tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ZESTIMATE_PROVIDER", "synthetic")

from app import app as flask_app  # noqa: E402

ADDRESS = "742 Evergreen Terrace, Springfield, IL 62704"


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_home_page_has_one_address_box(client):
    body = client.get("/").get_data(as_text=True)
    assert body.count('name="address"') == 1
    assert 'type="text"' in body


def test_estimate_renders_price_and_comps(client):
    resp = client.post("/estimate", data={"address": ADDRESS})
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Estimated value" in body
    assert "comparable homes" in body
    assert "How the model decided" in body


def test_blank_address_is_rejected(client):
    resp = client.post("/estimate", data={"address": "   "})
    assert resp.status_code == 400
    assert "Enter an address" in resp.get_data(as_text=True)


def test_api_returns_full_payload(client):
    resp = client.get("/api/estimate", query_string={"address": ADDRESS})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["estimate"]["point"] > 0
    assert data["estimate"]["low"] <= data["estimate"]["point"] <= data["estimate"]["high"]
    assert 5 <= len(data["comparables"]) <= 10
    assert data["off_market"] is True
    assert data["subject"]["sqft"] > 0
    assert data["data_source"]["provider"] == "synthetic"
    for comp in data["comparables"]:
        assert comp["price"] > 0
        assert comp["distance_mi"] >= 0


def test_api_requires_address(client):
    assert client.get("/api/estimate").status_code == 400


def test_api_honors_overrides(client):
    base = client.get("/api/estimate", query_string={"address": ADDRESS}).get_json()
    big = client.get(
        "/api/estimate", query_string={"address": ADDRESS, "sqft": 4000}
    ).get_json()
    assert big["subject"]["sqft"] == 4000
    assert big["estimate"]["point"] > base["estimate"]["point"]
