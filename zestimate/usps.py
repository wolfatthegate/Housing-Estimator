"""USPS address verification, run before we spend a geocode + model fit on a query.

Uses USPS's OAuth-secured Address API (developers.usps.com), the real-time
counterpart to the NCOA move-update process: NCOALink itself is a licensed,
batch-file product built for mailers cleaning large lists, not a per-request
lookup, so a live "does this address exist" check goes through the same USPS
address-data backend via the Addresses API instead.

Without USPS_CLIENT_ID / USPS_CLIENT_SECRET configured, verification is
skipped and every address is treated as valid -- this keeps the app usable
offline against the synthetic provider (see README), the same fallback
pattern RAPIDAPI_KEY uses.
"""

import time

import requests

from . import config
from .geo import _parse_locality


class AddressNotFoundError(RuntimeError):
    """Raised when USPS can't confirm the address is real and deliverable."""


_token_cache = {"value": "", "expires": 0.0}


def _split_address(raw: str) -> tuple:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    street = parts[0] if parts else ""
    city, state, zipcode = _parse_locality(raw)
    return street, city, state, zipcode


def _get_token() -> str:
    now = time.time()
    if _token_cache["value"] and now < _token_cache["expires"]:
        return _token_cache["value"]

    resp = requests.post(
        f"{config.USPS_BASE_URL}/oauth2/v3/token",
        json={
            "grant_type": "client_credentials",
            "client_id": config.USPS_CLIENT_ID,
            "client_secret": config.USPS_CLIENT_SECRET,
        },
        timeout=config.HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["value"] = data["access_token"]
    _token_cache["expires"] = now + float(data.get("expires_in", 3600)) - 30
    return _token_cache["value"]


def verify_address(raw_address: str) -> bool:
    """True if USPS confirms this is a real, deliverable address.

    Returns True (i.e. skips the check) whenever USPS credentials aren't
    configured, or the verification call itself fails -- a down or
    unconfigured verification service should never be why a user can't get
    an estimate.
    """
    if not (config.USPS_CLIENT_ID and config.USPS_CLIENT_SECRET):
        return True

    street, city, state, zipcode = _split_address(raw_address)
    if not street:
        return False

    try:
        token = _get_token()
        resp = requests.get(
            f"{config.USPS_BASE_URL}/addresses/v3/address",
            params={
                "streetAddress": street,
                "city": city,
                "state": state,
                "ZIPCode": zipcode,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=config.HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return True

    if resp.status_code == 404:
        return False
    if resp.status_code >= 400:
        return True

    data = (resp.json() or {}).get("address") or {}
    return bool(data.get("streetAddress"))
