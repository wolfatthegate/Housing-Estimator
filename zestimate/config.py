"""Runtime configuration, read once from the environment."""

import os

from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- Data provider ---------------------------------------------------------
# "auto"     -> use rapidapi if a key is present, else synthetic
# "rapidapi" -> force the live provider (errors loudly without a key)
# "synthetic"-> force the offline provider
PROVIDER = os.environ.get("ZESTIMATE_PROVIDER", "auto").lower()

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "").strip()
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST", "zillow-com1.p.rapidapi.com").strip()

# Nominatim (OpenStreetMap) is free and permits ~1 req/s with a real User-Agent.
GEOCODER_URL = os.environ.get(
    "GEOCODER_URL", "https://nominatim.openstreetmap.org/search"
)
GEOCODER_USER_AGENT = os.environ.get(
    "GEOCODER_USER_AGENT", "zestimate/1.0 (comparable-sales research tool)"
)
HTTP_TIMEOUT = _float("HTTP_TIMEOUT", 15.0)

# USPS Addresses API (OAuth2 client credentials). Without both of these set,
# address verification is skipped and every address is treated as valid.
USPS_CLIENT_ID = os.environ.get("USPS_CLIENT_ID", "").strip()
USPS_CLIENT_SECRET = os.environ.get("USPS_CLIENT_SECRET", "").strip()
USPS_BASE_URL = os.environ.get("USPS_BASE_URL", "https://apis.usps.com").strip()

# --- Comparable selection --------------------------------------------------
SEARCH_RADIUS_MI = _float("SEARCH_RADIUS_MI", 2.0)
MAX_SEARCH_RADIUS_MI = _float("MAX_SEARCH_RADIUS_MI", 8.0)
MIN_TRAINING_ROWS = _int("MIN_TRAINING_ROWS", 12)
MAX_TRAINING_ROWS = _int("MAX_TRAINING_ROWS", 250)
N_COMPS_SHOWN = _int("N_COMPS_SHOWN", 8)  # spec asks for 5-10

# --- Random forest ---------------------------------------------------------
RF_N_ESTIMATORS = _int("RF_N_ESTIMATORS", 400)
RF_MIN_SAMPLES_LEAF = _int("RF_MIN_SAMPLES_LEAF", 2)
RF_RANDOM_STATE = _int("RF_RANDOM_STATE", 42)

# --- Web -------------------------------------------------------------------
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = _int("FLASK_PORT", 5000)
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")
