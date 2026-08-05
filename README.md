# Home Value Estimator

Type an address — including **off-market homes that aren't listed** — and get an
estimated value backed by a random forest trained on the nearby listings and
recent sales.

The home page is a single text box. Everything else is derived.

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![model](https://img.shields.io/badge/model-RandomForest-green) ![tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)

---

## Quick start

Install once:

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

Run it:

```bash
python app.py
```

Open <http://127.0.0.1:5000>. It works immediately with no API key — see
*Where the data comes from* below for what that means.

Run the tests:

```bash
python -m pytest -q
```

### Running and stopping the server

**On macOS, port 5000 is usually taken by AirPlay Receiver** (ControlCenter
listens there). Flask will fail to bind and exit, often before you notice. Use a
different port:

```bash
FLASK_PORT=5057 ./venv/bin/python app.py
```

The `./venv/bin/python` form works without activating the venv. Add
`FLASK_DEBUG=1` to auto-reload on file save.

Stop it with `Ctrl+C`, or from another shell:

```bash
pkill -f "Python app.py"
```

> **Don't** stop it with `kill $(lsof -ti :5000)`. On macOS that PID list
> includes ControlCenter, so the command kills AirPlay rather than this app.
> (`lsof` ORs its filters, so adding `-c Python` returns *more* PIDs, not
> fewer.) Port-based killing is safe on a port you picked yourself:
> `kill $(lsof -ti :5057)`.

---

## Where the data comes from — read this first

**Zillow has no public API, and scraping zillow.com violates their Terms of
Service** (and is actively bot-blocked). So this app never touches zillow.com
directly. Instead it defines a provider interface and ships two implementations:

| Provider | When it's used | What you get |
|---|---|---|
| `synthetic` | No API key present (the default) | A deterministic, simulated neighborhood. Realistic *shape*, fabricated *numbers*. Great for development and for evaluating the model pipeline. |
| `rapidapi` | `RAPIDAPI_KEY` is set | Real listings and recent sales through a licensed third-party Zillow data API. |

To use live data, get a key for a Zillow data API on RapidAPI, then:

```bash
cp .env.example .env    # add RAPIDAPI_KEY=your_key_here
```

The UI always states which provider produced the numbers, so a simulated
estimate can never be mistaken for a real one. Swapping in a different vendor
(ATTOM, Redfin partner feeds, a local MLS export) means writing one class that
implements `fetch_subject` and `fetch_nearby` in `zestimate/providers/`.

Geocoding uses **Nominatim** (OpenStreetMap), which is free and permits ~1
request/second with a descriptive `User-Agent`. Results are cached on disk in
`.cache/`. If an address can't be geocoded — common for fictional or brand-new
addresses — the app falls back to deterministic stand-in coordinates and marks
the location as inexact rather than failing.

---

## How the estimate is produced

1. **Geocode** the address to a lat/lon.
2. **Gather** every priced home within `SEARCH_RADIUS_MI` (2 mi default). If
   fewer than 12 come back, the radius doubles until it hits 8 miles.
3. **Resolve the subject.** If the home is on the market, its facts come from
   the provider. If it's off-market — the case this app is built for — the
   facts are imputed from neighborhood medians and flagged `est.` in the UI.
4. **Fit a random forest** on those neighbors, fresh for every query.
5. **Predict**, then rank and display the 5–10 most comparable homes.

### The model

A `RandomForestRegressor` (400 trees) is fit per query on the local homes only.
A neighborhood-specific model beats a global one here: a few dozen homes on the
same streets carry more signal about one address than a national average does,
and it makes the result explainable — the comps shown *are* the training data.

- **Target:** `log(price)`. Home prices are right-skewed and multiplicative;
  training on logs keeps a single mansion from dominating the fit.
- **Features:** living area, beds, baths, lot size, age, north–south and
  east–west offset in miles from the subject (letting the forest learn
  micro-location), condo/townhouse flags, and sold-vs-listed.
- **Sample weights:** `1/(1+distance)^1.5`, scaled by evidence quality — a
  closed sale counts fully, an asking price 0.72, a provider AVM 0.5.
- **Range:** the 10th–90th percentile of the individual trees' predictions.
  This is an empirical spread across the ensemble, not a formal confidence
  interval, and the UI words it that way.
- **Reported accuracy:** k-fold cross-validated MAPE and MAE, plus out-of-bag
  R² when there are ≥20 training homes.
- **Guardrail:** every estimate is compared against a plain median-$/sqft
  benchmark. If the forest diverges more than 2× in either direction, the page
  says so instead of quietly reporting a bad number.

### Comparable selection

The forest trains on *everything* in radius; the table shows the handful worth
looking at, scored 0–1 on distance (34%), living area (28%), beds (12%), age
(10%), baths (8%), and home type (8%), with a 6% bonus for closed sales. Homes
more than 2.5× the subject's size are dropped as not-comparable.

---

## API

```bash
curl "http://127.0.0.1:5000/api/estimate?address=410+Terry+Ave+N,+Seattle,+WA+98109"
```

Returns the subject, the estimate with its range and accuracy metrics, feature
importances, and the full comparable set as JSON. Accepts the same optional
refinement parameters as the web form (`sqft`, `beds`, `baths`, `lot_sqft`,
`year_built`, `home_type`).

---

## Layout

```
app.py                       Flask routes, template filters
zestimate/
  config.py                  Env-driven settings
  geo.py                     Geocoding, haversine, address parsing
  model.py                   Feature build, random forest, Estimate
  comps.py                   Similarity scoring, comp selection
  service.py                 Pipeline: address -> ValuationReport
  viz.py                     SVG chart geometry
  providers/
    base.py                  Property + PropertyProvider interface
    synthetic.py             Offline deterministic provider
    rapidapi.py              Live licensed-API provider
templates/                   base / index / result
static/style.css             Light + dark, no JS, no CDN
tests/                       28 tests
```

The results page has no JavaScript and no external requests — the map and charts
are server-computed inline SVG, so it renders offline and in dark mode.

---

## Configuration

All optional; see `.env.example` for the full list with defaults.

| Variable | Default | Purpose |
|---|---|---|
| `ZESTIMATE_PROVIDER` | `auto` | `auto`, `rapidapi`, or `synthetic` |
| `RAPIDAPI_KEY` | — | Enables live data |
| `SEARCH_RADIUS_MI` | `2.0` | Initial comp search radius |
| `MIN_TRAINING_ROWS` | `12` | Below this, the radius expands |
| `N_COMPS_SHOWN` | `8` | Comps displayed (spec range: 5–10) |
| `MAX_SEARCH_RADIUS_MI` | `8.0` | Hard stop on radius expansion |
| `MAX_TRAINING_ROWS` | `250` | Cap on homes fed to the forest |
| `RF_N_ESTIMATORS` | `400` | Trees in the forest |
| `RF_MIN_SAMPLES_LEAF` | `2` | Leaf size floor |
| `RF_RANDOM_STATE` | `42` | Fixed, so results are reproducible |
| `FLASK_PORT` | `5000` | Change this on macOS — see above |
| `FLASK_HOST` | `127.0.0.1` | Set `0.0.0.0` to expose on your network |
| `FLASK_DEBUG` | `0` | `1` enables auto-reload |

---

## Limitations

Worth being blunt about:

- **A few dozen comps is a small training set.** The cross-validated MAPE on the
  results page is the honest read on any given estimate; treat a "Low"
  confidence badge as meaning the number is indicative, not reliable.
- **No condition, renovation, or view data.** Two identical-on-paper homes can
  differ 30% in price for reasons no listing feed captures.
- **Off-market homes are valued as a *typical* home at that location** until you
  correct the facts in the "Refine" panel. That is the single biggest accuracy
  lever available to the user.
- **Asking prices aren't sale prices.** They're down-weighted, not excluded.
- This is a statistical estimate, not an appraisal, and it is not affiliated
  with or endorsed by Zillow.
