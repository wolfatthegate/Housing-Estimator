# Home Value Estimator

Type an address — including **off-market homes that aren't listed** — and get an
estimated value backed by a random forest trained on the nearby listings and
recent sales.

The home page is a single text box. Everything else is derived.

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![model](https://img.shields.io/badge/model-RandomForest-green) ![tests](https://img.shields.io/badge/tests-68%20passing-brightgreen)

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
directly. Instead it defines a provider interface and ships three
implementations:

| Provider | When it's used | What you get |
|---|---|---|
| `synthetic` | No API key present (the default) | A deterministic, simulated neighborhood. Realistic *shape*, fabricated *numbers*. Great for development and for evaluating the model pipeline. |
| `rentcast` | `RENTCAST_API_KEY` is set | Public records and tax assessor filings: **recorded sale prices**, including homes that were never listed. |
| `rapidapi` | `RAPIDAPI_KEY` is set | Real listings and recent sales through a licensed third-party Zillow data API. |

`auto` picks the first of those that has a key, preferring `rentcast`.

To use live data, copy the example env file and add one key:

```bash
cp .env.example .env    # add RENTCAST_API_KEY=... or RAPIDAPI_KEY=...
```

### Why RentCast is preferred over a listings API

`build_features` scores a sold price above an asking price — `sold` weighs
`1.0`, `listed` `0.72`, `estimate` `0.5` — and `is_sold` is a model feature in
its own right. A listings API mostly surfaces *asking* prices; RentCast is built
on public records, so nearly every row it returns is a closed sale. It feeds the
forest the signal the forest already trusts most, and it covers off-market
homes, which is the case this app exists to handle.

The tradeoff is recency. RentCast returns whatever the *last* recorded sale was,
which may be decades old, and the model has no time dimension — a 2009 price
would be learned as if it closed today. So the provider drops sales older than
`RENTCAST_MAX_SALE_AGE_DAYS` (default 3 years). Widen it in thin markets where
comps are scarce; tighten it where prices move fast.

### Sale history and the repeat-sales index

RentCast also returns a `history` object per property — every recorded
transaction, not just the last one. Each comp's prior sales appear under its
address in the results table, and the full list is in the JSON API as
`sale_history`.

That history buys something better than a display detail. When one home sells
twice, the ratio of the two prices measures appreciation *with the house held
constant* — same lot, same street, largely the same structure — so it needs no
model to interpret. `trends.py` collects those matched pairs across the
neighborhood, takes the median annualized rate, and uses it to restate old sale
prices in today's dollars. It is the Case-Shiller/FHFA repeat-sales method at
neighborhood scale, computed from data the app already fetched.

Turn it on with `TIME_ADJUST_ENABLED=1`. Two things then change: the provider
keeps sales back to `TIME_ADJUST_MAX_SALE_AGE_DAYS` (10 years) instead of 3,
and every stale price is marked to market before the forest sees it. The
results page states the rate it used and shows the original price beside each
restated one, so nothing is silently rewritten.

**What it cannot see.** A home renovated between sales looks like appreciation;
a distressed sale followed by a flip looks like a boom. Neither is modeled.
They're excluded by holding period (`TIME_ADJUST_MIN_HOLD_YEARS`) and a rate cap
(`TIME_ADJUST_MAX_ANNUAL_RATE`), the median resists what survives those, and
below `TIME_ADJUST_MIN_PAIRS` matched pairs the index refuses to produce a
number at all rather than report a noisy one. Restated prices also carry the
index's uncertainty, so `sample_weights` trusts them in proportion to how far
they were moved.

The UI always states which provider produced the numbers, so a simulated
estimate can never be mistaken for a real one. Swapping in a different vendor
(ATTOM, Redfin partner feeds, a local MLS export) means writing one class that
implements `fetch_subject` and `fetch_nearby` in `zestimate/providers/` —
`rentcast.py` is the shortest example to copy.

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
importances, and the full comparable set as JSON — each comp carrying its
`sale_history`, and a top-level `price_trend` when time adjustment applied. Accepts the same optional
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
  trends.py                  Repeat-sales index, time adjustment
  viz.py                     SVG chart geometry
  providers/
    base.py                  Property + PropertyProvider interface
    synthetic.py             Offline deterministic provider
    rentcast.py              Public-records provider (recorded sales)
    rapidapi.py              Live licensed-API provider
templates/                   base / index / result
static/style.css             Light + dark, no JS, no CDN
tests/                       68 tests
Dockerfile                   Production image (gunicorn)
docker-compose.yml           App + Caddy, persistent cache volume
deploy/                      systemd unit + shared-Caddy site block
.github/workflows/deploy.yml Redeploy on push to main
```

The results page has no JavaScript and no external requests — the map and charts
are server-computed inline SVG, so it renders offline and in dark mode.

---

## Configuration

All optional; see `.env.example` for the full list with defaults.

| Variable | Default | Purpose |
|---|---|---|
| `ZESTIMATE_PROVIDER` | `auto` | `auto`, `rapidapi`, or `synthetic` |
| `RENTCAST_API_KEY` | — | Enables live data from public records |
| `RAPIDAPI_KEY` | — | Enables live data from a Zillow data API |
| `SEARCH_RADIUS_MI` | `2.0` | Initial comp search radius |
| `MIN_TRAINING_ROWS` | `12` | Below this, the radius expands |
| `N_COMPS_SHOWN` | `8` | Comps displayed (spec range: 5–10) |
| `MAX_SEARCH_RADIUS_MI` | `8.0` | Hard stop on radius expansion |
| `MAX_TRAINING_ROWS` | `250` | Cap on homes fed to the forest |
| `RF_N_ESTIMATORS` | `400` | Trees in the forest |
| `RF_MIN_SAMPLES_LEAF` | `2` | Leaf size floor |
| `RF_RANDOM_STATE` | `42` | Fixed, so results are reproducible |
| `RENTCAST_MAX_SALE_AGE_DAYS` | `1095` | Ignore sales older than this; `0` disables |
| `TIME_ADJUST_ENABLED` | `0` | `1` restates old sales in today's dollars |
| `TIME_ADJUST_MAX_SALE_AGE_DAYS` | `3650` | Sale window once adjustment is on |
| `TIME_ADJUST_MIN_PAIRS` | `8` | Fewer matched pairs than this ⇒ no adjustment |
| `TIME_ADJUST_MIN_HOLD_YEARS` | `0.75` | Shorter holds are flips, not market movement |
| `TIME_ADJUST_MAX_ANNUAL_RATE` | `0.35` | Pairs implying more are renovations |
| `TIME_ADJUST_MAX_FACTOR` | `2.0` | Cap on any single price restatement |
| `FLASK_PORT` | `5000` | Change this on macOS — see above |
| `FLASK_HOST` | `127.0.0.1` | Set `0.0.0.0` to expose on your network |
| `FLASK_DEBUG` | `0` | `1` enables auto-reload |

---

## Deployment

Live at <https://housing.wolfatthegate.com>, sharing a t3.micro with
another app. Pushing to `main` rebuilds and restarts it — nothing to run by
hand.

**This app cannot go on GitHub Pages.** Pages serves static files only, and
every request here geocodes an address, calls out to USPS and RapidAPI/RentCast,
and trains a random forest before rendering. It needs a Python process and a
place to keep secrets.

### Why a plain instance rather than a serverless platform

- `geo.py` caches geocodes on disk and sleeps to honor Nominatim's ~1 req/s
  limit. Ephemeral containers throw that cache away on every deploy and rotate
  outbound IPs, which is exactly what a rate limiter punishes.
- Training 400 trees per request is CPU-bound, so a cold start on a
  scale-to-zero platform lands on the user as a multi-second wait.

An always-on box with a real disk sidesteps both — and since one was already
running, the marginal cost of this app is **$0**.

### How it shares the box

The instance is a **t3.micro (1 GB)** already serving
`dentai-demo.wolfatthegate.com` through Caddy. Two consequences shape this
setup:

- **No second Caddy.** It would fight the running one for ports 80 and 443.
  gunicorn binds `127.0.0.1:8000` and the shared Caddy proxies to it.
- **No Docker.** `dockerd` plus `containerd` cost ~100 MiB resident, which is
  10% of this box, and building an image on it would spike memory hard enough
  to disturb the other site. gunicorn runs from a venv under systemd instead.

```
              :443  ┌─────────────────┐
   internet ───────▶│  Caddy (shared) │
                    └────────┬────────┘
             dentai-demo ────┤
             housing ────────┴──▶ 127.0.0.1:8000  ──▶ gunicorn (systemd)
```

`Dockerfile` and `docker-compose.yml` are kept for local containerized runs and
for hosts with room to spare; production does not use them.

### One-time setup

Ubuntu 24.04's system Python is 3.12, and every pinned dependency publishes a
`cp312` manylinux wheel, so nothing compiles.

1. **Swap first.** 1 GB is tight, and `pip install` of numpy plus scikit-learn
   is the peak. Without swap it can be OOM-killed — and the kernel may pick the
   *other* app as its victim.

   ```bash
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
   sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

2. **Clone and build the venv:**

   ```bash
   sudo apt-get update && sudo apt-get install -y git python3-venv
   git clone https://github.com/wolfatthegate/Housing-Estimator.git ~/Housing-Estimator
   cd ~/Housing-Estimator && python3 -m venv venv
   ./venv/bin/pip install -r requirements.txt
   cp .env.example .env    # add RENTCAST_API_KEY
   ```

3. **Install the service:**

   ```bash
   sudo cp deploy/housing-estimator.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now housing-estimator
   systemctl is-active housing-estimator
   ```

4. **Let CI restart it** without a password:

   ```bash
   echo 'ubuntu ALL=(ALL) NOPASSWD: /bin/systemctl restart housing-estimator' \
     | sudo tee /etc/sudoers.d/housing-estimator
   sudo chmod 440 /etc/sudoers.d/housing-estimator
   ```

5. **Add the Caddy site block** and reload:

   ```bash
   sudo mkdir -p /etc/caddy/sites
   sudo cp deploy/housing-estimator.caddy /etc/caddy/sites/
   grep -q 'import /etc/caddy/sites' /etc/caddy/Caddyfile \
     || echo 'import /etc/caddy/sites/*.caddy' | sudo tee -a /etc/caddy/Caddyfile
   sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
   ```

   Validate before reloading: a syntax error plus a reload takes the other site
   down with it.

6. **Point DNS** — a Route 53 `A` record for `housing.wolfatthegate.com`.
7. **Add GitHub secrets:** `EC2_HOST`, `EC2_USER` (`ubuntu`), `EC2_SSH_KEY`.

### Notes

The unit sets `MemoryHigh=380M` and `MemoryMax=450M`. On a shared 1 GB box that
matters: without a cap, a pathological query training a large forest could push
the kernel into killing whichever process it likes, including the other site's.
With one, this service is the only thing at risk.

Gunicorn runs **one worker with four threads** on purpose. The Nominatim
throttle in `geo.py` is a module-level global, so a second worker process would
be a second unsynchronized rate limiter. Raise `--threads` for concurrency;
raising `--workers` needs a shared throttle first.

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
