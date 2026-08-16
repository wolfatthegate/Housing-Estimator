FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# scikit-learn's tree builders link against OpenMP at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# The geocode cache lives on a mounted volume; a fresh named volume inherits
# the ownership of this directory, so chown it before dropping privileges.
RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# One worker on purpose: geo._throttle keeps Nominatim's 1 req/s limit in a
# module global, so each extra worker is another process that doesn't know
# about the others' requests. Threads give us concurrency without that.
# The timeout is generous because a request geocodes, fetches comps, and
# trains a 400-tree forest before it returns.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
     "--workers", "1", "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:app"]
