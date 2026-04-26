FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src ./src

# non-editable install: the package is copied into site-packages so the runtime
# stage does not need access to /build/src (an editable install would leave a
# .pth file pointing to a path that does not exist after the multi-stage copy).
RUN uv pip install --system --no-cache ".[track_a]"


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY openenv.yaml ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()" || exit 1

CMD ["uvicorn", "nyaya_mitra.env.server:app", "--host", "0.0.0.0", "--port", "8000"]
