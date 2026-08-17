FROM python:3.12-slim

RUN useradd --create-home --uid 1000 appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CGP_GATEWAY_HOST=0.0.0.0

WORKDIR /app

COPY --chown=appuser:appuser plugins/kaggle-gateway /app/plugins/kaggle-gateway

RUN python -m pip install --no-cache-dir /app/plugins/kaggle-gateway

USER appuser

CMD ["kaggle-gateway"]
