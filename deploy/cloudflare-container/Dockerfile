FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CGP_GATEWAY_HOST=0.0.0.0 \
    CGP_GATEWAY_PORT=8080 \
    CGP_GATEWAY_TRUSTED_PROXY=cloudflare-container

WORKDIR /app

COPY plugins/kaggle-gateway /app/plugins/kaggle-gateway
RUN python -m pip install --no-cache-dir /app/plugins/kaggle-gateway

EXPOSE 8080

CMD ["python", "-m", "chatgpt_plugin_kaggle_gateway.server"]
