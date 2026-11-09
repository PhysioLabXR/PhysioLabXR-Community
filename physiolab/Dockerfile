FROM node:22-bookworm-slim AS web
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.12.9 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
COPY backend/src/ ./src/
RUN uv sync --frozen --no-dev --no-cache
COPY --from=web /app/web/dist /app/web/dist
ENV PHYSIOLAB_WEB_DIR=/app/web/dist PHYSIOLAB_DATA_DIR=/app/data
EXPOSE 8000
CMD ["sh", "-c", ".venv/bin/uvicorn physiolab.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
