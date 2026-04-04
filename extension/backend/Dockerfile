FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/root/.local/bin:/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /tmp/uv-installer.sh
RUN sh /tmp/uv-installer.sh && rm /tmp/uv-installer.sh

# Copy dependency metadata first for better layer caching
COPY translator/pyproject.toml translator/uv.lock translator/.python-version /app/translator/
COPY extension/backend/pyproject.toml extension/backend/uv.lock extension/backend/.python-version /app/extension/backend/

WORKDIR /app/extension/backend

RUN uv sync --no-install-project

WORKDIR /

# Copy source after metadata
COPY translator/src /app/translator/src
COPY extension/backend/src /app/extension/backend/src
COPY extension/backend/resources /app/extension/backend/resources
COPY extension/backend/config.yaml /app/extension/backend/config.yaml
COPY extension/backend/main.py /app/extension/backend/main.py

WORKDIR /app/extension/backend

RUN uv sync --frozen --no-dev

EXPOSE 9000

CMD ["uv", "run", "--no-dev", "poe", "prod"]