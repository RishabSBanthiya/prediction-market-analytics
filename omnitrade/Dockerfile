# OmniTrade multi-platform trading system
# Usage:
#   docker build -t omnitrade .
#   docker run --env-file .env omnitrade directional --exchange polymarket --paper

# ---------- builder stage (compiles C extensions, discarded after) -----------
FROM python:3.11-slim AS builder

# System deps needed only for compiling wheels (cryptography, web3, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only pyproject.toml first so pip can resolve dependencies.
# We install deps in --target so they are easy to copy without gcc/libffi.
COPY pyproject.toml ./

# Create a minimal package stub so `pip install .` can find the package
# without copying the full source tree (preserves Docker layer cache).
RUN mkdir -p omnitrade && \
    echo '__version__ = "0.1.0"' > omnitrade/__init__.py && \
    pip install --no-cache-dir --prefix=/install .

# ---------- runtime stage (slim, no compiler toolchain) ---------------------
FROM python:3.11-slim AS runtime

# Prevent Python from writing .pyc files and enable unbuffered output for logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user for runtime security
RUN groupadd --gid 1000 omnitrade && \
    useradd --uid 1000 --gid omnitrade --create-home omnitrade

WORKDIR /app

# Copy installed packages from builder stage (no gcc/libffi in this image)
COPY --from=builder /install/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /install/bin /usr/local/bin

# Copy application code
COPY . /app

# Create data directory for SQLite databases and set ownership
RUN mkdir -p /app/data && chown -R omnitrade:omnitrade /app

USER omnitrade

# Health check: verify the bot process (run_bot.py) is actually running,
# not just that the module can be imported.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD pgrep -f "run_bot.py" > /dev/null || exit 1

# Default entrypoint runs the bot CLI; pass bot_type and flags via CMD / docker run args
ENTRYPOINT ["python", "scripts/run_bot.py"]

# Default: show help
CMD ["--help"]
