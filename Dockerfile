# Multi-stage build for Conduit with LiteLLM support
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./

# Install dependencies (including litellm)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[litellm]"

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY conduit/ ./conduit/
COPY conduit_litellm/ ./conduit_litellm/
COPY examples/ ./examples/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import conduit; import conduit_litellm" || exit 1

# Default command runs the demo
CMD ["python", "examples/litellm_integration.py"]
