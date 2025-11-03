###############################
# Base FastAPI Application Image
# Following program guide recommendations (multi-stage optional for frontend)
###############################
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build deps only if needed (left minimal here)
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Layer caching: copy requirements first
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Copy backend application code (exported into artifacts/app per course flow)
COPY artifacts/app/ ./app/

# Copy artifacts directory (db seeds, diagrams, etc.)
COPY artifacts/ ./artifacts/

# Non-root execution for security
RUN adduser --system --group appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default command – FastAPI via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]