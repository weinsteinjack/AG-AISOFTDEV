# For reference, a good .dockerignore file would contain:
# .git
# .gitignore
# .dockerignore
# __pycache__/
# *.pyc
# *.pyo
# *.pyd
# .pytest_cache/
# .venv/
# venv/
# env/

# ---- Stage 1: Builder ----
# This stage installs dependencies into a virtual environment.
FROM python:3.11-slim AS builder

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /opt/venv

# Create a virtual environment
RUN python -m venv .
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel to build packages efficiently
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy requirements file and install dependencies
# This is done in a separate layer to leverage Docker's caching mechanism.
# The layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: Runtime ----
# This stage creates the final, minimal production image.
FROM python:3.11-slim

# Set the working directory in the final image
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the path to include the virtual environment's binaries
ENV PATH="/opt/venv/bin:$PATH"

# Security: Create a dedicated non-root user and group
RUN adduser --system --group appuser

# Copy application code into the container
# The context for COPY is the root of the project where Dockerfile is located.
COPY artifacts/app/ /app/

# Grant ownership of the app directory to the non-root user
# This is crucial for security and for allowing the application
# (e.g., SQLite) to write files if needed.
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set environment variables for Python again in the final stage
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the port the application will run on
EXPOSE 8000

# Define the command to run the application
# Using exec form (JSON array) is recommended.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]