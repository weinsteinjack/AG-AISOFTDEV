# Use the official Python image as the base for the first stage
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Use the official Python image as the base for the final stage
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . .

# Copy the installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Expose the application port
EXPOSE 8000

# Run the application
ENV PYTHONPATH=/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]