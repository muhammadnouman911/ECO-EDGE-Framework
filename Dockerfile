# Use an official Python slim image for a lightweight footprint
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for this prototype)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the framework code
COPY . .

# Create a non-root user for security (best practice for journal code)
RUN useradd -m ecoedge
USER ecoedge

# Default command: Run the standard experiment scenario
CMD ["python", "experiments/run_experiment.py"]
