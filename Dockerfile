# Optimized single-stage build for faster deployment with cheat detection
FROM python:3.11-slim

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy and install requirements (separate for better caching)
COPY requirements_api.txt requirements.txt ./

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers pillow av && \
    pip install --no-cache-dir -r requirements_api.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code (including cheat detection module)
COPY server.py utils.py metrics.py cheat.py ./

# Copy YOLO model
COPY yolo11n-pose.pt ./

# Create necessary directories
RUN mkdir -p uploads results

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    HF_HOME=/app/.cache/huggingface

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/api/status')" || exit 1

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
