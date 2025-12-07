# ============================================================================
# AI Exercise Trainer - Hugging Face Spaces Deployment
# FastAPI + YOLO Pose Detection + Deepfake Detection + AI Analysis
# ============================================================================

FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, video processing, and ML models
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgeos-dev \
    # Video processing
    ffmpeg \
    # Build tools (CRITICAL for PyAV)
    gcc \
    g++ \
    pkg-config \
    # PyAV dependencies
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    # Utilities
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt requirements_api.txt ./

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_api.txt && \
    pip install --no-cache-dir python-dotenv

# Copy application code
COPY server.py utils.py metrics.py cheat.py ai.py ./

# Copy YOLO model files
COPY yolo11n-pose.pt yolo11n-pose.onnx ./

# Create necessary directories with proper permissions
RUN mkdir -p uploads results /tmp/torch /tmp/transformers /tmp/huggingface && \
    chmod -R 777 uploads results /tmp/torch /tmp/transformers /tmp/huggingface

# Set environment variables for ML model caching and optimization
ENV TORCH_HOME=/tmp/torch \
    TRANSFORMERS_CACHE=/tmp/transformers \
    HF_HOME=/tmp/huggingface \
    NUMEXPR_MAX_THREADS=4 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/api/status || exit 1

# Run FastAPI server optimized for Hugging Face Spaces
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "75"]
