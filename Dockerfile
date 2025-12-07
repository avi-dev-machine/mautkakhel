# ============================================================================
# AI Exercise Trainer - Hugging Face Spaces Dockerfile
# Optimized for CPU inference with cheat detection & AI analysis
# ============================================================================

FROM python:3.11-slim

# Set environment variables early
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Install system dependencies (optimized for CV + ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Video processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Utilities
    git \
    wget \
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# ============================================================================
# DEPENDENCY INSTALLATION (Layered for better caching)
# ============================================================================

# Copy requirements files first (better cache utilization)
COPY requirements.txt requirements_api.txt ./

# Install PyTorch CPU version first (largest dependency)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install Transformers and related ML packages
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    pillow==10.1.0 \
    av==11.0.0 \
    timm==0.9.12

# Install FastAPI and API dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Install core application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for AI analysis
RUN pip install --no-cache-dir \
    langchain==0.1.0 \
    langchain-google-genai==0.0.5 \
    google-generativeai==0.3.2

# Clear pip cache
RUN pip cache purge

# ============================================================================
# APPLICATION CODE
# ============================================================================

# Copy all application files
COPY server.py ./
COPY utils.py ./
COPY metrics.py ./
COPY cheat.py ./
COPY ai.py ./

# Copy YOLO pose model
COPY yolo11n-pose.pt ./

# Copy ONNX model if exists (fallback for faster inference)
COPY yolo11n-pose.onnx ./ 2>/dev/null || true

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads \
    /app/results \
    /app/.cache/transformers \
    /app/.cache/huggingface \
    /app/.cache/torch \
    && chmod -R 777 /app/uploads /app/results /app/.cache

# ============================================================================
# HUGGING FACE SPACES CONFIGURATION
# ============================================================================

# Create user for Hugging Face Spaces (security best practice)
RUN useradd -m -u 1000 user
USER user

# Set HOME to user directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user space
WORKDIR $HOME/app

# Copy everything to user directory
COPY --chown=user:user . $HOME/app

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# ============================================================================
# HEALTH CHECK
# ============================================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/api/status || exit 1

# ============================================================================
# STARTUP COMMAND
# ============================================================================

# Run FastAPI server with uvicorn (optimized for Hugging Face Spaces)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "120"]
