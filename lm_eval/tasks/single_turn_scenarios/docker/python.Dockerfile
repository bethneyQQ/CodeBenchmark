FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages commonly used in evaluations
RUN pip install --no-cache-dir \
    pytest \
    numpy \
    pandas \
    requests \
    matplotlib \
    scipy \
    scikit-learn

# Create non-root user for security
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /workspace

# Set resource limits via environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["/bin/bash"]