FROM rust:1.70-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /workspace

# Set Rust environment
ENV CARGO_HOME=/tmp/.cargo
ENV RUSTUP_HOME=/tmp/.rustup
ENV PATH="/tmp/.cargo/bin:${PATH}"

# Default command
CMD ["/bin/bash"]