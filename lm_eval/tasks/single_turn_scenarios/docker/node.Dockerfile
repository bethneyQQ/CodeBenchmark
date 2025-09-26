FROM node:18-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Install commonly used Node.js packages globally
RUN npm install -g \
    jest \
    mocha \
    chai \
    lodash \
    axios \
    express

# Create non-root user for security
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /workspace

# Set Node.js environment
ENV NODE_ENV=test
ENV NPM_CONFIG_CACHE=/tmp/.npm

# Default command
CMD ["/bin/bash"]