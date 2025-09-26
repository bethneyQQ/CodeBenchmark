FROM openjdk:17-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    maven \
    gradle \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /workspace

# Set Java environment
ENV JAVA_OPTS="-Xmx512m -Xms128m"
ENV MAVEN_OPTS="-Xmx512m"

# Default command
CMD ["/bin/bash"]