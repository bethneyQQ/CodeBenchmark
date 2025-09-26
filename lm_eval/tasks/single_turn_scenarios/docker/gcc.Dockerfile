FROM gcc:11

# Install additional tools
RUN apt-get update && apt-get install -y \
    make \
    cmake \
    gdb \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 sandbox
USER sandbox
WORKDIR /workspace

# Set C/C++ compilation flags for security
ENV CFLAGS="-Wall -Wextra -O2"
ENV CXXFLAGS="-Wall -Wextra -O2"

# Default command
CMD ["/bin/bash"]