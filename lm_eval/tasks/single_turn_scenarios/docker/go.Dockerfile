FROM golang:1.21-alpine

# Install system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    make \
    git

# Create non-root user for security
RUN adduser -D -u 1000 sandbox
USER sandbox
WORKDIR /workspace

# Set Go environment
ENV GO111MODULE=on
ENV GOPROXY=https://proxy.golang.org,direct
ENV GOSUMDB=sum.golang.org
ENV CGO_ENABLED=1

# Default command
CMD ["/bin/sh"]