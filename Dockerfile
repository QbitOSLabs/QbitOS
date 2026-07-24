# syntax=docker/dockerfile:1.7
# QbitOS Dockerfile — multi-stage build, pinned to Rust 1.85.0 (MSRV).
#
# Build targets:
#   runtime  — stripped, optimized image for production (default).
#   debug    — same binary, but with debug symbols and no strip, for profiling.
#
# Build args:
#   BUILDKIT_INLINE_CACHE=1   enables client-side cache reuse (set by CI).
#
# Usage:
#   docker build -t qbitos:latest --target runtime .
#   docker build -t qbitos:debug --target debug .

ARG BUILDKIT_INLINE_CACHE=1

# ─── Builder stage ───────────────────────────────────────────────────────────
FROM rust:1.85.0-bookworm AS builder
WORKDIR /build

# OpenContainers labels — emitted on the builder image too for traceability.
LABEL org.opencontainers.image.source="https://github.com/qbitos/qbitos" \
      org.opencontainers.image.license="Apache-2.0" \
      maintainer="QbitOS Project <maintainers@qbitos.org>"

# Install system dependencies for RocksDB (libclang) and other native libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libclang-dev libssl-dev pkg-config cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifest and source.
COPY Cargo.toml Cargo.lock ./
COPY kernel/ kernel/
COPY memory/ memory/
COPY planner/ planner/
COPY skills/ skills/
COPY tools/ tools/
COPY neural_engine/ neural_engine/
COPY vision/ vision/
COPY speech/ speech/
COPY world_model/ world_model/
COPY evolution/ evolution/
COPY data/ data/
COPY reasoning/ reasoning/
COPY learning/ learning/
COPY safety/ safety/
COPY training/ training/
COPY services/ services/
COPY algorithms/ algorithms/
COPY multilingual/ multilingual/
COPY cli/ cli/
COPY desktop/ desktop/
COPY examples_pkg/ examples_pkg/

# Build release binary. The workspace profile.release strips symbols & uses LTO,
# so the resulting binary is small. The `debug` target rebuilds with a debug
# profile to retain symbols — see the profile override below.
RUN cargo build --release -p qbitos-cli

# ─── Runtime base (shared by runtime & debug targets) ────────────────────────
FROM debian:bookworm-slim AS runtime-base
LABEL org.opencontainers.image.source="https://github.com/qbitos/qbitos" \
      org.opencontainers.image.license="Apache-2.0" \
      org.opencontainers.image.title="QbitOS" \
      org.opencontainers.image.description="A Rust 2026 native multimodal cognitive AI operating system" \
      maintainer="QbitOS Project <maintainers@qbitos.org>"

# Install runtime deps + curl for healthcheck + tini for proper signal handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 curl tini \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for the runtime image.
RUN groupadd --system --gid 1000 qbitos \
    && useradd --system --uid 1000 --gid qbitos \
       --no-create-home --shell /usr/sbin/nologin qbitos

# Create data directory owned by the non-root user.
RUN mkdir -p /var/lib/qbitos /opt/qbitos \
    && chown -R qbitos:qbitos /var/lib/qbitos /opt/qbitos

# Copy binary.
COPY --from=builder /build/target/release/qbitos /usr/local/bin/qbitos

# Copy migrations — required for the postgres backend to apply schema on startup.
COPY --from=builder --chown=qbitos:qbitos /build/memory/migrations/ /opt/qbitos/migrations/

VOLUME ["/var/lib/qbitos"]

EXPOSE 8080

ENV QBITOS_DATA_DIR=/var/lib/qbitos
ENV RUST_LOG=info
ENV QBITOS_LOG_FORMAT=json
ENV QBITOS_MIGRATIONS_DIR=/opt/qbitos/migrations

# Healthcheck: curl the /health endpoint every 30s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/api/v1/health || exit 1

# Use tini as PID 1 for proper signal forwarding and zombie reaping.
ENTRYPOINT ["/usr/bin/tini", "--", "qbitos"]
CMD ["serve", "--addr", "0.0.0.0:8080"]

# ─── Runtime target (production, stripped) ───────────────────────────────────
# The builder already strips via workspace profile.release; nothing extra to do.
FROM runtime-base AS runtime
LABEL org.opencontainers.image.target="runtime"
USER qbitos

# ─── Debug target (with debug symbols, no strip) ─────────────────────────────
# Rebuilds the CLI with a debug profile so perf/dtrace/samply can resolve
# symbols. Image is ~3-5x larger than runtime.
FROM builder AS debug-builder
RUN cargo build --release -p qbitos-cli \
    --config 'profile.release.strip="none"' \
    --config 'profile.release.debug=true'

FROM runtime-base AS debug
LABEL org.opencontainers.image.target="debug"
# Replace the stripped binary with the symbol-rich one.
COPY --from=debug-builder /build/target/release/qbitos /usr/local/bin/qbitos
USER qbitos
