#!/usr/bin/env bash
set -euo pipefail

# QbitOS Linux Installation Script
# Usage: sudo ./deploy/install.sh

set -e

INSTALL_DIR="/opt/qbitos"
BIN_DIR="/usr/local/bin"
DATA_DIR="/var/lib/qbitos"
USER="qbitos"

echo "╔══════════════════════════════════════════════╗"
echo "║     QbitOS Linux Installation                ║"
echo "╚══════════════════════════════════════════════╝"

# Create user
if ! id -u "$USER" >/dev/null 2>&1; then
    echo "Creating user '$USER'..."
    useradd --system --no-create-home --shell /usr/sbin/nologin "$USER"
fi

# Create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR" "$DATA_DIR" "$DATA_DIR/rocksdb" "$DATA_DIR/models"
mkdir -p /etc/qbitos

# Copy binary (if built)
if [ -f "target/release/qbitos" ]; then
    echo "Installing binary..."
    cp target/release/qbitos "$BIN_DIR/qbitos"
    chmod +x "$BIN_DIR/qbitos"
else
    echo "⚠ Binary not found at target/release/qbitos"
    echo "  Build first with: cargo build --release -p qbitos-cli"
    exit 1
fi

# Copy migrations
echo "Copying migrations..."
cp -r memory/migrations "$INSTALL_DIR/migrations"

# Copy env example
if [ ! -f /etc/qbitos/qbitos.env ]; then
    echo "Creating environment config..."
    cp .env.example /etc/qbitos/qbitos.env
fi

# Set permissions
echo "Setting permissions..."
chown -R "$USER:$USER" "$DATA_DIR"
chown -R "$USER:$USER" "$INSTALL_DIR"

# Install systemd service
echo "Installing systemd service..."
cp deploy/qbitos.service /etc/systemd/system/qbitos.service
systemctl daemon-reload
systemctl enable qbitos

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Installation complete!                      ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Edit config:  sudo nano /etc/qbitos/qbitos.env  ║"
echo "║  Start:        sudo systemctl start qbitos   ║"
echo "║  Status:       sudo systemctl status qbitos  ║"
echo "║  Logs:         journalctl -u qbitos -f       ║"
echo "╚══════════════════════════════════════════════╝"
