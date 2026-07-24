#!/usr/bin/env bash
set -euo pipefail

# QbitOS Uninstall Script (Linux/macOS)
# Usage: sudo ./deploy/uninstall.sh

INSTALL_DIR="/opt/qbitos"
BIN_DIR="/usr/local/bin"
DATA_DIR="/var/lib/qbitos"
SERVICE_FILE="/etc/systemd/system/qbitos.service"
ENV_FILE="/etc/qbitos/qbitos.env"
USER="qbitos"

echo "╔══════════════════════════════════════════════╗"
echo "║     QbitOS Uninstallation                    ║"
echo "╚══════════════════════════════════════════════╝"

# Stop service if running
if systemctl is-active --quiet qbitos 2>/dev/null; then
    echo "Stopping QbitOS service..."
    systemctl stop qbitos
fi

# Disable service
if [ -f "$SERVICE_FILE" ]; then
    echo "Removing systemd service..."
    systemctl disable qbitos 2>/dev/null || true
    rm -f "$SERVICE_FILE"
    systemctl daemon-reload
fi

# Remove binary
if [ -f "$BIN_DIR/qbitos" ]; then
    echo "Removing binary..."
    rm -f "$BIN_DIR/qbitos"
fi

# Remove install directory
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing install directory..."
    rm -rf "$INSTALL_DIR"
fi

# Remove env file
if [ -f "$ENV_FILE" ]; then
    echo "Removing environment config..."
    rm -f "$ENV_FILE"
    rmdir /etc/qbitos 2>/dev/null || true
fi

# Ask about data
if [ -d "$DATA_DIR" ]; then
    read -p "Remove data directory ($DATA_DIR)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing data directory..."
        rm -rf "$DATA_DIR"
    else
        echo "Data directory preserved at $DATA_DIR"
    fi
fi

# Remove user (optional)
if id -u "$USER" >/dev/null 2>&1; then
    read -p "Remove user '$USER'? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        userdel "$USER" 2>/dev/null || true
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Uninstallation complete.                    ║"
echo "╚══════════════════════════════════════════════╝"
