#!/usr/bin/env bash
set -euo pipefail

# QbitOS Backup Script
# Backs up: state snapshot, PostgreSQL, RocksDB data

BACKUP_DIR="${1:-/backup}"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/qbitos-$DATE"

echo "Creating backup at $BACKUP_PATH..."
mkdir -p "$BACKUP_PATH"

# State snapshot
if command -v qbitos &>/dev/null; then
    echo "Saving state snapshot..."
    qbitos state save "$BACKUP_PATH/state.json" || echo "  (state save failed)"
fi

# PostgreSQL backup
if [ -n "${DATABASE_URL:-}" ]; then
    echo "Backing up PostgreSQL..."
    pg_dump "$DATABASE_URL" > "$BACKUP_PATH/postgres.sql" 2>/dev/null || echo "  (pg_dump failed)"
fi

# RocksDB backup (if path exists)
if [ -d "${QBITOS_ROCKSDB_PATH:-/var/lib/qbitos/rocksdb}" ]; then
    echo "Backing up RocksDB..."
    tar czf "$BACKUP_PATH/rocksdb.tar.gz" -C "$(dirname "${QBITOS_ROCKSDB_PATH:-/var/lib/qbitos/rocksdb}")" "$(basename "${QBITOS_ROCKSDB_PATH:-/var/lib/qbitos/rocksdb}")" 2>/dev/null || echo "  (rocksdb backup failed)"
fi

# Compress
echo "Compressing..."
tar czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "qbitos-$DATE"
rm -rf "$BACKUP_PATH"

echo "Backup complete: $BACKUP_PATH.tar.gz"
echo "Size: $(du -h "$BACKUP_PATH.tar.gz" | cut -f1)"
