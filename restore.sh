#!/bin/bash
# Restore script for Jarvis AI Platform (PostgreSQL + Redis)
# Usage: ./restore.sh <backup-dir>

set -e

BACKUP_DIR=${1:-backup_latest}

if [ ! -d "$BACKUP_DIR" ]; then
  echo "[Jarvis] Backup directory $BACKUP_DIR does not exist!" >&2
  exit 1
fi

echo "[Jarvis] Restoring PostgreSQL database..."
psql "$DATABASE_URL" < "$BACKUP_DIR/jarvis_db.sql"

echo "[Jarvis] Restoring Redis..."
cat "$BACKUP_DIR/redis.rdb" | redis-cli --pipe

echo "[Jarvis] Restore complete!"
