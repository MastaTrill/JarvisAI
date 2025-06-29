#!/bin/bash
# Backup script for Jarvis AI Platform (PostgreSQL + Redis)
# Usage: ./backup.sh <backup-dir>

set -e

BACKUP_DIR=${1:-backup_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$BACKUP_DIR"

echo "[Jarvis] Backing up PostgreSQL database..."
pg_dump "$DATABASE_URL" > "$BACKUP_DIR/jarvis_db.sql"

echo "[Jarvis] Backing up Redis..."
redis-cli --rdb "$BACKUP_DIR/redis.rdb"

echo "[Jarvis] Backup complete! Files saved in $BACKUP_DIR"
