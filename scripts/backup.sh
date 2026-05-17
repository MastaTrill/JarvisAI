#!/bin/bash
# JarvisAI Backup Script
# Run daily via crontab: 0 2 * * * /app/scripts/backup.sh

set -e

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/app/backups"
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
pg_dump -h postgres -U jarvisadmin jarvisai_prod > $BACKUP_DIR/postgres_$DATE.sql
gzip $BACKUP_DIR/postgres_$DATE.sql

# Backup Redis
redis-cli -h redis -p 6379 BGSAVE
cp /var/lib/redis/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# Backup application data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz /app/data /app/models

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"