#!/bin/sh
# EMA PostgreSQL backup — runs inside the compose `backup` container.
#
# Dumps the full database (pg_dump custom format, compressed) once on
# startup and then every BACKUP_INTERVAL seconds, pruning to the newest
# BACKUP_KEEP files.  Connection params come from the PGHOST / PGUSER /
# PGPASSWORD / PGDATABASE env vars set in docker-compose.yml.
#
# Restore procedure (runbook): see docs/deployment.md → Backup & Restore.

set -eu

INTERVAL="${BACKUP_INTERVAL:-3600}"
KEEP="${BACKUP_KEEP:-14}"

log() { echo "[$(date -Is)] $*"; }

dump_once() {
  ts="$(date +%Y%m%d_%H%M%S)"
  file="/backups/ema_${ts}.dump"
  if pg_dump -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -Fc -f "$file" 2>&1; then
    log "backup ok: ${file}"
  else
    log "backup FAILED: ${file}"
    return 1
  fi
  # Prune to the newest KEEP dumps (ls -t lists newest first; tail -n +N
  # keeps lines N.., i.e. everything older than the KEEP-th newest).
  ls -1t /backups/ema_*.dump 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do
    log "prune ${old}"
    rm -f "$old"
  done
  return 0
}

log "backup container started (interval=${INTERVAL}s, keep=${KEEP})"
# One immediate dump on startup, then loop.  A failed startup dump still
# enters the loop — the next interval retries.
dump_once || log "startup dump failed — will retry in ${INTERVAL}s"

while true; do
  sleep "$INTERVAL"
  dump_once || log "backup failed — will retry in ${INTERVAL}s"
done
