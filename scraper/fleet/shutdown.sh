#!/bin/bash
# Spot preemption shutdown handler. GCE invokes this on SIGTERM and gives
# us up to 90s before forced shutdown. Final rsync of any unsynced images.

set +e
LOG=/var/log/bp-scraper-shutdown.log
exec >> "$LOG" 2>&1
echo "==== shutdown $(date -Is) ===="

ATTR() {
  curl -fs -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

INSTANCE_IDX=$(ATTR instance_idx)
BUCKET=$(ATTR bucket)
[ -z "$INSTANCE_IDX" ] || [ -z "$BUCKET" ] && { echo "missing metadata; bail"; exit 0; }

GCS_IMG="gs://$BUCKET/img/$INSTANCE_IDX"
GCS_FAIL="gs://$BUCKET/failures/$INSTANCE_IDX.csv"
GCS_LABELS="gs://$BUCKET/labels/$INSTANCE_IDX.csv"

# Race the 90s preemption window.
gcloud storage rsync /opt/bp/data/img "$GCS_IMG" --recursive 2>&1 || true
[ -f /opt/bp/data/labels.csv ] && \
  gcloud storage cp /opt/bp/data/labels.csv "$GCS_LABELS" || true
[ -f /opt/bp/data/scrape_failures.csv ] && \
  gcloud storage cp /opt/bp/data/scrape_failures.csv "$GCS_FAIL" || true

echo "shutdown rsync complete $(date -Is)"
