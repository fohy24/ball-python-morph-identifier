#!/bin/bash
# Startup script for a Spot ball-python scraper instance.
#
# Reads instance attributes:
#   range_start, range_end, instance_idx, bucket
# Self-deletes when finished. Designed to be re-runnable: on Spot preemption,
# a fresh instance with the same range_start/range_end will rsync down its
# previous output from GCS and resume.

set -euo pipefail

LOG=/var/log/bp-scraper.log
exec > >(tee -a "$LOG") 2>&1
echo "==== bp-scraper startup $(date -Is) ===="

META() {
  curl -fs -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"
}
ATTR() {
  curl -fs -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

INSTANCE_NAME=$(META name)
ZONE_FULL=$(META zone)
ZONE=${ZONE_FULL##*/}
RANGE_START=$(ATTR range_start 2>/dev/null || echo "")
RANGE_END=$(ATTR range_end 2>/dev/null || echo "")
INSTANCE_IDX=$(ATTR instance_idx)
BUCKET=$(ATTR bucket)
NUM_WORKERS=$(ATTR num_workers 2>/dev/null || echo 1)
[ -z "$NUM_WORKERS" ] && NUM_WORKERS=1
# Phase 1/2 chunk-driven mode: instance gets a chunk_url pointing at a
# (n,url) CSV in GCS instead of using --mode full + --start/--end.
CHUNK_URL=$(ATTR chunk_url 2>/dev/null || echo "")
LABELS_ONLY=$(ATTR labels_only 2>/dev/null || echo "0")
[ -z "$LABELS_ONLY" ] && LABELS_ONLY="0"
PROJECT=$(curl -fs -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/project/project-id")

echo "instance=$INSTANCE_NAME zone=$ZONE idx=$INSTANCE_IDX range=[$RANGE_START,$RANGE_END) bucket=$BUCKET num_workers=$NUM_WORKERS chunk_url=$CHUNK_URL labels_only=$LABELS_ONLY"

GCS_IMG="gs://$BUCKET/img/$INSTANCE_IDX"
GCS_FAIL="gs://$BUCKET/failures/$INSTANCE_IDX.csv"
GCS_LABELS="gs://$BUCKET/labels/$INSTANCE_IDX.csv"
GCS_DONE="gs://$BUCKET/done/$INSTANCE_IDX.marker"

# IMPORTANT: scraper resolves DATA_DIR relative to its own parent.parent. Put
# the script in scraper/ so DATA_DIR == /opt/bp/data, matching what we rsync.
WORK=/opt/bp
mkdir -p "$WORK/scraper" "$WORK/data/img"

# --- Skip if already done (idempotent re-runs) ---
if gcloud storage ls "$GCS_DONE" >/dev/null 2>&1; then
  echo "done marker already present at $GCS_DONE; self-deleting"
  gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
  exit 0
fi

# --- Install dependencies ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget xvfb python3-pip python3-venv >/dev/null
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O /tmp/chrome.deb
apt-get install -y -qq /tmp/chrome.deb >/dev/null
python3 -m venv /opt/venv
/opt/venv/bin/pip install --quiet undetected-chromedriver beautifulsoup4 requests pandas
echo "deps installed: chrome=$(google-chrome --version), python=$(/opt/venv/bin/python --version)"

# --- Pull code + URL list ---
gcloud storage cp "gs://$BUCKET/code/rescrape_images.py" "$WORK/scraper/rescrape_images.py"
gcloud storage cp "gs://$BUCKET/code/all_urls.csv" "$WORK/data/all_urls.csv"
touch "$WORK/data/complete_url.csv" "$WORK/data/selected_gene_df.csv"

# --- Pull per-instance chunk if we're in chunk mode ---
CHUNK_LOCAL=""
if [ -n "$CHUNK_URL" ]; then
  CHUNK_LOCAL="$WORK/data/chunk.csv"
  gcloud storage cp "$CHUNK_URL" "$CHUNK_LOCAL"
  echo "chunk pulled: $(wc -l < "$CHUNK_LOCAL") rows from $CHUNK_URL"
fi

# --- Resume: rsync down prior progress for this instance index ---
# In labels-only mode we don't need any images locally, so skip the rsync.
if [ "$LABELS_ONLY" != "1" ]; then
  echo "syncing prior progress from $GCS_IMG"
  gcloud storage rsync "$GCS_IMG" "$WORK/data/img" --recursive 2>/dev/null || true
  ls "$WORK/data/img" | wc -l | xargs -I{} echo "{} images already on disk"
fi
if [ "$NUM_WORKERS" -le 1 ]; then
  gcloud storage cp "$GCS_LABELS" "$WORK/data/labels.csv" 2>/dev/null || true
else
  for w in $(seq 0 $((NUM_WORKERS - 1))); do
    gcloud storage cp "gs://$BUCKET/labels/${INSTANCE_IDX}-w${w}.csv" "$WORK/data/labels-w${w}.csv" 2>/dev/null || true
  done
fi

# --- Background uploader: rsync to GCS every 60s ---
# Globs handle both single-worker (labels.csv -> labels/IDX.csv) and
# multi-worker (labels-wN.csv -> labels/IDX-wN.csv) layouts.
(
  while true; do
    sleep 60
    if [ "$LABELS_ONLY" != "1" ]; then
      gcloud storage rsync "$WORK/data/img" "$GCS_IMG" --recursive 2>/dev/null || true
    fi
    for f in "$WORK"/data/labels*.csv; do
      [ -e "$f" ] || continue
      bn=$(basename "$f" .csv)
      gcloud storage cp "$f" "gs://$BUCKET/labels/${INSTANCE_IDX}${bn#labels}.csv" 2>/dev/null || true
    done
    for f in "$WORK"/data/scrape_failures*.csv; do
      [ -e "$f" ] || continue
      bn=$(basename "$f" .csv)
      gcloud storage cp "$f" "gs://$BUCKET/failures/${INSTANCE_IDX}${bn#scrape_failures}.csv" 2>/dev/null || true
    done
  done
) &
UPLOADER_PID=$!

# --- Build the scraper invocation args ---
SCRAPER_ARGS=()
if [ -n "$CHUNK_LOCAL" ]; then
  SCRAPER_ARGS+=(--listings-csv "$CHUNK_LOCAL")
else
  SCRAPER_ARGS+=(--mode full --url-csv ../data/all_urls.csv --start "$RANGE_START" --end "$RANGE_END")
fi
if [ "$LABELS_ONLY" = "1" ]; then
  SCRAPER_ARGS+=(--labels-only)
fi

# --- Run the scraper(s) under Xvfb ---
echo "==== starting scraper(s) $(date -Is) args=${SCRAPER_ARGS[*]} ===="
cd "$WORK/scraper"
SCRAPER_EXIT=0
if [ "$NUM_WORKERS" -le 1 ]; then
  set +e
  xvfb-run -a /opt/venv/bin/python rescrape_images.py "${SCRAPER_ARGS[@]}" 2>&1 | tee -a "$LOG"
  SCRAPER_EXIT=$?
  set -e
else
  # Multi-worker: each worker gets its own $HOME so undetected-chromedriver
  # has its own patched chromedriver binary and user-data dir under it.
  # Without this, multiple parallel uc.Chrome() calls race on the shared
  # ~/.local/share/undetected_chromedriver/ paths and only one survives.
  range_size=$((RANGE_END - RANGE_START))
  chunk=$((range_size / NUM_WORKERS))
  pids=()
  for w in $(seq 0 $((NUM_WORKERS - 1))); do
    sub_start=$((RANGE_START + w * chunk))
    if [ "$w" -eq $((NUM_WORKERS - 1)) ]; then
      sub_end=$RANGE_END
    else
      sub_end=$((RANGE_START + (w + 1) * chunk))
    fi
    WORKER_HOME="/var/lib/bp-worker-${w}"
    mkdir -p "$WORKER_HOME/.local" "$WORKER_HOME/.cache"
    echo "spawning worker $w for [$sub_start,$sub_end) (HOME=$WORKER_HOME)"
    HOME="$WORKER_HOME" \
    XDG_CACHE_HOME="$WORKER_HOME/.cache" \
    XDG_DATA_HOME="$WORKER_HOME/.local/share" \
      xvfb-run -a /opt/venv/bin/python rescrape_images.py \
        --mode full \
        --url-csv ../data/all_urls.csv \
        --start "$sub_start" \
        --end "$sub_end" \
        --labels-csv "$WORK/data/labels-w${w}.csv" \
        --failures-csv "$WORK/data/scrape_failures-w${w}.csv" \
        > "$LOG.w${w}" 2>&1 &
    pids+=($!)
    # Small stagger so each worker's UC cache write completes before the next
    # one starts (UC takes ~10-15s to patch chromedriver on first run).
    sleep 12
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      SCRAPER_EXIT=1
    fi
  done
fi
echo "==== scraper exit=$SCRAPER_EXIT $(date -Is) ===="

# --- Stop uploader, do final flush ---
kill "$UPLOADER_PID" 2>/dev/null || true
if [ "$LABELS_ONLY" != "1" ]; then
  gcloud storage rsync "$WORK/data/img" "$GCS_IMG" --recursive 2>/dev/null || true
fi
for f in "$WORK"/data/labels*.csv; do
  [ -e "$f" ] || continue
  bn=$(basename "$f" .csv)
  gcloud storage cp "$f" "gs://$BUCKET/labels/${INSTANCE_IDX}${bn#labels}.csv" 2>/dev/null || true
done
for f in "$WORK"/data/scrape_failures*.csv; do
  [ -e "$f" ] || continue
  bn=$(basename "$f" .csv)
  gcloud storage cp "$f" "gs://$BUCKET/failures/${INSTANCE_IDX}${bn#scrape_failures}.csv" 2>/dev/null || true
done

# --- Mark done + self-delete only if scraper succeeded ---
if [ "$SCRAPER_EXIT" -eq 0 ]; then
  echo "$(date -Is) instance=$INSTANCE_NAME range=[$RANGE_START,$RANGE_END)" | \
    gcloud storage cp - "$GCS_DONE"
  echo "wrote done marker; self-deleting"
  gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
else
  echo "scraper failed; leaving instance up for inspection"
fi
