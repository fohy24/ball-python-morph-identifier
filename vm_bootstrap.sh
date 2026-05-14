#!/bin/bash
# Startup script for the L4 training VM.
# Runs as root on first boot, idempotent on subsequent boots.
# Logs to /var/log/bootstrap.log and (once disk is mounted) /mnt/data/bootstrap.log.

set -e
LOG=/var/log/bootstrap.log
exec > >(tee -a "$LOG") 2>&1
echo "=== Bootstrap starting at $(date -u) ==="

DEVICE=/dev/disk/by-id/google-bp-train-data
MNT=/mnt/data

# 1. Wait for the persistent disk to attach (up to ~2.5 min)
for i in {1..30}; do
  if [ -e "$DEVICE" ]; then break; fi
  echo "Waiting for $DEVICE ($i/30)..."
  sleep 5
done
[ -e "$DEVICE" ] || { echo "FATAL: $DEVICE never appeared"; exit 1; }

# 2. Format ext4 if blank
if ! blkid "$DEVICE" >/dev/null 2>&1; then
  echo "Formatting $DEVICE as ext4"
  mkfs.ext4 -F "$DEVICE"
fi

# 3. Mount + persist via fstab
mkdir -p "$MNT"
if ! mountpoint -q "$MNT"; then
  mount "$DEVICE" "$MNT"
fi
chmod 1777 "$MNT"   # sticky-bit world-writable like /tmp; private VM, single-user
if ! grep -q "google-bp-train-data" /etc/fstab; then
  echo "$DEVICE $MNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab
fi

# Tee future output to the persistent log too
exec > >(tee -a "$MNT/bootstrap.log") 2>&1

# 4. Sync images + CSV + training script from GCS (idempotent rsync/cp)
echo "=== Syncing data from GCS ==="
gcloud storage rsync --recursive gs://bp-scraper-images/img "$MNT/img"
gcloud storage cp gs://bp-scraper-images/data/top37_gene_df.csv "$MNT/top37_gene_df.csv"
gcloud storage cp gs://bp-scraper-images/code/training_selected_genes_efficientnet_vm.py "$MNT/training_selected_genes_efficientnet_vm.py"

# 5. Install gcsfs into the DL VM's bundled Python (try conda first, fall back to system)
PIP=""
for cand in /opt/conda/bin/pip pip3 pip; do
  if command -v "$cand" >/dev/null 2>&1; then PIP="$cand"; break; fi
done
[ -n "$PIP" ] || { echo "FATAL: no pip found"; exit 1; }
echo "Using $PIP"
"$PIP" install --upgrade gcsfs

# 6. Sanity checks (warn but don't fail bootstrap on these)
echo "=== Sanity checks ==="
nvidia-smi || echo "WARN: nvidia-smi failed"

PYTHON=""
for cand in /opt/conda/bin/python python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
done
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'CUDA OK:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no device')" || echo "WARN: torch import failed"
"$PYTHON" -c "import gcsfs, pandas, sklearn, torchvision, PIL, tqdm, numpy; print('all deps OK')" || echo "WARN: dep import failed"

# 7. Sentinel — clients can poll for this file to know bootstrap is done
touch "$MNT/bootstrap.done"
echo "=== Bootstrap complete at $(date -u) ==="
