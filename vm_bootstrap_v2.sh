#!/bin/bash
# Bootstrap for v2 redeployment. Disk already formatted + populated from v1 round.
# Idempotent: safe to re-run.
set -e
LOG=/var/log/bootstrap.log
exec > >(tee -a "$LOG") 2>&1
echo "=== Bootstrap v2 starting at $(date -u) ==="

DEVICE=/dev/disk/by-id/google-bp-train-data
MNT=/mnt/data

# 1. Wait for the persistent disk to attach (up to ~2.5 min)
for i in {1..30}; do
  [ -e "$DEVICE" ] && break
  echo "Waiting for $DEVICE ($i/30)..."; sleep 5
done
[ -e "$DEVICE" ] || { echo "FATAL: $DEVICE never appeared"; exit 1; }

# 2. Sanity-check: disk MUST already have ext4 from round 1 (no mkfs)
if ! blkid "$DEVICE" >/dev/null 2>&1; then
  echo "FATAL: $DEVICE has no filesystem; expected ext4 from prior deployment"; exit 1
fi

# 3. Mount + persist via fstab (idempotent)
mkdir -p "$MNT"
mountpoint -q "$MNT" || mount "$DEVICE" "$MNT"
chmod 1777 "$MNT"
grep -q "google-bp-train-data" /etc/fstab || \
  echo "$DEVICE $MNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab

# 4. Mirror future output to a persistent log on the data disk
exec > >(tee -a "$MNT/bootstrap.log") 2>&1
echo "=== Verifying disk state ==="
echo "Image count: $(ls "$MNT/img" | wc -l)  (expect 271,355)"
echo "CSV lines:   $(wc -l < "$MNT/top37_gene_df.csv")  (expect 188,437)"

# 5. Pull updated training script (overwrites the stale v1-era copy)
gcloud storage cp gs://bp-scraper-images/code/training_selected_genes_efficientnet_vm.py \
  "$MNT/training_selected_genes_efficientnet_vm.py"
grep -E "^IMAGE_SIZE|^BATCH_SIZE|^SAVE_AS_VERSION|^USE_GRAD_CKPT|^USE_BF16|^USE_COMPILE" \
  "$MNT/training_selected_genes_efficientnet_vm.py"

# 6. Clean v1 smoke / stale artifacts
rm -f "$MNT/smoke.py" "$MNT/smoke.log" "$MNT/train.log"

# 7. Install all deps the v2 script needs (matplotlib + pyarrow added since v1 bootstrap)
PIP=""; for c in /opt/conda/bin/pip pip3 pip; do command -v "$c" >/dev/null && { PIP="$c"; break; }; done
[ -n "$PIP" ] || { echo "FATAL: no pip"; exit 1; }
echo "Using $PIP"
"$PIP" install --upgrade --quiet gcsfs pandas scikit-learn tqdm pyarrow matplotlib

# 8. Sanity checks (BF16 + compute capability matter for v2)
echo "=== Sanity checks ==="
nvidia-smi
PYTHON=""; for c in /opt/conda/bin/python python3 python; do command -v "$c" >/dev/null && { PYTHON="$c"; break; }; done
"$PYTHON" - <<'PY'
import torch, torchvision
print(f"torch {torch.__version__}, torchvision {torchvision.__version__}")
print(f"CUDA OK: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
import gcsfs, pandas, sklearn, PIL, tqdm, numpy, matplotlib, pyarrow
print("all deps OK")
PY

# 9. Sentinel
touch "$MNT/bootstrap_v2.done"
echo "=== Bootstrap v2 complete at $(date -u) ==="
