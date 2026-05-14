# Fleet — distributed morphmarket scrape on GCE Spot

Spins up ~20 GCE Spot e2-small instances across North American zones, each
scraping a disjoint slice of `data/all_urls.csv` and uploading images +
labels to `gs://bp-scraper-images/`. Designed for the case where any instance
can be preempted at any time without losing more than ~60 s of work.

## One-time setup (already done)

```bash
gcloud storage buckets create gs://bp-scraper-images --location=us \
  --default-storage-class=STANDARD --uniform-bucket-level-access \
  --project=beaming-opus-494700-q4

# Grant default Compute SA write to bucket and self-delete on the project
COMPUTE_SA="$(gcloud projects describe beaming-opus-494700-q4 \
  --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding gs://bp-scraper-images \
  --member=serviceAccount:$COMPUTE_SA --role=roles/storage.objectAdmin
gcloud projects add-iam-policy-binding beaming-opus-494700-q4 \
  --member=serviceAccount:$COMPUTE_SA --role=roles/compute.instanceAdmin.v1

gcloud storage cp scraper/rescrape_images.py gs://bp-scraper-images/code/
gcloud storage cp data/all_urls.csv          gs://bp-scraper-images/code/
```

## Run

```bash
# Launch instances for any range that isn't already done. Idempotent — safe to rerun.
python scraper/fleet/fleet.py up --instances 20

# Check progress (DONE / RUNNING / MISSING per range).
python scraper/fleet/fleet.py status
python scraper/fleet/fleet.py status --count-images   # extra GCS calls; slower

# Tear down everything if you need to abort.
python scraper/fleet/fleet.py down
```

If a range goes MISSING (Spot preempted twice in a row, etc.), `fleet.py up`
will recreate it. Each instance rsyncs prior images + labels from GCS on
boot, so resume is per-image — repeat work is bounded by the 60 s upload
interval.

## After all ranges DONE

```bash
python scraper/fleet/fleet.py merge
```

Prints commands to:
1. Flatten `gs://.../img/IDX/*` into one prefix
2. Concat the 20 per-instance `labels-IDX.csv` shards into a single
   `labels.csv` (matches the format `data_cleaning.ipynb` expects)
3. Optionally rsync everything down for local training

The merged `labels.csv` is the same shape as the original
`scraper-pic.py` output (`index, genes, sex, origin, price, birth, url`).
Feed it into `data_cleaning.ipynb` to drop hets, one-hot encode, and filter
to top 20 morphs → `selected_gene_df.csv`.

## Cost (at the rates measured: 11 s/listing)

| Item | Spot | On-demand |
|---|---|---|
| 20 × e2-small × 17.6 h | $1.76 | $5.88 |
| GCS multi-region storage 132 GB × 1 mo | $3.43 | $3.43 |
| Egress to your laptop (skip if training off GCS) | $15.84 | $15.84 |

Spot preemption: 30 s warning → SIGTERM → `shutdown.sh` rsyncs final
images + labels to GCS, instance dies, `fleet.py up` recreates the range
on next invocation.

## Files

| File | Purpose |
|---|---|
| `fleet.py` | CLI: up / status / down / merge |
| `startup.sh` | per-instance bootstrap (apt + pip, pull code, run scraper, periodic rsync, self-delete on success) |
| `shutdown.sh` | SIGTERM handler — final rsync within the 90-second preemption window |
| `../rescrape_images.py` | the actual scraper (image download + metadata extraction) |
