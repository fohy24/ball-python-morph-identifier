"""Build coverage map from the GCS image bucket.

Walks gs://bp-scraper-images/img/ recursively, extracts the `n` value from each
`{n}-{i}.png` filename, and produces two CSVs against data/all_urls.csv:

  data/phase1_covered.csv   — (n, url) for every n with at least one image
  data/phase2_remaining.csv — (n, url) for every n with no images yet

These files drive the Phase 1 (re-label) and Phase 2 (image+label) fleets.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ALL_URLS = ROOT / "data" / "all_urls.csv"
COVERED_OUT = ROOT / "data" / "phase1_covered.csv"
REMAINING_OUT = ROOT / "data" / "phase2_remaining.csv"

BUCKET_IMG_PREFIX = "gs://bp-scraper-images/img/"
GCLOUD = shutil.which("gcloud") or "gcloud"
N_FROM_PATH_RE = re.compile(r"/(\d+)-\d+\.png$")


def main() -> None:
    if not ALL_URLS.exists():
        sys.exit(f"missing {ALL_URLS}")

    print(f"listing {BUCKET_IMG_PREFIX} (this can take a minute)...", flush=True)
    out = subprocess.check_output(
        [GCLOUD, "storage", "ls", "-r", BUCKET_IMG_PREFIX],
        text=True,
    )

    covered_ns = set()
    for line in out.splitlines():
        m = N_FROM_PATH_RE.search(line.strip())
        if m:
            covered_ns.add(int(m.group(1)))
    print(f"  {len(covered_ns)} unique n values covered", flush=True)

    urls = pd.read_csv(ALL_URLS)["url"].astype(str).tolist()
    n_total = len(urls)
    print(f"  all_urls.csv has {n_total} URLs", flush=True)

    covered_rows = [(n, urls[n]) for n in sorted(covered_ns) if n < n_total]
    remaining_rows = [(n, urls[n]) for n in range(n_total) if n not in covered_ns]

    pd.DataFrame(covered_rows, columns=["n", "url"]).to_csv(COVERED_OUT, index=False)
    pd.DataFrame(remaining_rows, columns=["n", "url"]).to_csv(REMAINING_OUT, index=False)
    print(
        f"wrote {len(covered_rows)} -> {COVERED_OUT}\n"
        f"wrote {len(remaining_rows)} -> {REMAINING_OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
