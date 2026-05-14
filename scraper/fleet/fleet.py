"""Fleet manager for the morphmarket ball-python image scrape on GCE Spot.

Subcommands:
  up       : Launch the fleet (idempotent; creates only ranges that aren't done).
  status   : Show progress for each range, plus instance liveness.
  down     : Delete all running fleet instances.
  merge    : Print a gsutil command to flatten gs://bucket/img/IDX/ into one prefix.
  phase1   : Re-label covered URLs (labels-only mode, reads phase1_covered.csv).
  phase2   : Scrape remaining URLs (images + labels, reads phase2_remaining.csv).

Reads no shared state besides what's in GCS / GCE; safe to interrupt and re-run.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GCLOUD = shutil.which("gcloud") or "gcloud"

ROOT = Path(__file__).resolve().parent.parent.parent
URL_CSV = ROOT / "data" / "all_urls.csv"
STARTUP_SH = Path(__file__).resolve().parent / "startup.sh"
SHUTDOWN_SH = Path(__file__).resolve().parent / "shutdown.sh"

PROJECT = "beaming-opus-494700-q4"
BUCKET = "bp-scraper-images"
# e2-medium on-demand: 4 GB RAM (vs e2-small's 2 GB) avoids InvalidSessionIdException
# Chrome crashes; on-demand avoids Spot preemption mid-scrape.
MACHINE = "e2-medium"
DISK_SIZE_GB = 15
DISK_TYPE = "pd-balanced"
IMAGE_PROJECT = "debian-cloud"
IMAGE_FAMILY = "debian-12"
INSTANCE_PREFIX = "bp-scraper"
SERVICE_ACCOUNT_SCOPES = "https://www.googleapis.com/auth/cloud-platform"

ZONES = [
    "us-central1-a",
    "us-central1-b",
    "us-east1-b",
    "us-east1-c",
    "us-east1-d",
    "us-east4-a",
    "us-east4-b",
    "us-east4-c",
    "us-east5-a",
    "us-east5-b",
    "us-west1-a",
    "us-west1-b",
    "us-west2-a",
    "us-west2-b",
    "us-west3-a",
    "us-west4-a",
    "us-west4-b",
    "us-south1-a",
    "northamerica-northeast1-a",
    "northamerica-northeast1-b",
]
# Tried in order if a range's primary zone returns ZONE_RESOURCE_POOL_EXHAUSTED.
FALLBACK_ZONES = [
    "us-west2-a",
    "us-west2-b",
    "us-west3-a",
    "us-west4-a",
    "us-west4-b",
    "us-south1-a",
    "northamerica-northeast1-a",
    "northamerica-northeast1-b",
    "us-central1-c",
    "us-central1-f",
]


@dataclass
class Range:
    idx: int
    start: int
    end: int
    zone: str

    @property
    def name(self) -> str:
        return f"{INSTANCE_PREFIX}-{self.idx:02d}"


def partition(total: int, n: int, zones: list[str]) -> list[Range]:
    """Split `total` URLs across `n` instances. Zones cycle if n > len(zones).

    Per-region quota is the practical ceiling. Default per-region GCE quota
    is 24 vCPUs and we have 20 zones spanning ~10 regions, so cycling fits
    up to ~100 e2-medium instances without bumping per-region limits.
    """
    chunk = total // n
    extra = total % n
    out, cursor = [], 0
    for i in range(n):
        size = chunk + (1 if i < extra else 0)
        zone = zones[i % len(zones)]
        out.append(Range(idx=i, start=cursor, end=cursor + size, zone=zone))
        cursor += size
    return out


def gcloud(*args: str, check: bool = True, capture: bool = True) -> str:
    cmd = [GCLOUD, *args, "--quiet"]
    p = subprocess.run(cmd, capture_output=capture, text=True, shell=False)
    if check and p.returncode != 0:
        sys.stderr.write(f"FAIL: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}\n")
        raise SystemExit(p.returncode)
    return p.stdout


def list_done_indices() -> set[int]:
    out = gcloud(
        "storage", "ls", f"gs://{BUCKET}/done/",
        check=False,
    )
    idx = set()
    for line in out.splitlines():
        m = re.search(r"/(\d+)\.marker$", line.strip())
        if m:
            idx.add(int(m.group(1)))
    return idx


def list_running_instances() -> dict[str, str]:
    out = gcloud(
        "compute", "instances", "list",
        f"--project={PROJECT}",
        f"--filter=name~^{INSTANCE_PREFIX}-",
        "--format=json",
    )
    rows = json.loads(out or "[]")
    result = {}
    for r in rows:
        zone = r["zone"].split("/")[-1]
        result[r["name"]] = zone
    return result


def count_images_in(idx: int) -> int:
    out = gcloud(
        "storage", "ls", f"gs://{BUCKET}/img/{idx}/",
        check=False,
    )
    return sum(1 for line in out.splitlines() if line.strip().endswith(".png"))


def total_url_count() -> int:
    if not URL_CSV.exists():
        sys.exit(f"missing {URL_CSV} — upload it before running fleet up")
    n = 0
    with URL_CSV.open(encoding="utf-8") as f:
        next(f)  # header
        for _ in f:
            n += 1
    return n


def _try_create(name: str, zone: str, r: Range) -> bool:
    """Attempt to create one instance in `zone`. Returns True on success."""
    cmd = [
        GCLOUD, "compute", "instances", "create", name,
        f"--project={PROJECT}",
        f"--zone={zone}",
        f"--machine-type={MACHINE}",
        f"--image-project={IMAGE_PROJECT}",
        f"--image-family={IMAGE_FAMILY}",
        f"--boot-disk-size={DISK_SIZE_GB}GB",
        f"--boot-disk-type={DISK_TYPE}",
        "--maintenance-policy=MIGRATE",
        f"--scopes={SERVICE_ACCOUNT_SCOPES}",
        f"--metadata=range_start={r.start},range_end={r.end},instance_idx={r.idx},bucket={BUCKET}",
        f"--metadata-from-file=startup-script={STARTUP_SH.as_posix()},shutdown-script={SHUTDOWN_SH.as_posix()}",
        "--quiet",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 0:
        return True
    err = (p.stderr or "") + (p.stdout or "")
    if "ZONE_RESOURCE_POOL_EXHAUSTED" in err or "does not have enough resources" in err:
        print(f"    {zone}: ZONE_RESOURCE_POOL_EXHAUSTED")
    else:
        print(f"    {zone}: failed — {err.strip()[:200]}")
    return False


def create_one(r: Range) -> None:
    name = r.name
    # Try the primary zone first, then fall back through alternates if the
    # primary returns ZONE_RESOURCE_POOL_EXHAUSTED (common for e2-medium).
    candidates = [r.zone] + [z for z in FALLBACK_ZONES if z != r.zone]
    print(f"  creating {name} for [{r.start},{r.end})")
    for zone in candidates:
        if _try_create(name, zone, r):
            print(f"    -> created in {zone}")
            return
    raise SystemExit(f"all zones exhausted for {name}")


def cmd_up(args: argparse.Namespace) -> None:
    total = total_url_count()
    print(f"all_urls.csv has {total} URLs; partitioning into {args.instances}")
    ranges = partition(total, args.instances, ZONES[: args.instances])

    done = list_done_indices()
    running = list_running_instances()
    print(f"already done: {sorted(done)} ({len(done)})")
    print(f"already running: {sorted(running)} ({len(running)})")

    to_create = []
    for r in ranges:
        if r.idx in done:
            continue
        if r.name in running:
            continue
        to_create.append(r)

    if not to_create:
        print("nothing to create — every range is done or already running")
        return

    print(f"creating {len(to_create)} instances")
    for r in to_create:
        try:
            create_one(r)
        except SystemExit:
            print(f"  failed to create {r.name}; skipping")


def cmd_status(args: argparse.Namespace) -> None:
    total = total_url_count()
    ranges = partition(total, args.instances, ZONES[: args.instances])
    done = list_done_indices()
    running = list_running_instances()

    print(f"{'idx':>3}  {'range':<18}  {'zone':<28}  {'status':<10}  imgs")
    print("-" * 80)
    n_done = n_running = n_missing = 0
    for r in ranges:
        if r.idx in done:
            status = "DONE"
            n_done += 1
        elif r.name in running:
            status = "RUNNING"
            n_running += 1
        else:
            status = "MISSING"
            n_missing += 1
        imgs = count_images_in(r.idx) if args.count_images else "-"
        print(
            f"{r.idx:>3}  [{r.start:>6},{r.end:>6})  {r.zone:<28}  "
            f"{status:<10}  {imgs}"
        )
    print("-" * 80)
    print(f"done={n_done} running={n_running} missing={n_missing} of {len(ranges)}")


def cmd_down(args: argparse.Namespace) -> None:
    running = list_running_instances()
    if not running:
        print("no fleet instances running")
        return
    print(f"deleting {len(running)} instances:")
    for name, zone in running.items():
        print(f"  {name} ({zone})")
        gcloud(
            "compute", "instances", "delete", name,
            f"--project={PROJECT}",
            f"--zone={zone}",
            check=False,
        )


def _try_create_phase(
    name: str,
    zone: str,
    instance_idx: str,
    chunk_url: str,
    labels_only: bool,
) -> bool:
    """Create one phase-1/2 instance. instance_idx is a string like 'p1-00'."""
    metadata_pieces = [
        f"instance_idx={instance_idx}",
        f"bucket={BUCKET}",
        f"chunk_url={chunk_url}",
    ]
    if labels_only:
        metadata_pieces.append("labels_only=1")
    metadata = ",".join(metadata_pieces)

    cmd = [
        GCLOUD, "compute", "instances", "create", name,
        f"--project={PROJECT}",
        f"--zone={zone}",
        f"--machine-type={MACHINE}",
        f"--image-project={IMAGE_PROJECT}",
        f"--image-family={IMAGE_FAMILY}",
        f"--boot-disk-size={DISK_SIZE_GB}GB",
        f"--boot-disk-type={DISK_TYPE}",
        "--maintenance-policy=MIGRATE",
        f"--scopes={SERVICE_ACCOUNT_SCOPES}",
        f"--metadata={metadata}",
        f"--metadata-from-file=startup-script={STARTUP_SH.as_posix()},shutdown-script={SHUTDOWN_SH.as_posix()}",
        "--quiet",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 0:
        return True
    err = (p.stderr or "") + (p.stdout or "")
    if "ZONE_RESOURCE_POOL_EXHAUSTED" in err or "does not have enough resources" in err:
        print(f"    {zone}: ZONE_RESOURCE_POOL_EXHAUSTED")
    else:
        print(f"    {zone}: failed — {err.strip()[:200]}")
    return False


def _list_done_with_marker_prefix(prefix: str) -> set[str]:
    """List existing done markers matching `<prefix>NN.marker`."""
    out = gcloud("storage", "ls", f"gs://{BUCKET}/done/", check=False)
    found = set()
    pat = re.compile(rf"/({re.escape(prefix)}\d+)\.marker$")
    for line in out.splitlines():
        m = pat.search(line.strip())
        if m:
            found.add(m.group(1))
    return found


def _list_running_with_name_prefix(name_prefix: str) -> set[str]:
    out = gcloud(
        "compute", "instances", "list",
        f"--project={PROJECT}",
        f"--filter=name~^{name_prefix}",
        "--format=json",
    )
    rows = json.loads(out or "[]")
    return {r["name"] for r in rows}


def _phase_run(
    *,
    phase: str,
    csv_path: Path,
    labels_only: bool,
    n_instances: int,
    chunk_gcs_dir: str,
) -> None:
    """Shared driver for cmd_phase1 and cmd_phase2."""
    if not csv_path.exists():
        sys.exit(f"missing {csv_path} — run scraper/fleet/build_coverage.py first")

    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"{csv_path.name} has {n_total} listings; splitting into {n_instances} chunks")

    # Even split with remainder distributed to early chunks.
    chunks_dir = ROOT / "data" / f"{phase}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = n_total // n_instances
    extra = n_total % n_instances
    chunk_paths: list[tuple[int, Path]] = []
    cursor = 0
    for i in range(n_instances):
        sz = chunk_size + (1 if i < extra else 0)
        chunk = df.iloc[cursor:cursor + sz]
        path = chunks_dir / f"chunk_{i:02d}.csv"
        chunk.to_csv(path, index=False)
        chunk_paths.append((i, path))
        cursor += sz

    # Upload chunks to GCS (idempotent — overwrite is fine).
    print(f"uploading {len(chunk_paths)} chunks to {chunk_gcs_dir}")
    for i, path in chunk_paths:
        gcloud(
            "storage", "cp",
            str(path),
            f"{chunk_gcs_dir}/chunk_{i:02d}.csv",
            check=False,
        )

    marker_prefix = f"{phase}-"        # e.g. 'p1-' -> done markers p1-00.marker
    name_prefix = f"{INSTANCE_PREFIX}-{phase}-"   # e.g. 'bp-scraper-p1-'

    done = _list_done_with_marker_prefix(marker_prefix)
    running = _list_running_with_name_prefix(name_prefix)
    print(f"already done markers: {len(done)}")
    print(f"already running instances: {len(running)}")

    created = 0
    skipped_done = 0
    skipped_running = 0
    failed = 0
    for i, path in chunk_paths:
        idx_str = f"{phase}-{i:02d}"
        name = f"{INSTANCE_PREFIX}-{idx_str}"
        if idx_str in done:
            skipped_done += 1
            continue
        if name in running:
            skipped_running += 1
            continue
        n_rows = max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)
        chunk_url = f"{chunk_gcs_dir}/chunk_{i:02d}.csv"
        primary_zone = ZONES[i % len(ZONES)]
        candidates = [primary_zone] + [z for z in FALLBACK_ZONES if z != primary_zone]
        print(f"  creating {name} ({n_rows} listings)")
        for z in candidates:
            if _try_create_phase(name, z, idx_str, chunk_url, labels_only=labels_only):
                print(f"    -> created in {z}")
                created += 1
                break
        else:
            print(f"    failed: all zones exhausted for {name}")
            failed += 1
    print(
        f"summary: created={created} skipped_done={skipped_done} "
        f"skipped_running={skipped_running} failed={failed}"
    )


def cmd_phase1(args: argparse.Namespace) -> None:
    """Phase 1: re-label every URL we already have images for."""
    _phase_run(
        phase="p1",
        csv_path=ROOT / "data" / "phase1_covered.csv",
        labels_only=True,
        n_instances=args.instances,
        chunk_gcs_dir=f"gs://{BUCKET}/code/phase1",
    )


def cmd_phase2(args: argparse.Namespace) -> None:
    """Phase 2: scrape remaining URLs (images + labels)."""
    _phase_run(
        phase="p2",
        csv_path=ROOT / "data" / "phase2_remaining.csv",
        labels_only=False,
        n_instances=args.instances,
        chunk_gcs_dir=f"gs://{BUCKET}/code/phase2",
    )


def cmd_phase_status(args: argparse.Namespace) -> None:
    """Show progress of a phase fleet (idx, status, labels-row count)."""
    phase = args.phase
    chunk_dir = ROOT / "data" / f"{phase}_chunks"
    if not chunk_dir.exists():
        sys.exit(f"no chunk dir {chunk_dir} — run phase{phase[-1]} first")
    chunks = sorted(chunk_dir.glob("chunk_*.csv"))
    done = _list_done_with_marker_prefix(f"{phase}-")
    running = _list_running_with_name_prefix(f"{INSTANCE_PREFIX}-{phase}-")

    n_done = n_running = n_missing = 0
    print(f"{'idx':>5}  {'rows':>6}  status")
    print("-" * 40)
    for path in chunks:
        m = re.match(r"chunk_(\d+)\.csv", path.name)
        if not m:
            continue
        i = int(m.group(1))
        idx_str = f"{phase}-{i:02d}"
        name = f"{INSTANCE_PREFIX}-{idx_str}"
        n_rows = max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)
        if idx_str in done:
            status = "DONE"
            n_done += 1
        elif name in running:
            status = "RUNNING"
            n_running += 1
        else:
            status = "MISSING"
            n_missing += 1
        print(f"{idx_str:>5}  {n_rows:>6}  {status}")
    print("-" * 40)
    print(
        f"done={n_done} running={n_running} missing={n_missing} of {len(chunks)}"
    )


def cmd_merge(args: argparse.Namespace) -> None:
    print(
        f"# 1. Flatten per-instance image prefixes into one directory:\n"
        f"gcloud storage cp 'gs://{BUCKET}/img/*/*' gs://{BUCKET}/img-merged/\n\n"
        f"# 2. Concatenate per-instance labels.csv files into one:\n"
        f"#    Each labels-NN.csv has its own header; only keep the first.\n"
        f"gcloud storage cp 'gs://{BUCKET}/labels/*.csv' /tmp/labels-shards/ --recursive\n"
        f"python -c \"\\\n"
        f"import pandas as pd, glob;\\\n"
        f"frames = [pd.read_csv(p) for p in sorted(glob.glob('/tmp/labels-shards/*.csv'))];\\\n"
        f"out = pd.concat(frames, ignore_index=True);\\\n"
        f"print(f'merged {{len(out)}} rows from {{len(frames)}} shards');\\\n"
        f"out.to_csv('/tmp/labels.csv', index=False)\"\n"
        f"gcloud storage cp /tmp/labels.csv gs://{BUCKET}/labels.csv\n\n"
        f"# 3. Or rsync everything to local for training off-cloud:\n"
        f"gcloud storage rsync gs://{BUCKET}/img/ ./data/img-by-instance/ --recursive\n"
        f"gcloud storage rsync gs://{BUCKET}/labels/ ./data/labels-by-instance/ --recursive\n\n"
        f"# 4. Hand the merged labels.csv to data_cleaning.ipynb to produce\n"
        f"#    labels_extended.csv and selected_gene_df.csv (existing pipeline).\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GCE Spot fleet for morphmarket image scrape.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("up", help="Create instances for any ranges that aren't done.")
    p_up.add_argument("--instances", type=int, default=100)
    p_up.set_defaults(func=cmd_up)

    p_status = sub.add_parser("status", help="Show per-range status.")
    p_status.add_argument("--instances", type=int, default=100)
    p_status.add_argument(
        "--count-images",
        action="store_true",
        help="Count images per range in GCS (extra API calls).",
    )
    p_status.set_defaults(func=cmd_status)

    p_down = sub.add_parser("down", help="Delete all fleet instances.")
    p_down.set_defaults(func=cmd_down)

    p_merge = sub.add_parser("merge", help="Print commands to flatten/download output.")
    p_merge.set_defaults(func=cmd_merge)

    p_p1 = sub.add_parser("phase1", help="Phase 1: relabel-only fleet over phase1_covered.csv")
    p_p1.add_argument("--instances", type=int, default=40)
    p_p1.set_defaults(func=cmd_phase1)

    p_p2 = sub.add_parser("phase2", help="Phase 2: full scrape over phase2_remaining.csv")
    p_p2.add_argument("--instances", type=int, default=40)
    p_p2.set_defaults(func=cmd_phase2)

    p_pstat = sub.add_parser("phase_status", help="Show per-chunk status of a phase fleet.")
    p_pstat.add_argument("phase", choices=["p1", "p2"])
    p_pstat.set_defaults(func=cmd_phase_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
