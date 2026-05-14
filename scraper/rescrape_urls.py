"""Scrape every ball python listing URL (active + sold) from morphmarket's
/all/?state=any index page, using a logged-in session.

Output: data/all_urls.csv (one column 'url' with relative paths). Pass this file
to rescrape_images.py via --mode full --url-csv data/all_urls.csv to download
images for the resulting URLs.

Usage:
    python scraper/rescrape_urls.py
    python scraper/rescrape_urls.py --start 1500 --output data/all_urls.csv
"""

import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional, Set

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescrape_images import (
    _detect_chrome_major,
    _is_cloudflare_challenge,
    _make_driver,
    perform_manual_login,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "all_urls.csv"
DEFAULT_FAILURES = DATA_DIR / "url_scrape_failures.csv"

INDEX_URL_TMPL = (
    "https://www.morphmarket.com/all/c/reptiles/pythons/ball-pythons"
    "?state=any&view=grid&page={page}"
)
ANIMAL_CARD_RE = re.compile(r"^animalCard--")
PAGE_LOAD_DELAY = 5.0
PAGE_LOAD_JITTER = 2.0
CLOUDFLARE_BACKOFF = 90
CLOUDFLARE_MAX_RETRIES = 8


def load_existing_urls(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        return set()
    return set(df["url"].dropna().astype(str).tolist())


def append_urls(csv_path: Path, new_urls):
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["url"])
        for url in new_urls:
            writer.writerow([url])
        f.flush()


def record_failure(failures_path: Path, page, reason, detail):
    is_new = not failures_path.exists()
    with failures_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["page", "reason", "detail"])
        writer.writerow([page, reason, (detail or "")[:200]])
        f.flush()


def fetch_page_urls(driver, page_num):
    page_url = INDEX_URL_TMPL.format(page=page_num)
    driver.get(page_url)
    time.sleep(PAGE_LOAD_DELAY + random.random() * PAGE_LOAD_JITTER)

    for retry in range(CLOUDFLARE_MAX_RETRIES + 1):
        page_source = driver.page_source
        if not _is_cloudflare_challenge(page_source):
            break
        if retry >= CLOUDFLARE_MAX_RETRIES:
            return None, "cloudflare"
        wait = CLOUDFLARE_BACKOFF * (2 ** retry)
        print(
            f"  cloudflare on page {page_num}, sleeping {wait}s "
            f"(retry {retry + 1}/{CLOUDFLARE_MAX_RETRIES})",
            flush=True,
        )
        time.sleep(wait)
        driver.get(page_url)
        time.sleep(PAGE_LOAD_DELAY + random.random() * PAGE_LOAD_JITTER)

    title = ""
    try:
        title = driver.title or ""
    except Exception:
        pass
    if "Log In" in title or "Sign In" in title:
        return None, "login_required"

    soup = BeautifulSoup(page_source, "html.parser")
    cards = soup.find_all("a", class_=ANIMAL_CARD_RE)
    hrefs = [a.get("href") for a in cards if a.get("href")]
    return hrefs, None


def main():
    parser = argparse.ArgumentParser(
        description="Scrape morphmarket listing URLs from /all/?state=any (login required)."
    )
    parser.add_argument("--start", type=int, default=1, help="First page (1-indexed). Default 1.")
    parser.add_argument(
        "--end",
        type=int,
        default=1000000,
        help="Maximum page to attempt; will stop earlier on consecutive empty pages. Default 10000.",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path."
    )
    parser.add_argument(
        "--failures",
        type=str,
        default=str(DEFAULT_FAILURES),
        help="CSV to log per-page failures.",
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="Skip the manual-login pause. /all/?state=any normally requires it; "
        "use this only if you've already authenticated in the same Chrome profile.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless (likely blocked by Cloudflare).",
    )
    parser.add_argument(
        "--chrome-version",
        type=int,
        default=None,
        help="Pin Chrome major version. Auto-detected on Windows if omitted.",
    )
    parser.add_argument(
        "--empty-pages-to-stop",
        type=int,
        default=2,
        help="Stop after this many consecutive empty pages. Default 2.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Append to CSV every N pages. Default 1 (every page).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    failures_path = Path(args.failures)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seen = load_existing_urls(output_path)
    print(f"loaded {len(seen)} existing URLs from {output_path}", flush=True)

    driver = _make_driver(headless=args.headless, chrome_version=args.chrome_version)
    if not args.no_login:
        perform_manual_login(driver)

    consecutive_empty = 0
    total_new_this_run = 0
    pending = []
    started_at = time.time()
    try:
        for page in range(args.start, args.end + 1):
            try:
                hrefs, err = fetch_page_urls(driver, page)
            except (TimeoutException, WebDriverException) as e:
                print(
                    f"page {page}: ERR {type(e).__name__}: {str(e)[:120]}",
                    flush=True,
                )
                record_failure(failures_path, page, type(e).__name__, str(e))
                continue

            if err == "cloudflare":
                print(f"page {page}: CLOUDFLARE retries exhausted, skipping", flush=True)
                record_failure(failures_path, page, "cloudflare", "exhausted retries")
                continue

            if err == "login_required":
                print(
                    f"page {page}: redirected to login. Session lost or filter requires auth. "
                    "Re-run with login.",
                    flush=True,
                )
                record_failure(failures_path, page, "login_required", "")
                break

            new_urls = [h for h in hrefs if h not in seen]

            if not hrefs:
                consecutive_empty += 1
                print(
                    f"page {page}: 0 cards (consecutive_empty={consecutive_empty})",
                    flush=True,
                )
                if consecutive_empty >= args.empty_pages_to_stop:
                    print(
                        f"stopping at page {page}: {consecutive_empty} consecutive empty pages",
                        flush=True,
                    )
                    break
                continue

            consecutive_empty = 0
            seen.update(new_urls)
            pending.extend(new_urls)
            total_new_this_run += len(new_urls)

            if len(pending) >= args.save_every or page == args.end:
                append_urls(output_path, pending)
                pending = []

            elapsed = time.time() - started_at
            pages_done = page - args.start + 1
            rate = pages_done / elapsed if elapsed > 0 else 0
            eta_min = (args.end - page) / rate / 60 if rate > 0 else 0
            print(
                f"page {page}: {len(hrefs)} cards, {len(new_urls)} new "
                f"(total_seen={len(seen)} +{total_new_this_run} this run) "
                f"rate={rate:.2f} pg/s eta_to_end={eta_min:.0f}min",
                flush=True,
            )
    finally:
        if pending:
            append_urls(output_path, pending)
        try:
            driver.quit()
        except Exception:
            pass

    print(f"done. {len(seen)} total URLs in {output_path}", flush=True)


if __name__ == "__main__":
    main()
