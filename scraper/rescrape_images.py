"""Re-scrape images for ball python listings, preserving the {n}-{i}.png filename
contract that links data/img/ files to the surviving CSVs in data/.

Usage:
    python scraper/rescrape_images.py --smoke-test
    python scraper/rescrape_images.py --mode targeted
    python scraper/rescrape_images.py --mode full
    python scraper/rescrape_images.py --mode targeted --start 0 --end 1000
"""

import argparse
import csv
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "img"
SELECTED_CSV = DATA_DIR / "selected_gene_df.csv"
COMPLETE_URL_CSV = DATA_DIR / "complete_url.csv"
FAILURES_CSV = DATA_DIR / "scrape_failures.csv"
LABELS_CSV = DATA_DIR / "labels.csv"

MORPHMARKET_BASE = "https://www.morphmarket.com"
LOGIN_URL = "https://www.morphmarket.com/account/sign-in/"
PAGE_LOAD_DELAY = 1.0  # tiny buffer after carousel-wait succeeds (lets sibling DOM settle)
PAGE_LOAD_JITTER = 0.5
PAGE_LOAD_TIMEOUT = 60  # selenium driver.get() hard cap; without this Chrome can hang forever
# 'eager' returns when DOMContentLoaded fires — we only need the carousel <img>
# elements, not every ad/analytics asset. Cuts driver.get() time dramatically
# on e2-small (was 4k+ PageLoadTimeout failures with the default 'normal').
PAGE_LOAD_STRATEGY = "eager"
CAROUSEL_WAIT_TIMEOUT = 15  # how long to wait for the carousel <img> to appear
# Selectors used by WebDriverWait. The carousel is the happy-path; if it never
# appears within CAROUSEL_WAIT_TIMEOUT we conclude the listing has no images
# (sold w/o photos, or removed) and bail.
CAROUSEL_SELECTOR = 'img[class^="thumbCarouselImage--"]'
LISTING_TITLE_SELECTOR = 'h1[class^="animalTitle--"]'
IMAGE_REQUEST_TIMEOUT = 20
CLOUDFLARE_BACKOFF_BASE = 90
CLOUDFLARE_MAX_RETRIES = 3
DRIVER_RESTART_AFTER_CF = 3

THUMB_CLASS_RE = re.compile(r"^thumbCarouselImage--")
# Trait chips: <a href="/[region]/c/reptiles/pythons/ball-pythons/trait/<slug>">
# wrapping a <span class="MuiChip-label chipLabel--<hash> ...">Display Name</span>
TRAIT_LINK_HREF_RE = re.compile(r"/c/reptiles/pythons/ball-pythons/trait/")
CHIP_LABEL_CLASS_RE = re.compile(r"chipLabel--")
ORIGIN_PATTERNS = [
    re.compile(r"^warning--"),
    re.compile(r"^success--"),
    re.compile(r"^danger--"),
]
PRICE_RE = re.compile(r"^salePrice--")
LABEL_VAL_CONTAINER_RE = re.compile(r"^labelValueContainer--")


def get_pic_url(soup):
    """Extract image URLs from a morphmarket listing page.

    Uses a regex-based class match because morphmarket rotates the hash suffix
    on its CSS-in-JS class names (e.g. thumbCarouselImage--RFBqw was the 2024
    hash; current hash is thumbCarouselImage--QhAGV).
    """
    pic_container = soup.find_all("img", class_=THUMB_CLASS_RE)
    pic_urls = [
        img["srcset"].split(", ")[0].split(" ")[0]
        for img in pic_container
        if "srcset" in img.attrs
    ]
    if not pic_urls:
        sources = soup.find_all("source")
        if sources and "srcset" in sources[0].attrs:
            pic_urls = [sources[0]["srcset"].split(" ")[0]]
    return pic_urls


def get_gene(soup):
    """Extract morph names from MUI Chip links on a morphmarket listing page.

    Each trait is rendered as:
        <a href="/all/c/reptiles/pythons/ball-pythons/trait/<slug>">
          <span class="MuiChip-root ...">
            <span class="MuiChip-label chipLabel--<hash> ...">Display Name</span>
          </span>
        </a>

    We collect every distinct chip-label text in document order. The href slug
    is URL-encoded and not always 1:1 with the displayed text (e.g. slug
    `pos%20het%20hypo` displays as `66% Het Hypo`), so we trust the visible
    text — that's what `data_cleaning.ipynb` will match against the morph
    vocabulary.

    Returns a Python list (e.g. ['Black Pastel', 'Champagne', 'Het Desert
    Ghost', '66% Het Hypo']). Empty list when no chips are found (typically
    sold/inactive listings).
    """
    out, seen = [], set()
    for a in soup.find_all("a", href=TRAIT_LINK_HREF_RE):
        chip = a.find("span", class_=CHIP_LABEL_CLASS_RE)
        if not chip:
            continue
        text = chip.get_text(strip=True)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def get_origin(soup):
    for pattern in ORIGIN_PATTERNS:
        for span in soup.find_all("span", class_=pattern):
            text = span.get_text(strip=True)
            if text:
                return text
    return "unknown"


def get_sex(soup):
    """Resolve the listing's sex.

    Morphmarket switched FontAwesome 5 (svg) -> 6 (i) somewhere along the way,
    and the page has multiple sex icons (related listings in sidebar). The
    authoritative one is inside the labelValueContainer that has <b>Sex:</b>.
    """
    for div in soup.find_all("div", class_=LABEL_VAL_CONTAINER_RE):
        b = div.find("b", string=lambda s: s and s.strip().rstrip(":") == "Sex")
        if not b:
            continue
        html = str(div)
        if "fa-venus-mars" in html:
            return "mixed"
        if "fa-mars" in html:
            return "male"
        if "fa-venus" in html:
            return "female"
        # No icon — try a text value (rare).
        span = div.find("span")
        if span:
            t = span.get_text(strip=True).lower()
            if t in ("male", "female", "mixed", "unknown"):
                return t
        break
    # Last-resort fallback: any explicit sex icon anywhere.
    if soup.find(["i", "svg"], class_="fa-venus-mars"):
        return "mixed"
    if soup.find(["i", "svg"], class_="fa-mars"):
        return "male"
    if soup.find(["i", "svg"], class_="fa-venus"):
        return "female"
    return "unknown"


def get_price(soup):
    """Extract listing price as a float (USD). Returns 0.0 if not found.

    Tries h1.salePrice--* first, then any class containing "price" (case
    insensitive) with a currency-like body, then a regex over the full page
    text. The 2024 selector returned strings like "$3,500.00" but morphmarket
    now serves "US$\xa03,500.00" (with a non-breaking space) — strip both.
    """
    candidates = []
    h = soup.find("h1", class_=PRICE_RE)
    if h:
        candidates.append(h.get_text(strip=True))
    for el in soup.find_all(class_=re.compile(r"[Pp]rice--")):
        candidates.append(el.get_text(strip=True))
    for text in candidates:
        if not text:
            continue
        m = re.search(r"[\d,]+(?:\.\d+)?", text.replace("\xa0", " "))
        if m:
            try:
                return float(m.group().replace(",", ""))
            except ValueError:
                continue
    return 0.0


def get_birth(soup):
    for div in soup.find_all("div", class_=LABEL_VAL_CONTAINER_RE):
        if div.find("b", string=lambda s: s and s.strip() == "Birth:"):
            span = div.find("span")
            if span:
                return span.get_text(strip=True)
    return ""


def build_targeted_jobs():
    df = pd.read_csv(SELECTED_CSV)
    parts = df["index"].astype(str).str.split("-", n=1, expand=True)
    df["n"] = parts[0].astype(int)
    df["i"] = parts[1].astype(int)
    grouped = (
        df.groupby("n")
        .agg(url=("url", "first"), needed=("i", lambda s: set(s.tolist())))
        .sort_index()
        .reset_index()
    )
    return [(int(row["n"]), str(row["url"]), set(row["needed"])) for _, row in grouped.iterrows()]


def build_full_jobs(csv_path: Optional[Path] = None):
    path = Path(csv_path) if csv_path else COMPLETE_URL_CSV
    df = pd.read_csv(path)
    url_col = "url" if "url" in df.columns else df.columns[0]
    jobs = []
    for n, rel in enumerate(df[url_col].tolist()):
        url = MORPHMARKET_BASE + rel if str(rel).startswith("/") else str(rel)
        jobs.append((n, url, None))
    return jobs


def build_listings_jobs(csv_path):
    """Build a job list from an explicit (n, url) CSV.

    Used by Phase 1 (re-label covered URLs) and Phase 2 (scrape remaining
    URLs) where each instance gets a pre-computed chunk that maps n -> url.
    Unlike build_full_jobs (which uses row position as n), here n is read
    from the CSV column so filenames stay aligned with all_urls.csv.
    """
    df = pd.read_csv(csv_path)
    if "n" not in df.columns or "url" not in df.columns:
        raise ValueError(f"{csv_path}: expected columns 'n' and 'url'")
    jobs = []
    for n, rel in zip(df["n"].astype(int).tolist(), df["url"].astype(str).tolist()):
        url = MORPHMARKET_BASE + rel if rel.startswith("/") else rel
        jobs.append((int(n), url, None))
    return jobs


def _detect_chrome_major() -> Optional[int]:
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["reg", "query", r"HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon", "/v", "version"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            m = re.search(r"version\s+REG_SZ\s+(\d+)\.", out)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None
    # Linux / macOS: ask the chrome binary directly.
    for binary in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        try:
            out = subprocess.check_output([binary, "--version"], text=True, stderr=subprocess.DEVNULL)
            m = re.search(r"(\d+)\.", out)
            if m:
                return int(m.group(1))
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def _make_driver(headless: bool = False, chrome_version: Optional[int] = None):
    """Build an undetected-chromedriver instance.

    Headless is OFF by default because morphmarket's Cloudflare protection
    blocks headless undetected-chromedriver in testing (10/10 challenges seen)
    while non-headless gets 0/10 challenges.
    """
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.page_load_strategy = PAGE_LOAD_STRATEGY
    if chrome_version is None:
        chrome_version = _detect_chrome_major()
    driver = uc.Chrome(options=options, version_main=chrome_version)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def perform_manual_login(driver, non_interactive_wait: int = 120):
    print("Opening morphmarket login page...", flush=True)
    driver.get(LOGIN_URL)
    print(
        "\n>>> Please log in to morphmarket in the visible Chrome window.\n"
        ">>> When you see the logged-in homepage (your avatar in the top-right corner),\n"
        ">>> press Enter here to continue.",
        flush=True,
    )
    if sys.stdin.isatty():
        try:
            input(">>> Press Enter when logged in: ")
        except EOFError:
            print(f"  (stdin closed — waiting {non_interactive_wait}s)", flush=True)
            time.sleep(non_interactive_wait)
    else:
        print(
            f"  (non-interactive shell detected — sleeping {non_interactive_wait}s "
            "to give you time to log in)",
            flush=True,
        )
        time.sleep(non_interactive_wait)
    print("Continuing with logged-in session.", flush=True)


_LAST_ERROR_CLOUDFLARE = False


def _is_cloudflare_challenge(page_source: str) -> bool:
    return "Just a moment" in page_source or "cf-challenge" in page_source.lower()


def _last_error_was_cloudflare() -> bool:
    return _LAST_ERROR_CLOUDFLARE


def _record_failure(n, url, expected, got, error_type, error_msg):
    new_file = not FAILURES_CSV.exists()
    with FAILURES_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["timestamp", "n", "url", "expected_count", "got_count", "error_type", "error_msg"]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                n,
                url,
                "" if expected is None else expected,
                "" if got is None else got,
                error_type,
                (error_msg or "")[:200],
            ]
        )
        f.flush()


def _record_label(n, i, genes, sex, origin, price, birth, url):
    """Append one row to the labels CSV.

    Columns mirror the original scraper-pic.py output (`index, genes, sex,
    origin, price, birth, url`). `genes` is stored as `repr(list)` —
    e.g. "['Black Pastel', 'Champagne', 'Het Desert Ghost', '66% Het Hypo']"
    — so `data_cleaning.ipynb` can `ast.literal_eval` it back to a list.
    """
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LABELS_CSV.exists()
    if not isinstance(genes, list):
        # Backward-compat: if a caller still passes a string, wrap it.
        genes = [genes] if genes else []
    with LABELS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["index", "genes", "sex", "origin", "price", "birth", "url"])
        writer.writerow([f"{n}-{i}", repr(genes), sex, origin, price, birth, url])
        f.flush()


def _all_present(n: int, needed: Optional[set]) -> bool:
    if needed is None:
        return False
    for i in needed:
        if not (IMG_DIR / f"{n}-{i}.png").exists():
            return False
    return True


def _download_listing(driver, n, url, needed, labels_only=False):
    global _LAST_ERROR_CLOUDFLARE
    _LAST_ERROR_CLOUDFLARE = False
    if not labels_only and _all_present(n, needed):
        return "skip", 0, 0

    page = None
    for retry in range(CLOUDFLARE_MAX_RETRIES + 1):
        try:
            driver.get(url)
        except TimeoutException:
            # Page took longer than PAGE_LOAD_TIMEOUT. Skip — better than hanging
            # the whole instance on one slow listing.
            _record_failure(
                n, url, len(needed) if needed else None, 0,
                "PageLoadTimeout", f"driver.get exceeded {PAGE_LOAD_TIMEOUT}s",
            )
            return "error", 0, 0
        # Wait for the carousel <img> to actually render. With pageLoadStrategy
        # = 'eager', driver.get returns at DOMContentLoaded, but the carousel
        # is hydrated by client JS slightly later — especially on slow CPUs.
        # If it doesn't appear in CAROUSEL_WAIT_TIMEOUT seconds, we'll fall
        # through to the soup parse and either pick up a <source> fallback or
        # log no_image_element.
        try:
            WebDriverWait(driver, CAROUSEL_WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, CAROUSEL_SELECTOR))
            )
        except TimeoutException:
            pass  # let downstream selectors decide
        time.sleep(PAGE_LOAD_DELAY + random.random() * PAGE_LOAD_JITTER)
        page = driver.page_source
        if not _is_cloudflare_challenge(page):
            break
        if retry >= CLOUDFLARE_MAX_RETRIES:
            _LAST_ERROR_CLOUDFLARE = True
            _record_failure(
                n, url, len(needed) if needed else None, 0, "CloudflareChallenge", "exhausted retries"
            )
            return "error", 0, 0
        wait = CLOUDFLARE_BACKOFF_BASE * (2 ** retry)
        print(
            f"  cloudflare detected, sleeping {wait}s (retry {retry + 1}/{CLOUDFLARE_MAX_RETRIES})",
            flush=True,
        )
        time.sleep(wait)

    soup = BeautifulSoup(page, "html.parser")
    pic_urls = get_pic_url(soup)

    # Extract per-listing metadata once (cheap, soup is already parsed).
    genes = get_gene(soup)
    sex = get_sex(soup)
    origin = get_origin(soup)
    price = get_price(soup)
    birth = get_birth(soup)

    if not pic_urls:
        _record_failure(
            n, url, len(needed) if needed else None, 0, "no_image_element",
            "get_pic_url returned empty",
        )
        return "error", 0, 0

    if labels_only:
        # Phase 1: just write a label row per image position; no fetches.
        for i in range(len(pic_urls)):
            _record_label(n, i, genes, sex, origin, price, birth, url)
        return "ok", len(pic_urls), len(pic_urls)

    saved = 0
    failed_images = 0
    for i, pic in enumerate(pic_urls):
        if needed is not None and i not in needed:
            continue
        out = IMG_DIR / f"{n}-{i}.png"
        if out.exists():
            continue
        try:
            resp = requests.get(pic, timeout=IMAGE_REQUEST_TIMEOUT)
            resp.raise_for_status()
            out.write_bytes(resp.content)
            _record_label(n, i, genes, sex, origin, price, birth, url)
            saved += 1
        except (requests.RequestException, OSError) as e:
            _record_failure(
                n, url, len(needed) if needed else len(pic_urls), saved,
                type(e).__name__, str(e),
            )
            failed_images += 1
        time.sleep(0.3 + 0.4 * random.random())

    expected = len(needed) if needed is not None else len(pic_urls)
    if needed is not None and saved + failed_images < len(needed):
        _record_failure(
            n, url, len(needed), saved,
            "fewer_images_than_expected",
            f"listing returned {len(pic_urls)} images, needed positions {sorted(needed)}",
        )
    return "ok", saved, expected


def download_one(driver, n, url, needed, labels_only=False):
    try:
        return _download_listing(driver, n, url, needed, labels_only=labels_only)
    except (TimeoutException, WebDriverException, requests.RequestException,
            AttributeError, IndexError, KeyError) as e:
        _record_failure(n, url, len(needed) if needed else None, 0, type(e).__name__, str(e))
        return "error", 0, 0
    except Exception as e:
        _record_failure(n, url, len(needed) if needed else None, 0,
                        f"unexpected:{type(e).__name__}", str(e))
        return "error", 0, 0


def run_smoke_test(headless: bool = False, chrome_version: Optional[int] = None,
                   login: bool = False, sample_size: int = 3):
    df = pd.read_csv(SELECTED_CSV)
    sample = df["url"].dropna().sample(sample_size, random_state=42).tolist()
    driver = _make_driver(headless=headless, chrome_version=chrome_version)
    if login:
        perform_manual_login(driver)

    cf_count = 0
    ok_count = 0
    empty_count = 0
    err_count = 0
    total_imgs = 0
    try:
        for url in sample:
            print(f"smoke: {url}", flush=True)
            try:
                driver.get(url)
            except (TimeoutException, WebDriverException) as e:
                print(f"  ERR: driver.get raised {type(e).__name__}: {e}", flush=True)
                err_count += 1
                continue
            time.sleep(PAGE_LOAD_DELAY + random.random() * PAGE_LOAD_JITTER)
            page = driver.page_source
            if _is_cloudflare_challenge(page):
                print("  CF: cloudflare challenge detected", flush=True)
                cf_count += 1
                continue
            soup = BeautifulSoup(page, "html.parser")
            try:
                pic_urls = get_pic_url(soup)
            except Exception as e:
                print(f"  ERR: get_pic_url raised {type(e).__name__}: {e}", flush=True)
                err_count += 1
                continue
            if not pic_urls:
                title = soup.find("title")
                title_text = title.text[:60] if title else "no title"
                print(f"  EMPTY: no carousel images (likely sold); title={title_text!r}", flush=True)
                empty_count += 1
                continue
            try:
                resp = requests.head(pic_urls[0], timeout=IMAGE_REQUEST_TIMEOUT, allow_redirects=True)
                ctype = resp.headers.get("content-type", "")
                if resp.status_code != 200 or not ctype.startswith("image/"):
                    print(f"  ERR: image HEAD status={resp.status_code} ctype={ctype!r}", flush=True)
                    err_count += 1
                    continue
            except requests.RequestException as e:
                print(f"  ERR: image fetch raised {type(e).__name__}: {e}", flush=True)
                err_count += 1
                continue
            ok_count += 1
            total_imgs += len(pic_urls)
            print(f"  OK: {len(pic_urls)} images; served as {ctype}", flush=True)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print(
        f"smoke: ok={ok_count} cloudflare={cf_count} empty={empty_count} err={err_count} "
        f"total_images_extracted={total_imgs}",
        flush=True,
    )
    selectors_ok = ok_count > 0
    cf_ok = cf_count <= sample_size // 2
    passed = selectors_ok and cf_ok
    if not selectors_ok:
        print("  FAIL reason: 0 listings yielded images. Selectors may be broken or "
              "all URLs sampled are sold (try --login).", flush=True)
    if not cf_ok:
        print(f"  FAIL reason: {cf_count}/{sample_size} hit Cloudflare. "
              "Try non-headless or use --login.", flush=True)
    print("smoke test: " + ("PASSED" if passed else "FAILED"), flush=True)
    return passed


def run_jobs(jobs, start, end, headless: bool = False,
             chrome_version: Optional[int] = None, login: bool = False,
             labels_only: bool = False):
    total_jobs = len(jobs)
    if end < 0 or end > total_jobs:
        end = total_jobs
    if start < 0:
        start = 0
    sliced = jobs[start:end]
    if not sliced:
        print("no jobs in range", flush=True)
        return

    if not labels_only:
        IMG_DIR.mkdir(parents=True, exist_ok=True)
    driver = _make_driver(headless=headless, chrome_version=chrome_version)
    if login:
        perform_manual_login(driver)
    total_saved = 0
    total_expected = 0
    listings_skipped = 0
    listings_errored = 0
    consecutive_cf = 0
    started_at = time.time()
    try:
        for idx, (n, url, needed) in enumerate(sliced):
            status, saved, expected = download_one(driver, n, url, needed, labels_only=labels_only)
            total_saved += saved
            total_expected += expected
            if status == "skip":
                listings_skipped += 1
            elif status == "error":
                listings_errored += 1

            # Restart Chrome if we keep hitting Cloudflare — new session = fresh fingerprint
            if status == "error" and _last_error_was_cloudflare():
                consecutive_cf += 1
                if consecutive_cf >= DRIVER_RESTART_AFTER_CF:
                    print(
                        f"  restarting Chrome after {consecutive_cf} consecutive Cloudflare hits",
                        flush=True,
                    )
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    time.sleep(30)
                    driver = _make_driver(headless=headless, chrome_version=chrome_version)
                    if login:
                        perform_manual_login(driver)
                    consecutive_cf = 0
            else:
                consecutive_cf = 0

            if idx % 5 == 0 or idx == len(sliced) - 1:
                elapsed = time.time() - started_at
                pct = 100 * (idx + 1) / len(sliced)
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta_s = (len(sliced) - (idx + 1)) / rate if rate > 0 else 0
                print(
                    f"[{idx + 1}/{len(sliced)} {pct:.1f}%] n={n} status={status} "
                    f"saved={saved}/{expected} | "
                    f"totals: saved={total_saved}/{total_expected} "
                    f"skipped={listings_skipped} errored={listings_errored} | "
                    f"rate={rate:.2f} listings/s eta={eta_s / 60:.1f}min",
                    flush=True,
                )
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    print(
        f"done. listings: {len(sliced)} processed, {listings_skipped} skipped, "
        f"{listings_errored} errored. images: {total_saved} saved of {total_expected} expected.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Re-scrape morphmarket ball python images.")
    parser.add_argument("--mode", choices=["targeted", "full"], default="targeted")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Verify CSS selectors against sampled URLs and exit.",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=3,
        help="Number of URLs to sample in smoke test (default 3).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless. Default is visible Chrome because morphmarket's Cloudflare "
        "blocks headless undetected-chromedriver.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Open the morphmarket login page first and pause so you can sign in "
        "manually. Required to access sold listings.",
    )
    parser.add_argument(
        "--chrome-version",
        type=int,
        default=None,
        help="Pin Chrome major version (e.g. 147). Auto-detected on Windows if omitted.",
    )
    parser.add_argument(
        "--url-csv",
        type=str,
        default=None,
        help="Override the URL CSV used for --mode full "
        "(default: data/complete_url.csv). Pair with rescrape_urls.py output, "
        "e.g. --url-csv data/all_urls.csv.",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Override path for the labels CSV. Used when running multiple "
        "concurrent workers on one host so each gets its own file.",
    )
    parser.add_argument(
        "--failures-csv",
        type=str,
        default=None,
        help="Override path for the failures CSV (same multi-worker reason).",
    )
    parser.add_argument(
        "--listings-csv",
        type=str,
        default=None,
        help="CSV with columns 'n' and 'url' to process exactly those listings. "
        "When set, --mode/--url-csv/--start/--end are ignored.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Visit each listing and write labels but skip image downloads "
        "and skip the all-images-present resume check. Used by Phase 1 to "
        "re-label listings whose images are already in GCS.",
    )
    args = parser.parse_args()

    if args.labels_csv:
        global LABELS_CSV
        LABELS_CSV = Path(args.labels_csv)
    if args.failures_csv:
        global FAILURES_CSV
        FAILURES_CSV = Path(args.failures_csv)

    if args.smoke_test:
        ok = run_smoke_test(
            headless=args.headless,
            chrome_version=args.chrome_version,
            login=args.login,
            sample_size=args.smoke_size,
        )
        sys.exit(0 if ok else 1)

    if args.listings_csv:
        jobs = build_listings_jobs(args.listings_csv)
        print(
            f"listings mode: {len(jobs)} listings from {args.listings_csv}"
            f"{' (labels-only)' if args.labels_only else ''}",
            flush=True,
        )
        # Process the entire listings CSV regardless of --start/--end
        start, end = 0, len(jobs)
    elif args.mode == "targeted":
        if args.url_csv:
            print("warning: --url-csv is ignored in targeted mode", flush=True)
        jobs = build_targeted_jobs()
        print(f"targeted mode: {len(jobs)} unique listings", flush=True)
        start, end = args.start, args.end
    else:
        jobs = build_full_jobs(args.url_csv)
        print(
            f"full mode: {len(jobs)} listings from "
            f"{args.url_csv or COMPLETE_URL_CSV}",
            flush=True,
        )
        start, end = args.start, args.end

    run_jobs(
        jobs,
        start,
        end,
        headless=args.headless,
        chrome_version=args.chrome_version,
        login=args.login,
        labels_only=args.labels_only,
    )


if __name__ == "__main__":
    main()
