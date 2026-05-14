"""Tiny morphmarket scraping probe for a fresh GCE instance.

Hits 10 ball python listing URLs using undetected-chromedriver under Xvfb,
reports CF rate / success rate / images per listing / wall time. The script
intentionally has no dependency on the rest of the repo so it can be SCPed and
run standalone.
"""

import json
import random
import re
import time

import requests
import undetected_chromedriver as uc
from bs4 import BeautifulSoup


URLS = [
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762183",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3839051",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762182",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762181",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762180",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762179",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762178",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762177",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762176",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3762175",
]

THUMB_RE = re.compile(r"^thumbCarouselImage--")
PAGE_DELAY = 6.0
PAGE_JITTER = 2.0


def get_pic_urls(soup):
    imgs = soup.find_all("img", class_=THUMB_RE)
    urls = [
        img["srcset"].split(", ")[0].split(" ")[0]
        for img in imgs
        if "srcset" in img.attrs
    ]
    if not urls:
        srcs = soup.find_all("source")
        if srcs and "srcset" in srcs[0].attrs:
            urls = [srcs[0]["srcset"].split(" ")[0]]
    return urls


def _detect_chrome_major():
    import re as _re
    import subprocess
    try:
        out = subprocess.check_output(["google-chrome", "--version"], text=True).strip()
        m = _re.search(r"(\d+)\.", out)
        if m:
            return int(m.group(1))
    except Exception:
        return None


def main():
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=options, version_main=_detect_chrome_major())

    cf = 0
    ok = 0
    empty = 0
    err = 0
    total_imgs = 0
    image_fetch_ok = 0
    started = time.time()

    for url in URLS:
        try:
            driver.get(url)
            time.sleep(PAGE_DELAY + random.random() * PAGE_JITTER)
            page = driver.page_source
            if "Just a moment" in page or "cf-challenge" in page.lower():
                cf += 1
                print(f"CF      {url[-7:]}", flush=True)
                continue
            soup = BeautifulSoup(page, "html.parser")
            urls_found = get_pic_urls(soup)
            if not urls_found:
                empty += 1
                title = soup.find("title")
                t = title.text[:50] if title else None
                print(f"EMPTY   {url[-7:]}  title={t!r}", flush=True)
                continue
            ok += 1
            total_imgs += len(urls_found)
            try:
                resp = requests.get(urls_found[0], timeout=15)
                if resp.status_code == 200 and resp.headers.get(
                    "content-type", ""
                ).startswith("image/"):
                    image_fetch_ok += 1
            except requests.RequestException:
                pass
            print(f"OK      {url[-7:]}  imgs={len(urls_found)}", flush=True)
        except Exception as e:
            err += 1
            print(f"ERR     {url[-7:]}  {type(e).__name__}: {e}", flush=True)

    elapsed = time.time() - started
    summary = {
        "n_urls": len(URLS),
        "ok": ok,
        "cloudflare": cf,
        "empty": empty,
        "err": err,
        "total_imgs_extracted": total_imgs,
        "image_downloads_verified": image_fetch_ok,
        "elapsed_sec": round(elapsed, 1),
        "sec_per_url": round(elapsed / len(URLS), 2),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)

    try:
        driver.quit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
