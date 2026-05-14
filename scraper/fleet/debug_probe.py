"""Probe specific listing URLs against the live scraper code, with timing.

Designed to be SCPed to a GCE debug instance after rescrape_images.py is in
the same directory. Runs the same _make_driver, get_pic_url, get_gene as the
fleet, so any GCE-vs-local timing or rendering difference is reproducible.
"""

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescrape_images as r  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from selenium.common.exceptions import TimeoutException  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.support import expected_conditions as EC  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402


URLS = [
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3610959",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3526156",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3725045",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/2785557",
    "https://www.morphmarket.com/gb/c/reptiles/pythons/ball-pythons/3594474",
    "https://www.morphmarket.com/us/c/reptiles/pythons/ball-pythons/3575545",
]


def main():
    driver = r._make_driver()
    results = []
    for url in URLS:
        rec = {"url": url[-7:]}
        try:
            t0 = time.time()
            driver.get(url)
            rec["t_get"] = round(time.time() - t0, 2)
            try:
                WebDriverWait(driver, r.CAROUSEL_WAIT_TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, r.CAROUSEL_SELECTOR))
                )
                rec["carousel"] = "found"
            except TimeoutException:
                rec["carousel"] = "timeout"
            rec["t_total"] = round(time.time() - t0, 2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            rec["imgs"] = len(r.get_pic_url(soup))
            rec["genes"] = r.get_gene(soup)[:80]
            rec["sex"] = r.get_sex(soup)
            rec["price"] = r.get_price(soup)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[:500]
        print(json.dumps(rec), flush=True)
        results.append(rec)

    try:
        driver.quit()
    except Exception:
        pass

    success = sum(1 for r_ in results if r_.get("imgs", 0) > 0)
    print(f"SUMMARY: {success}/{len(URLS)} listings yielded images", flush=True)


if __name__ == "__main__":
    main()
