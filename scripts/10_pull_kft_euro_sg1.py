"""
Script 10 — KFT + Euro + PGA 2017–2018 SG Data (FIXED v3)
Endpoint: /historical-raw-data/rounds?tour=T&event_id=all&year=Y&file_format=csv
Output: sg_rounds_pga_2017_2018.csv, sg_rounds_kft_2017_2022.csv, sg_rounds_euro_2017_2022.csv

v3 changes:
- Retry logic (up to 3 attempts per year) for timeout/network failures
- Longer timeout (600s) for large bulk CSV downloads
- Per-event fallback: if bulk CSV fails after retries, falls back to
  fetching event-by-event using the event list
- This addresses the validation failures:
  * KFT missing 2022
  * Euro missing 2017, 2021, 2022

Run: python 10_pull_kft_euro_sg.py --key YOUR_API_KEY
"""

import argparse
import csv
import io
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "https://feeds.datagolf.com"
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

TOUR_CONFIGS = [
    ("pga",  [2017, 2018],                               "sg_rounds_pga_2017_2018.csv"),
    ("kft",  [2017, 2018, 2019, 2020, 2021, 2022],       "sg_rounds_kft_2017_2022.csv"),
    ("euro", [2017, 2018, 2019, 2020, 2021, 2022],       "sg_rounds_euro_2017_2022.csv"),
]

MAX_RETRIES = 3
BULK_TIMEOUT = 600    # 10 min for large bulk CSV
EVENT_TIMEOUT = 120   # 2 min for single event


def fetch_csv(url: str, timeout: int = BULK_TIMEOUT) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = ""
        try: body = e.read().decode("utf-8")[:300]
        except: pass
        print(f"HTTP {e.code}: {e.reason}" + (f" — {body}" if body else ""))
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"timeout/network: {e}")
        return None


def fetch_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"error: {e}")
        return None


def fetch_bulk_with_retry(tour: str, year: int, api_key: str) -> tuple[list[dict], list[str] | None]:
    """Try bulk CSV up to MAX_RETRIES times. Returns (rows, fieldnames) or ([], None)."""
    url = (f"{BASE_URL}/historical-raw-data/rounds"
           f"?tour={tour}&event_id=all&year={year}"
           f"&file_format=csv&key={api_key}")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"    Attempt {attempt}/{MAX_RETRIES} (bulk CSV)...", end=" ", flush=True)
        csv_text = fetch_csv(url)
        if csv_text is not None:
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)
            if rows:
                print(f"✓ {len(rows):,} rows")
                return rows, reader.fieldnames
            print("0 rows")
            return [], None
        if attempt < MAX_RETRIES:
            wait = 10 * attempt
            print(f"retrying in {wait}s...")
            time.sleep(wait)
        else:
            print("FAILED all attempts")
    return [], None


def fetch_per_event_fallback(tour: str, year: int, api_key: str) -> tuple[list[dict], list[str] | None]:
    """Fallback: get event list, then fetch each event individually."""
    print(f"    Trying per-event fallback for {tour}/{year}...")

    # Get event list
    elist_url = (f"{BASE_URL}/historical-raw-data/event-list"
                 f"?tour={tour}&file_format=json&key={api_key}")
    events = fetch_json(elist_url)
    if not events or not isinstance(events, list):
        print(f"    Could not fetch event list for {tour}")
        return [], None

    # Filter to this year
    year_events = []
    for e in events:
        for key in ["calendar_year", "year"]:
            try:
                if int(e.get(key, 0)) == year:
                    year_events.append(e)
                    break
            except (ValueError, TypeError):
                continue

    print(f"    Found {len(year_events)} events for {tour}/{year}")

    all_rows = []
    fieldnames = None

    for e in year_events:
        eid = e.get("event_id", e.get("id", ""))
        ename = e.get("event_name", e.get("name", "?"))
        print(f"      {ename}...", end=" ", flush=True)

        url = (f"{BASE_URL}/historical-raw-data/rounds"
               f"?tour={tour}&event_id={eid}&year={year}"
               f"&file_format=csv&key={api_key}")
        csv_text = fetch_csv(url, timeout=EVENT_TIMEOUT)

        if csv_text is None:
            print("skip")
            time.sleep(1)
            continue

        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        if rows:
            if fieldnames is None:
                fieldnames = reader.fieldnames
            all_rows.extend(rows)
            print(f"{len(rows)} rows")
        else:
            print("0 rows")
        time.sleep(1.0)

    return all_rows, fieldnames


def main(api_key: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 10 — KFT + Euro + PGA 2017–2018 SG Data [FIXED v3]")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Features: retry × {MAX_RETRIES}, per-event fallback")
    print("=" * 60)

    log_lines = []

    for tour, seasons, filename in TOUR_CONFIGS:
        print(f"\n{'─' * 60}")
        print(f"  Tour: {tour.upper()}  Seasons: {seasons}")
        print(f"{'─' * 60}")

        out_path = os.path.join(OUT_DIR, filename)
        tour_rows = 0
        fieldnames = None
        csv_out = None
        writer = None

        for year in seasons:
            print(f"\n  Season {year}:")

            # Try bulk first
            rows, fnames = fetch_bulk_with_retry(tour, year, api_key)

            # If bulk failed, try per-event
            if not rows:
                rows, fnames = fetch_per_event_fallback(tour, year, api_key)

            if not rows:
                log_lines.append(f"EMPTY: {tour}/{year} (0 rows after all attempts)")
                continue

            if fieldnames is None and fnames:
                fieldnames = fnames
                csv_out = open(out_path, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(csv_out, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()

            if writer is None:
                continue

            events = set(r.get("event_id", "") for r in rows)
            for row in rows:
                writer.writerow(row)
                tour_rows += 1

            print(f"    → {len(rows):,} rows | {len(events)} events")
            log_lines.append(f"{tour}/{year}: {len(rows):,} rows, {len(events)} events")
            time.sleep(2.0)

        if csv_out:
            csv_out.close()

        print(f"\n  {tour.upper()} complete: {tour_rows:,} rows → {out_path}")

    # Log
    log_path = os.path.join(LOG_DIR, "10_kft_euro_sg.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\n")
        for line in log_lines: f.write(f"  {line}\n")

    # Final summary
    print(f"\n── Final Summary ────────────────────────────────────────")
    for tour, seasons, filename in TOUR_CONFIGS:
        path = os.path.join(OUT_DIR, filename)
        if os.path.exists(path):
            with open(path) as f:
                n = sum(1 for _ in f) - 1
            print(f"  {filename:<45} {n:>7,} rows")
        else:
            print(f"  {filename:<45} MISSING")

    print(f"\n  ✓ Script 10 complete. Run 05_validate_all.py for final check.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    main(args.key)
