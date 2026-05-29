"""
Script 02 — Pull Tournament Schedule / Event List 2017–2022
Endpoint: /historical-raw-data/event-list  (correct endpoint for historical data)
Output: data/raw/schedule_2017_2022.csv

NOTE: get-schedule only returns current/upcoming seasons.
For historical event metadata we use the historical-raw-data/event-list
endpoint which returns all events available in the raw data system,
including event_id, event_name, course, calendar_year, and data coverage flags.

Run: python 02_pull_schedule.py --key YOUR_API_KEY
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "https://feeds.datagolf.com"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")

# Tours to pull event lists for
TOURS = ["pga", "kft", "euro"]

# Calendar years we care about
YEARS_WANTED = set(range(2017, 2023))   # 2017–2022 inclusive

MAJOR_FRAGMENTS = ["masters", "pga championship", "u.s. open", "us open",
                   "the open championship", "open championship"]
WGC_FRAGMENTS   = ["wgc", "world golf championships", "dell technologies",
                   "bridgestone", "mexico championship", "match play",
                   "fedex st. jude"]

FIELDNAMES = [
    "tour",
    "event_id",
    "event_name",
    "calendar_year",
    "course",
    "course_id",
    "latitude",
    "longitude",
    "sg_categories",      # "yes", "partial", "no" — SG data availability
    "traditional_stats",  # same
    "is_major",
    "is_wgc",
]


def fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("  → Invalid API key.")
        elif e.code == 403:
            print("  → Requires Scratch PLUS.")
        elif e.code == 400:
            print(f"  → Bad request. URL was: {url}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"  Network error: {e.reason}")
        sys.exit(1)


def classify(name: str) -> tuple[int, int]:
    low = name.lower()
    return (
        int(any(f in low for f in MAJOR_FRAGMENTS)),
        int(any(f in low for f in WGC_FRAGMENTS)),
    )


def main(api_key: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 02 — Historical Event List 2017–2022")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tours: {TOURS}")
    print("=" * 60)

    all_rows = []
    warnings = []

    for tour in TOURS:
        # Correct endpoint: historical-raw-data/event-list
        # Returns ALL historical events for the tour — we filter by year
        url = (f"{BASE_URL}/historical-raw-data/event-list"
               f"?tour={tour}&file_format=json&key={api_key}")

        print(f"\n  Fetching event list for tour={tour}...")
        data = fetch_json(url)

        # Response is a list of event objects
        events = data if isinstance(data, list) else data.get("events", [])
        print(f"    Total events in system: {len(events)}")

        # Filter to years we want
        filtered = []
        for ev in events:
            yr = ev.get("calendar_year", ev.get("year", 0))
            try:
                if int(yr) in YEARS_WANTED:
                    filtered.append(ev)
            except (ValueError, TypeError):
                warnings.append(f"{tour}: event {ev.get('event_id')} has bad year '{yr}'")

        print(f"    Events in 2017–2022: {len(filtered)}")

        for ev in filtered:
            name = ev.get("event_name", ev.get("name", ""))
            is_major, is_wgc = classify(name)
            row = {
                "tour":              tour,
                "event_id":          ev.get("event_id", ""),
                "event_name":        name,
                "calendar_year":     ev.get("calendar_year", ev.get("year", "")),
                "course":            ev.get("course", ev.get("course_name", "")),
                "course_id":         ev.get("course_id", ""),
                "latitude":          ev.get("latitude",  ev.get("lat", "")),
                "longitude":         ev.get("longitude", ev.get("lon", ev.get("lng", ""))),
                "sg_categories":     ev.get("sg_categories", ""),
                "traditional_stats": ev.get("traditional_stats", ""),
                "is_major":          is_major,
                "is_wgc":            is_wgc,
            }
            all_rows.append(row)

        time.sleep(0.5)

    # ── Validate ──────────────────────────────────────────────────────────────
    missing_id   = [r for r in all_rows if not r["event_id"]]
    missing_name = [r for r in all_rows if not r["event_name"]]
    missing_yr   = [r for r in all_rows if not r["calendar_year"]]

    if missing_id:
        warnings.append(f"{len(missing_id)} rows missing event_id")
    if missing_name:
        warnings.append(f"{len(missing_name)} rows missing event_name")
    if missing_yr:
        warnings.append(f"{len(missing_yr)} rows missing calendar_year")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    ! {w}")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "schedule_2017_2022.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Written: {out_path}")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "02_schedule.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\n")
        f.write(f"Tours: {TOURS}\n")
        f.write(f"Total rows: {len(all_rows)}\n")
        f.write(f"Warnings: {len(warnings)}\n")
        for w in warnings:
            f.write(f"  {w}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    by_tour = {}
    by_year = {}
    for r in all_rows:
        by_tour[r["tour"]]               = by_tour.get(r["tour"], 0) + 1
        by_year[str(r["calendar_year"])] = by_year.get(str(r["calendar_year"]), 0) + 1

    sg_yes     = sum(1 for r in all_rows if r.get("sg_categories") == "yes")
    sg_partial = sum(1 for r in all_rows if r.get("sg_categories") == "partial")
    sg_no      = sum(1 for r in all_rows if r.get("sg_categories") == "no")
    majors     = sum(1 for r in all_rows if r["is_major"])

    print("\n── Summary ──────────────────────────────────────────────")
    print("  By tour:")
    for t, cnt in sorted(by_tour.items()):
        print(f"    {t.upper():<6} {cnt} events")
    print("  By year:")
    for yr in sorted(by_year.keys()):
        print(f"    {yr}  {by_year[yr]} events")
    print(f"  SG data available  : {sg_yes} full / {sg_partial} partial / {sg_no} none")
    print(f"  Majors             : {majors}")
    print(f"\n  ✓ Script 02 complete. Next: run 03_pull_sg_data.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull DataGolf historical event list")
    parser.add_argument("--key", required=True, help="DataGolf API key")
    args = parser.parse_args()
    main(args.key)
