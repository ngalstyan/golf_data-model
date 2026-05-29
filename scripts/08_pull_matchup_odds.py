"""
Script 08 — Historical Matchup Odds 2019–2022 (FIXED)
Endpoint: /historical-odds/matchups?tour=pga&event_id=EID&year=YYYY&book=B
Output: data/raw/matchup_odds_2019_2022.csv

CONFIRMED from diagnostics:
- Single-event response:
  { book, event_completed, event_id, event_name,
    odds: [ {bet_type, close_time, open_time,
             p1_close, p1_dg_id, p1_open, p1_outcome, p1_outcome_text, p1_player_name,
             p2_close, p2_dg_id, p2_open, p2_outcome, p2_outcome_text, p2_player_name,
             tie_rule}, ... ],
    season, year }
- bet_type values: "72-hole Match", possibly others
- p1/p2 fields are FLAT (not nested dicts) — the old code crashed because
  it called m.get("p1").get(...) on what was actually a flat structure

Run: python 08_pull_matchup_odds.py --key YOUR_API_KEY
"""

import argparse
import csv
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "https://feeds.datagolf.com"
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
SEASONS  = [2019, 2020, 2021, 2022]
BOOKS    = ["pinnacle", "draftkings"]

FIELDNAMES = [
    "event_id", "event_name", "event_completed", "season", "calendar_year",
    "bookmaker", "bet_type", "tie_rule", "open_time", "close_time",
    "p1_dg_id", "p1_player_name", "p1_open", "p1_close",
    "p1_outcome", "p1_outcome_text",
    "p2_dg_id", "p2_player_name", "p2_open", "p2_close",
    "p2_outcome", "p2_outcome_text",
]


def fetch_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try: body = e.read().decode("utf-8")[:300]
        except: pass
        print(f"HTTP {e.code}" + (f": {body[:80]}" if body else ""))
        return None
    except urllib.error.URLError as e:
        print(f"net error: {e.reason}")
        return None


def get_odds_event_list(api_key: str) -> list[dict]:
    url = (f"{BASE_URL}/historical-odds/event-list"
           f"?tour=pga&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data is None:
        return []
    return data if isinstance(data, list) else []


def main(api_key: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 08 — Historical Matchup Odds 2019–2022 [FIXED]")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Books: {BOOKS}")
    print("=" * 60)

    print("\n  Fetching event list...")
    events = get_odds_event_list(api_key)
    print(f"  Total events: {len(events)}")

    events_in_window = []
    for e in events:
        for key in ["calendar_year", "year"]:
            try:
                val = int(e.get(key, 0))
                if 2019 <= val <= 2022:
                    e["_year"] = val
                    events_in_window.append(e)
                    break
            except (ValueError, TypeError):
                continue

    print(f"  Events in 2019–2022: {len(events_in_window)}")

    out_path = os.path.join(OUT_DIR, "matchup_odds_2019_2022.csv")
    log_lines = []
    total_rows = 0
    no_data_count = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for e in events_in_window:
            event_id = e.get("event_id", e.get("id", ""))
            event_name = e.get("event_name", e.get("name", ""))
            cal_year = e["_year"]

            print(f"\n  {event_name} ({cal_year}):")

            for book in BOOKS:
                print(f"    {book}...", end=" ", flush=True)

                url = (f"{BASE_URL}/historical-odds/matchups"
                       f"?tour=pga&event_id={event_id}&year={cal_year}"
                       f"&book={book}&odds_format=decimal"
                       f"&file_format=json&key={api_key}")

                data = fetch_json(url)
                if data is None or not isinstance(data, dict):
                    print("SKIP")
                    time.sleep(0.5)
                    continue

                event_completed = data.get("event_completed", "")
                season = data.get("season", "")
                odds_data = data.get("odds", [])

                # odds can be a string when no data exists
                if not isinstance(odds_data, list):
                    no_data_count += 1
                    print(f"no data")
                    time.sleep(0.5)
                    continue

                book_rows = 0
                for m in odds_data:
                    if not isinstance(m, dict):
                        continue

                    row = {
                        "event_id": event_id,
                        "event_name": event_name,
                        "event_completed": event_completed,
                        "season": season,
                        "calendar_year": cal_year,
                        "bookmaker": book,
                        "bet_type": m.get("bet_type", ""),
                        "tie_rule": m.get("tie_rule", ""),
                        "open_time": m.get("open_time", ""),
                        "close_time": m.get("close_time", ""),
                        "p1_dg_id": m.get("p1_dg_id", ""),
                        "p1_player_name": m.get("p1_player_name", ""),
                        "p1_open": m.get("p1_open", ""),
                        "p1_close": m.get("p1_close", ""),
                        "p1_outcome": m.get("p1_outcome", ""),
                        "p1_outcome_text": m.get("p1_outcome_text", ""),
                        "p2_dg_id": m.get("p2_dg_id", ""),
                        "p2_player_name": m.get("p2_player_name", ""),
                        "p2_open": m.get("p2_open", ""),
                        "p2_close": m.get("p2_close", ""),
                        "p2_outcome": m.get("p2_outcome", ""),
                        "p2_outcome_text": m.get("p2_outcome_text", ""),
                    }
                    writer.writerow(row)
                    book_rows += 1
                    total_rows += 1

                print(f"{book_rows} matchups")
                log_lines.append(f"{event_name}/{book}: {book_rows}")
                time.sleep(0.7)

    log_path = os.path.join(LOG_DIR, "08_matchup_odds.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\nTotal: {total_rows}\n")
        f.write(f"No-data events: {no_data_count}\n")
        for line in log_lines: f.write(f"  {line}\n")

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Total matchups     : {total_rows:,}")
    print(f"  No-data events     : {no_data_count}")
    print(f"  Output             : {out_path}")
    print(f"\n  ✓ Script 08 complete. Next: run 09_pull_course_fit.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    main(args.key)
