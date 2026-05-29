"""
Script 11 — Historical Matchup Odds 2023–2024 (Holdout)
Endpoint: /historical-odds/matchups?tour=pga&event_id=EID&year=YYYY&book=B
Output: data/raw/holdout/matchup_odds_2023_2024.csv

Rate limit: 45 requests/minute → 1.5s delay between requests + retry on 429.
Appends to existing CSV so you can resume after rate limiting.

Run: python 11_pull_holdout_matchup_odds.py --key YOUR_API_KEY
     python 11_pull_holdout_matchup_odds.py --key YOUR_API_KEY --resume
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
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "holdout")
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
SEASONS  = [2023, 2024]
BOOKS    = ["pinnacle"]

FIELDNAMES = [
    "event_id", "event_name", "event_completed", "season", "calendar_year",
    "bookmaker", "bet_type", "tie_rule", "open_time", "close_time",
    "p1_dg_id", "p1_player_name", "p1_open", "p1_close",
    "p1_outcome", "p1_outcome_text",
    "p2_dg_id", "p2_player_name", "p2_open", "p2_close",
    "p2_outcome", "p2_outcome_text",
]

REQUEST_DELAY = 1.5  # seconds between requests (45/min = 1.33s min)
MAX_RETRIES = 3
RETRY_WAIT = 65  # seconds to wait on 429 before retrying


def fetch_json(url: str, retries: int = MAX_RETRIES) -> dict | list | None:
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode("utf-8")[:300]
            except: pass

            if e.code == 429 and attempt < retries - 1:
                print(f"RATE LIMITED — waiting {RETRY_WAIT}s...", end=" ", flush=True)
                time.sleep(RETRY_WAIT)
                continue
            elif e.code == 400:
                print(f"HTTP 400: {body[:80]}")
                return None
            else:
                print(f"HTTP {e.code}" + (f": {body[:80]}" if body else ""))
                return None
        except urllib.error.URLError as e:
            print(f"net error: {e.reason}")
            return None
    return None


def get_odds_event_list(api_key: str) -> list[dict]:
    url = (f"{BASE_URL}/historical-odds/event-list"
           f"?tour=pga&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data is None:
        return []
    return data if isinstance(data, list) else []


def get_already_fetched(out_path: str) -> set:
    """Read existing CSV and return set of (event_id, calendar_year, book) already fetched."""
    done = set()
    if not os.path.exists(out_path):
        return done
    try:
        import pandas as pd
        df = pd.read_csv(out_path)
        for _, row in df.drop_duplicates(subset=["event_id", "calendar_year", "bookmaker"]).iterrows():
            done.add((str(row["event_id"]), int(row["calendar_year"]), str(row["bookmaker"])))
    except Exception:
        pass
    return done


def main(api_key: str, resume: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 11 — Historical Matchup Odds 2023–2024 [Holdout]")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Books: {BOOKS}  |  Delay: {REQUEST_DELAY}s  |  Resume: {resume}")
    print("=" * 60)

    print("\n  Fetching event list...")
    events = get_odds_event_list(api_key)
    time.sleep(REQUEST_DELAY)
    print(f"  Total events: {len(events)}")

    events_in_window = []
    for e in events:
        for key in ["calendar_year", "year"]:
            try:
                val = int(e.get(key, 0))
                if val in SEASONS:
                    e["_year"] = val
                    events_in_window.append(e)
                    break
            except (ValueError, TypeError):
                continue

    print(f"  Events in 2023–2024: {len(events_in_window)}")

    out_path = os.path.join(OUT_DIR, "matchup_odds_2023_2024.csv")

    # Determine which events we already have
    already_done = set()
    write_mode = "w"
    write_header = True
    if resume:
        already_done = get_already_fetched(out_path)
        if already_done:
            write_mode = "a"
            write_header = False
            print(f"  Resuming: {len(already_done)} event/book combos already fetched")

    log_lines = []
    total_rows = 0
    skipped_done = 0
    no_data_count = 0

    with open(out_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, e in enumerate(events_in_window):
            event_id = e.get("event_id", e.get("id", ""))
            event_name = e.get("event_name", e.get("name", ""))
            cal_year = e["_year"]

            for book in BOOKS:
                # Skip if already fetched
                if (str(event_id), int(cal_year), book) in already_done:
                    skipped_done += 1
                    continue

                print(f"\n  [{i+1}/{len(events_in_window)}] {event_name} ({cal_year}) / {book}...", end=" ", flush=True)

                url = (f"{BASE_URL}/historical-odds/matchups"
                       f"?tour=pga&event_id={event_id}&year={cal_year}"
                       f"&book={book}&odds_format=decimal"
                       f"&file_format=json&key={api_key}")

                data = fetch_json(url)
                if data is None or not isinstance(data, dict):
                    print("SKIP")
                    time.sleep(REQUEST_DELAY)
                    continue

                event_completed = data.get("event_completed", "")
                season = data.get("season", "")
                odds_data = data.get("odds", [])

                if not isinstance(odds_data, list):
                    no_data_count += 1
                    print(f"no data")
                    time.sleep(REQUEST_DELAY)
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
                f.flush()  # Flush after each event so data is saved
                time.sleep(REQUEST_DELAY)

    log_path = os.path.join(LOG_DIR, "11_holdout_matchup_odds.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\nNew rows: {total_rows}\n")
        f.write(f"Skipped (already done): {skipped_done}\n")
        f.write(f"No-data events: {no_data_count}\n")
        for line in log_lines: f.write(f"  {line}\n")

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  New matchups       : {total_rows:,}")
    print(f"  Skipped (resumed)  : {skipped_done}")
    print(f"  No-data events     : {no_data_count}")
    print(f"  Output             : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Skip events already in the output CSV")
    args = parser.parse_args()
    main(args.key, resume=args.resume)
