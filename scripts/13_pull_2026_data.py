"""
Script 13 — Pull 2026 Season Data (SG Rounds + Schedule + Odds + Matchup Odds)

Pulls all PGA Tour data for 2026 from the DataGolf API:
  1. Round-level SG data → data/raw/sg_rounds_pga_2026.csv
  2. Schedule/events     → data/raw/schedule_2026.csv
  3. Outright odds       → data/raw/odds_2026.csv
  4. Matchup odds        → data/raw/matchup_odds_2026.csv

Rate limit: 45 requests/minute → 1.5s delay between requests.
Supports --resume to skip already-fetched events (for matchup/odds).

Run: python 13_pull_2026_data.py --key YOUR_API_KEY
     python 13_pull_2026_data.py --key YOUR_API_KEY --resume
     python 13_pull_2026_data.py --key YOUR_API_KEY --only rounds
     python 13_pull_2026_data.py --key YOUR_API_KEY --only matchups
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
from pathlib import Path

BASE_URL = "https://feeds.datagolf.com"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "golf_model" / "data" / "raw"
LOG_DIR = SCRIPT_DIR.parent / "logs"
SEASONS = [2026]
BOOKS = ["pinnacle", "draftkings"]
MARKETS = ["win", "top_5", "make_cut"]

REQUEST_DELAY = 1.5  # seconds between requests
MAX_RETRIES = 3
RETRY_WAIT = 65  # seconds to wait on 429

MATCHUP_FIELDNAMES = [
    "event_id", "event_name", "event_completed", "season", "calendar_year",
    "bookmaker", "bet_type", "tie_rule", "open_time", "close_time",
    "p1_dg_id", "p1_player_name", "p1_open", "p1_close",
    "p1_outcome", "p1_outcome_text",
    "p2_dg_id", "p2_player_name", "p2_open", "p2_close",
    "p2_outcome", "p2_outcome_text",
]


def fetch_json(url: str, retries: int = MAX_RETRIES) -> dict | list | None:
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
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
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return None
    return None


def fetch_csv(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        print(f"HTTP {e.code}: {e.reason}" + (f" — {body}" if body else ""))
        return None
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        return None


def get_already_fetched(out_path: Path, key_cols: list[str]) -> set:
    """Read existing CSV and return set of key tuples already fetched."""
    done = set()
    if not out_path.exists():
        return done
    try:
        import pandas as pd
        df = pd.read_csv(out_path)
        for _, row in df.drop_duplicates(subset=key_cols).iterrows():
            done.add(tuple(str(row[c]) for c in key_cols))
    except Exception:
        pass
    return done


# ==========================================================================
# 1. SG ROUNDS
# ==========================================================================

def pull_sg_rounds(api_key: str):
    print("\n" + "=" * 60)
    print("STEP 1: SG Rounds (PGA 2026)")
    print("=" * 60)

    out_path = OUT_DIR / "sg_rounds_pga_2026.csv"
    total_rows = 0
    fieldnames = None
    csv_file = None
    writer = None

    for year in SEASONS:
        print(f"\n  Fetching PGA rounds for {year}...", end=" ", flush=True)
        url = (f"{BASE_URL}/historical-raw-data/rounds"
               f"?tour=pga&event_id=all&year={year}"
               f"&file_format=csv&key={api_key}")

        csv_text = fetch_csv(url)
        if csv_text is None:
            print("FAILED")
            continue

        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        if not rows:
            print("0 rows")
            continue

        if fieldnames is None:
            fieldnames = reader.fieldnames
            csv_file = open(out_path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

        for row in rows:
            writer.writerow(row)
            total_rows += 1

        events = set(r.get("event_id", "") for r in rows)
        sg_present = sum(1 for r in rows if r.get("sg_ott", ""))
        print(f"{len(rows):,} rows | {len(events)} events | SG: {sg_present}/{len(rows)}")
        time.sleep(2.0)

    if csv_file:
        csv_file.close()

    print(f"\n  Saved: {out_path} ({total_rows:,} rows)")
    return total_rows


# ==========================================================================
# 2. SCHEDULE
# ==========================================================================

def pull_schedule(api_key: str):
    print("\n" + "=" * 60)
    print("STEP 2: Schedule/Events (2026)")
    print("=" * 60)

    out_path = OUT_DIR / "schedule_2026.csv"

    for year in SEASONS:
        print(f"\n  Fetching event list for {year}...", end=" ", flush=True)
        url = (f"{BASE_URL}/historical-raw-data/event-list"
               f"?file_format=csv&key={api_key}")

        csv_text = fetch_csv(url)
        if csv_text is None:
            print("FAILED")
            continue

        import pandas as pd
        from io import StringIO
        df = pd.read_csv(StringIO(csv_text))

        # Filter to 2026 events
        year_col = None
        for col in ["calendar_year", "year", "season"]:
            if col in df.columns:
                year_col = col
                break

        if year_col:
            df = df[df[year_col].astype(int) == year]

        df.to_csv(out_path, index=False)
        print(f"{len(df)} events")
        print(f"  Saved: {out_path}")
        time.sleep(2.0)


# ==========================================================================
# 3. OUTRIGHT ODDS
# ==========================================================================

def pull_odds(api_key: str, resume: bool = False):
    print("\n" + "=" * 60)
    print("STEP 3: Outright Odds (2026)")
    print("=" * 60)

    out_path = OUT_DIR / "odds_2026.csv"

    # Get event list from odds system
    print("  Fetching odds event list...", end=" ", flush=True)
    url = f"{BASE_URL}/historical-odds/event-list?tour=pga&file_format=json&key={api_key}"
    all_events = fetch_json(url) or []
    if isinstance(all_events, dict):
        all_events = all_events.get("events", [])
    time.sleep(REQUEST_DELAY)

    events_2026 = [
        ev for ev in all_events
        if int(ev.get("calendar_year", ev.get("year", 0))) in SEASONS
    ]
    print(f"{len(events_2026)} events in 2026")

    if not events_2026:
        print("  No 2026 events found in odds system yet.")
        return

    # Check resume
    already_done = set()
    if resume and out_path.exists():
        already_done = get_already_fetched(out_path, ["event_id", "bookmaker", "market"])
        print(f"  Resuming: {len(already_done)} combos already fetched")

    all_rows = []
    for ev in events_2026:
        event_id = ev.get("event_id", ev.get("dg_event_id", ""))
        event_name = ev.get("event_name", ev.get("name", str(event_id)))
        cal_year = ev.get("calendar_year", ev.get("year", ""))
        print(f"\n  {event_name} ({cal_year}):")

        for book in BOOKS:
            for market in MARKETS:
                if (str(event_id), book, market) in already_done:
                    continue

                req_url = (
                    f"{BASE_URL}/historical-odds/outrights"
                    f"?tour=pga&event_id={event_id}&year={cal_year}"
                    f"&market={market}&book={book}"
                    f"&odds_format=decimal&file_format=json&key={api_key}"
                )
                data = fetch_json(req_url)
                if data is None or isinstance(data, str):
                    time.sleep(REQUEST_DELAY)
                    continue

                if isinstance(data, dict):
                    players = data.get("odds", data.get("players", []))
                elif isinstance(data, list):
                    players = data
                else:
                    time.sleep(REQUEST_DELAY)
                    continue

                if not isinstance(players, list):
                    time.sleep(REQUEST_DELAY)
                    continue

                count = 0
                for row in players:
                    if not isinstance(row, dict):
                        continue
                    row["event_id"] = event_id
                    row["event_name"] = event_name
                    row["event_completed"] = ev.get("event_completed", "")
                    row["calendar_year"] = cal_year
                    row["bookmaker"] = book
                    row["market"] = market
                    all_rows.append(row)
                    count += 1

                print(f"    {book}/{market}: {count} rows")
                time.sleep(REQUEST_DELAY)

    if all_rows:
        import pandas as pd
        combined = pd.DataFrame(all_rows)
        if resume and out_path.exists():
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, combined], ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path} ({len(combined):,} rows)")
    else:
        print("  No odds data collected.")


# ==========================================================================
# 4. MATCHUP ODDS
# ==========================================================================

def pull_matchup_odds(api_key: str, resume: bool = False):
    print("\n" + "=" * 60)
    print("STEP 4: Matchup Odds (2026)")
    print("=" * 60)

    out_path = OUT_DIR / "matchup_odds_2026.csv"

    # Get event list
    print("  Fetching event list...", end=" ", flush=True)
    events = fetch_json(
        f"{BASE_URL}/historical-odds/event-list?tour=pga&file_format=json&key={api_key}"
    ) or []
    time.sleep(REQUEST_DELAY)

    events_2026 = []
    for e in events:
        for key in ["calendar_year", "year"]:
            try:
                val = int(e.get(key, 0))
                if val in SEASONS:
                    e["_year"] = val
                    events_2026.append(e)
                    break
            except (ValueError, TypeError):
                continue

    print(f"{len(events_2026)} events in 2026")

    if not events_2026:
        print("  No 2026 events found in odds system yet.")
        return

    # Resume support
    already_done = set()
    write_mode = "w"
    write_header = True
    if resume and out_path.exists():
        already_done = get_already_fetched(out_path, ["event_id", "calendar_year", "bookmaker"])
        if already_done:
            write_mode = "a"
            write_header = False
            print(f"  Resuming: {len(already_done)} combos already fetched")

    total_rows = 0
    no_data = 0

    with open(out_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCHUP_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, e in enumerate(events_2026):
            event_id = e.get("event_id", e.get("id", ""))
            event_name = e.get("event_name", e.get("name", ""))
            cal_year = e["_year"]

            for book in ["pinnacle"]:  # Pinnacle only for matchups
                if (str(event_id), str(cal_year), book) in already_done:
                    continue

                print(f"\n  [{i+1}/{len(events_2026)}] {event_name} ({cal_year}) / {book}...",
                      end=" ", flush=True)

                url = (f"{BASE_URL}/historical-odds/matchups"
                       f"?tour=pga&event_id={event_id}&year={cal_year}"
                       f"&book={book}&odds_format=decimal"
                       f"&file_format=json&key={api_key}")

                data = fetch_json(url)
                if data is None or not isinstance(data, dict):
                    print("SKIP")
                    time.sleep(REQUEST_DELAY)
                    continue

                odds_data = data.get("odds", [])
                if not isinstance(odds_data, list):
                    no_data += 1
                    print("no data")
                    time.sleep(REQUEST_DELAY)
                    continue

                book_rows = 0
                for m in odds_data:
                    if not isinstance(m, dict):
                        continue
                    row = {
                        "event_id": event_id,
                        "event_name": event_name,
                        "event_completed": data.get("event_completed", ""),
                        "season": data.get("season", ""),
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
                f.flush()
                time.sleep(REQUEST_DELAY)

    print(f"\n  Saved: {out_path} ({total_rows:,} new matchups)")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull 2026 PGA data from DataGolf API")
    parser.add_argument("--key", required=True, help="DataGolf API key")
    parser.add_argument("--resume", action="store_true",
                        help="Skip events already in output CSVs")
    parser.add_argument("--only", choices=["rounds", "schedule", "odds", "matchups"],
                        help="Pull only one data type")
    args = parser.parse_args()

    print("=" * 60)
    print("Script 13 — Pull 2026 PGA Season Data")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:  {OUT_DIR}")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.only is None or args.only == "rounds":
        pull_sg_rounds(args.key)
    if args.only is None or args.only == "schedule":
        pull_schedule(args.key)
    if args.only is None or args.only == "odds":
        pull_odds(args.key, resume=args.resume)
    if args.only is None or args.only == "matchups":
        pull_matchup_odds(args.key, resume=args.resume)

    print(f"\n{'=' * 60}")
    print("Done! Files saved to:", OUT_DIR)
    print(f"{'=' * 60}")
