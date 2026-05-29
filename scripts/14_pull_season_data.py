"""
Script 14 — Pull Season Data (SG Rounds + Schedule + Odds + Matchup Odds)

Unified script that pulls PGA Tour data for multiple seasons from DataGolf API.
Replaces scripts 12 and 13 with smart incremental behavior:

  - SG rounds: downloads, compares with existing file, only overwrites if new data
  - Schedule: always overwrites (tiny, one API call)
  - Odds: skips completed events by default, re-fetches incomplete/new events
  - Matchups: same smart resume as odds

Output per season:
  1. Round-level SG data → data/raw/sg_rounds_pga_{year}.csv
  2. Schedule/events     → data/raw/schedule_{year}.csv
  3. Outright odds       → data/raw/odds_{year}.csv
  4. Matchup odds        → data/raw/matchup_odds_{year}.csv

Rate limit: 45 requests/minute → 1.5s delay between requests.

Run: python 14_pull_season_data.py --key YOUR_API_KEY
     python 14_pull_season_data.py --key YOUR_API_KEY --seasons 2026
     python 14_pull_season_data.py --key YOUR_API_KEY --only rounds
     python 14_pull_season_data.py --key YOUR_API_KEY --full
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
DEFAULT_SEASONS = [2025, 2026]
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


def get_completed_events(out_path: Path, outcome_col: str = "outcome") -> set[str]:
    """Return event_ids where outcome data exists (event results are final).

    For odds CSVs, checks the 'outcome' column (finishing positions like '1', 'T5', 'CUT').
    For matchup CSVs, checks 'p1_outcome' column.
    An event is considered completed if >50% of its rows have a non-null outcome.
    """
    if not out_path.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(out_path)
        if outcome_col not in df.columns:
            return set()
        completed = set()
        for eid, group in df.groupby("event_id"):
            filled = group[outcome_col].notna().sum()
            if filled > len(group) * 0.5:
                completed.add(str(eid))
        return completed
    except Exception:
        return set()


# ==========================================================================
# 1. SG ROUNDS — download, compare, conditional overwrite
# ==========================================================================

def pull_sg_rounds(api_key: str, seasons: list[int]):
    print("\n" + "=" * 60)
    print(f"STEP 1: SG Rounds (PGA {', '.join(str(s) for s in seasons)})")
    print("=" * 60)

    for year in seasons:
        out_path = OUT_DIR / f"sg_rounds_pga_{year}.csv"
        print(f"\n  Fetching PGA rounds for {year}...", end=" ", flush=True)
        url = (f"{BASE_URL}/historical-raw-data/rounds"
               f"?tour=pga&event_id=all&year={year}"
               f"&file_format=csv&key={api_key}")

        csv_text = fetch_csv(url)
        if csv_text is None:
            print("FAILED")
            continue

        reader = csv.DictReader(io.StringIO(csv_text))
        new_rows = list(reader)
        new_fieldnames = reader.fieldnames

        if not new_rows:
            print("0 rows from API")
            continue

        new_count = len(new_rows)
        new_events = set(r.get("event_id", "") for r in new_rows)
        sg_present = sum(1 for r in new_rows if r.get("sg_ott", ""))

        # Compare with existing file
        if out_path.exists():
            try:
                import pandas as pd
                existing = pd.read_csv(out_path)
                existing_count = len(existing)
                existing_events = set(existing["event_id"].astype(str).unique())

                if new_count <= existing_count:
                    print(f"no new data ({existing_count:,} rows, {len(existing_events)} events)")
                    continue

                added_events = new_events - {str(e) for e in existing_events}
                print(f"UPDATED: {existing_count:,} -> {new_count:,} rows (+{new_count - existing_count:,})")
                if added_events:
                    print(f"    New events: {added_events}")
            except Exception:
                pass  # Can't read existing file, just overwrite

        # Write new data
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in new_rows:
                writer.writerow(row)

        print(f"  {new_count:,} rows | {len(new_events)} events | SG: {sg_present}/{new_count}")
        print(f"  Saved: {out_path}")
        time.sleep(2.0)


# ==========================================================================
# 2. SCHEDULE — always overwrite (tiny payload)
# ==========================================================================

def pull_schedule(api_key: str, seasons: list[int]):
    print("\n" + "=" * 60)
    print(f"STEP 2: Schedule/Events ({', '.join(str(s) for s in seasons)})")
    print("=" * 60)

    # One API call returns all seasons
    print("\n  Fetching event list...", end=" ", flush=True)
    url = (f"{BASE_URL}/historical-raw-data/event-list"
           f"?file_format=csv&key={api_key}")

    csv_text = fetch_csv(url)
    if csv_text is None:
        print("FAILED")
        return

    import pandas as pd
    df_all = pd.read_csv(io.StringIO(csv_text))

    year_col = None
    for col in ["calendar_year", "year", "season"]:
        if col in df_all.columns:
            year_col = col
            break

    if not year_col:
        print("WARNING: no year column found in schedule data")
        return

    for year in seasons:
        out_path = OUT_DIR / f"schedule_{year}.csv"
        df_year = df_all[df_all[year_col].astype(int) == year]
        df_year.to_csv(out_path, index=False)
        print(f"\n  {year}: {len(df_year)} events -> {out_path}")

    time.sleep(2.0)


# ==========================================================================
# 3. OUTRIGHT ODDS — smart resume (skip completed events by default)
# ==========================================================================

def pull_odds(api_key: str, seasons: list[int], full: bool = False):
    print("\n" + "=" * 60)
    print(f"STEP 3: Outright Odds ({', '.join(str(s) for s in seasons)})")
    print("=" * 60)

    # Get event list from odds system (one call returns all seasons)
    print("  Fetching odds event list...", end=" ", flush=True)
    url = f"{BASE_URL}/historical-odds/event-list?tour=pga&file_format=json&key={api_key}"
    all_events = fetch_json(url) or []
    if isinstance(all_events, dict):
        all_events = all_events.get("events", [])
    time.sleep(REQUEST_DELAY)

    seasons_set = set(seasons)
    season_events = [
        ev for ev in all_events
        if int(ev.get("calendar_year", ev.get("year", 0))) in seasons_set
    ]
    print(f"{len(season_events)} events across {seasons}")

    if not season_events:
        print("  No events found in odds system.")
        return

    for year in seasons:
        out_path = OUT_DIR / f"odds_{year}.csv"
        year_events = [
            ev for ev in season_events
            if int(ev.get("calendar_year", ev.get("year", 0))) == year
        ]

        if not year_events:
            print(f"\n  {year}: no events in odds system")
            continue

        # Determine which events to skip (completed = data is final)
        completed_ids = set()
        if not full:
            completed_ids = get_completed_events(out_path)
            if completed_ids:
                print(f"\n  {year}: {len(completed_ids)} completed events will be kept from cache")

        # Load existing data for completed events (we'll preserve these rows)
        existing_completed_rows = []
        if completed_ids and out_path.exists():
            import pandas as pd
            existing_df = pd.read_csv(out_path)
            existing_completed_rows = existing_df[
                existing_df["event_id"].astype(str).isin(completed_ids)
            ].to_dict("records")

        # Fetch incomplete/new events
        new_rows = []
        for ev in year_events:
            event_id = ev.get("event_id", ev.get("dg_event_id", ""))
            event_name = ev.get("event_name", ev.get("name", str(event_id)))
            cal_year = ev.get("calendar_year", ev.get("year", ""))

            if str(event_id) in completed_ids:
                continue

            print(f"\n  {event_name} ({cal_year}):")

            for book in BOOKS:
                for market in MARKETS:
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
                        new_rows.append(row)
                        count += 1

                    print(f"    {book}/{market}: {count} rows")
                    time.sleep(REQUEST_DELAY)

        # Merge: preserved completed rows + freshly fetched rows
        import pandas as pd
        frames = []
        if existing_completed_rows:
            frames.append(pd.DataFrame(existing_completed_rows))
        if new_rows:
            frames.append(pd.DataFrame(new_rows))

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(out_path, index=False)
            print(f"\n  {year} saved: {out_path} ({len(combined):,} rows)")
        else:
            print(f"\n  {year}: no odds data collected.")


# ==========================================================================
# 4. MATCHUP ODDS — smart resume (skip completed events by default)
# ==========================================================================

def pull_matchup_odds(api_key: str, seasons: list[int], full: bool = False):
    print("\n" + "=" * 60)
    print(f"STEP 4: Matchup Odds ({', '.join(str(s) for s in seasons)})")
    print("=" * 60)

    # Get event list (one call returns all seasons)
    print("  Fetching event list...", end=" ", flush=True)
    events = fetch_json(
        f"{BASE_URL}/historical-odds/event-list?tour=pga&file_format=json&key={api_key}"
    ) or []
    time.sleep(REQUEST_DELAY)

    seasons_set = set(seasons)
    season_events = []
    for e in events:
        for key in ["calendar_year", "year"]:
            try:
                val = int(e.get(key, 0))
                if val in seasons_set:
                    e["_year"] = val
                    season_events.append(e)
                    break
            except (ValueError, TypeError):
                continue

    print(f"{len(season_events)} events across {seasons}")

    if not season_events:
        print("  No events found in odds system.")
        return

    for year in seasons:
        out_path = OUT_DIR / f"matchup_odds_{year}.csv"
        year_events = [e for e in season_events if e["_year"] == year]

        if not year_events:
            print(f"\n  {year}: no events in odds system")
            continue

        # Determine which events to skip
        completed_ids = set()
        if not full:
            completed_ids = get_completed_events(out_path, outcome_col="p1_outcome")
            if completed_ids:
                print(f"\n  {year}: {len(completed_ids)} completed events will be kept from cache")

        # Load existing data for completed events
        existing_completed_rows = []
        if completed_ids and out_path.exists():
            import pandas as pd
            existing_df = pd.read_csv(out_path)
            existing_completed_rows = existing_df[
                existing_df["event_id"].astype(str).isin(completed_ids)
            ].to_dict("records")

        # Fetch incomplete/new events
        total_new = 0
        new_rows = []

        for i, e in enumerate(year_events):
            event_id = e.get("event_id", e.get("id", ""))
            event_name = e.get("event_name", e.get("name", ""))
            cal_year = e["_year"]

            if str(event_id) in completed_ids:
                continue

            for book in ["pinnacle"]:  # Pinnacle only for matchups
                print(f"\n  [{i+1}/{len(year_events)}] {event_name} ({cal_year}) / {book}...",
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
                    new_rows.append(row)
                    book_rows += 1
                    total_new += 1

                print(f"{book_rows} matchups")
                time.sleep(REQUEST_DELAY)

        # Merge: preserved completed rows + freshly fetched rows
        import pandas as pd
        frames = []
        if existing_completed_rows:
            frames.append(pd.DataFrame(existing_completed_rows))
        if new_rows:
            frames.append(pd.DataFrame(new_rows))

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            # Ensure column order matches expected fieldnames
            for col in MATCHUP_FIELDNAMES:
                if col not in combined.columns:
                    combined[col] = ""
            combined.to_csv(out_path, index=False)
            print(f"\n  {year} saved: {out_path} ({len(combined):,} rows, {total_new} new)")
        else:
            print(f"\n  {year}: no matchup data collected.")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull PGA season data from DataGolf API (smart incremental by default)"
    )
    parser.add_argument("--key", required=True, help="DataGolf API key")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS,
                        help=f"Seasons to pull (default: {DEFAULT_SEASONS})")
    parser.add_argument("--only", choices=["rounds", "schedule", "odds", "matchups"],
                        help="Pull only one data type")
    parser.add_argument("--full", action="store_true",
                        help="Force full re-download (ignore smart resume)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Script 14 — Pull PGA Season Data ({', '.join(str(s) for s in args.seasons)})")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode:    {'FULL (re-download all)' if args.full else 'SMART (skip completed events)'}")
    print(f"Output:  {OUT_DIR}")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.only is None or args.only == "rounds":
        pull_sg_rounds(args.key, args.seasons)
    if args.only is None or args.only == "schedule":
        pull_schedule(args.key, args.seasons)
    if args.only is None or args.only == "odds":
        pull_odds(args.key, args.seasons, full=args.full)
    if args.only is None or args.only == "matchups":
        pull_matchup_odds(args.key, args.seasons, full=args.full)

    print(f"\n{'=' * 60}")
    print("Done! Files saved to:", OUT_DIR)
    print(f"{'=' * 60}")
