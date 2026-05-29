# pull_additional_seasons.py
import requests
import pandas as pd
import time
import argparse
from pathlib import Path
from io import StringIO

DATA_DIR = Path("/Users/galstyann/Documents/golf_data-model/golf_model/data/raw")
BASE_URL = "https://feeds.datagolf.com"


def pull_sg_rounds(api_key, seasons):
    """Pull strokes-gained round data for additional seasons."""
    all_rows = []

    for season in seasons:
        print(f"Pulling SG rounds for {season}...")
        url = f"{BASE_URL}/historical-raw-data/rounds"
        params = {
            "tour": "pga",
            "year": season,          # ← must be "year", not "season"
            "event_id": "all",
            "file_format": "csv",
            "key": api_key,
        }
        r = requests.get(url, params=params)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            df["_season"] = season
            all_rows.append(df)
            print(f"  ✓ {len(df):,} rows")
        else:
            print(f"  ✗ Failed: {r.status_code} — {r.text[:200]}")
        time.sleep(1)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_path = DATA_DIR / "sg_rounds_pga_2023_2024.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(combined):,} rows)")
        return combined
    return pd.DataFrame()


def pull_schedule(api_key, seasons):
    """Pull schedule/events for additional seasons."""
    all_rows = []

    for season in seasons:
        print(f"Pulling schedule for {season}...")
        url = f"{BASE_URL}/historical-raw-data/event-list"
        params = {
            "season": season,
            "file_format": "csv",
            "key": api_key,
        }
        r = requests.get(url, params=params)
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            all_rows.append(df)
            print(f"  ✓ {len(df):,} rows")
        else:
            print(f"  ✗ Failed: {r.status_code} — {r.text[:200]}")
        time.sleep(1)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_path = DATA_DIR / "schedule_2023_2024.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(combined):,} rows)")
        return combined
    return pd.DataFrame()


def pull_odds(api_key, seasons):
    """Pull historical odds — exact same approach as 04_pull_odds.py."""
    import json, urllib.request

    BOOKS   = ["pinnacle", "draftkings"]
    MARKETS = ["win", "top_5", "make_cut"]

    # Step 1: get event list from the odds system (not the schedule CSV)
    url = f"{BASE_URL}/historical-odds/event-list?tour=pga&file_format=json&key={api_key}"
    resp = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"}), timeout=30)
    all_events = json.loads(resp.read().decode("utf-8"))
    if isinstance(all_events, dict):
        all_events = all_events.get("events", [])

    # Step 2: filter to requested seasons
    events_in_window = [
        ev for ev in all_events
        if int(ev.get("calendar_year", ev.get("year", 0))) in seasons
    ]
    print(f"  Total events in odds system: {len(all_events)}")
    print(f"  Events in {seasons}: {len(events_in_window)}")

    all_rows = []

    for ev in events_in_window:
        event_id   = ev.get("event_id", ev.get("dg_event_id", ""))
        event_name = ev.get("event_name", ev.get("name", str(event_id)))
        cal_year   = ev.get("calendar_year", ev.get("year", ""))

        print(f"\n  {event_name} ({cal_year}):")

        for book in BOOKS:
            for market in MARKETS:
                req_url = (
                    f"{BASE_URL}/historical-odds/outrights"
                    f"?tour=pga&event_id={event_id}&year={cal_year}"
                    f"&market={market}&book={book}"
                    f"&odds_format=decimal&file_format=json&key={api_key}"
                )
                r = requests.get(req_url)
                if r.status_code != 200:
                    continue

                data = r.json()
                if isinstance(data, str):
                    continue

                if isinstance(data, dict):
                    players = data.get("odds", data.get("players", []))
                elif isinstance(data, list):
                    players = data
                else:
                    continue

                for row in players:
                    if not isinstance(row, dict):
                        continue
                    row["event_id"]   = event_id
                    row["event_name"] = event_name
                    row["calendar_year"] = cal_year
                    row["bookmaker"]  = book
                    row["market"]     = market
                    all_rows.append(row)

                print(f"    {book}/{market}: {len([r for r in all_rows if r.get('event_id')==event_id and r.get('bookmaker')==book and r.get('market')==market])} rows")
                time.sleep(0.3)

    if all_rows:
        combined = pd.DataFrame(all_rows)
        out_path = DATA_DIR / "odds_2023_2024.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(combined):,} rows)")
        return combined

    print("  ✗ No rows collected")
    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="DataGolf API key")
    args = parser.parse_args()

    SEASONS = [2023, 2024]

    print("=" * 50)
    print("Pulling 2023-2024 data from DataGolf")
    print("=" * 50)

    pull_sg_rounds(args.key, SEASONS)
    pull_schedule(args.key, SEASONS)
    pull_odds(args.key, SEASONS)

    print("\n✓ Done. Run 05_validate_all.py to verify.")