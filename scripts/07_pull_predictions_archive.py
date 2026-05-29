"""
Script 07 — Pre-Tournament Predictions Archive 2019–2022 (FIXED)
Endpoint: /preds/pre-tournament-archive?event_id=EID&year=YYYY
Output: data/raw/predictions_archive_2019_2022.csv

CONFIRMED from diagnostics:
- Response structure:
  { baseline: [ {dg_id, player_name, fin_text, win, top_5, top_10, top_20,
                 top_30, make_cut, first_round_leader, top_3}, ... ],
    baseline_history_fit: [ same structure ],
    event_completed, event_id, event_name,
    models_available: ["baseline", "baseline_history_fit"] }
- Values are DECIMAL ODDS (not probabilities): win=9.57 means 9.57 to 1
- Must iterate per-event (no event_id=all support for this endpoint)
- Event IDs come from historical-odds/event-list

Run: python 07_pull_predictions_archive.py --key YOUR_API_KEY
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

FIELDNAMES = [
    "event_id", "event_name", "event_completed", "calendar_year",
    "model", "dg_id", "player_name", "fin_text",
    "win", "top_3", "top_5", "top_10", "top_20", "top_30",
    "make_cut", "first_round_leader",
]


def fetch_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
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
    """Use historical-odds/event-list to get event IDs."""
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
    print("Script 07 — Pre-Tournament Predictions Archive [FIXED]")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Get event list
    print("\n  Fetching event list from historical-odds/event-list...")
    events = get_odds_event_list(api_key)
    print(f"  Total events: {len(events)}")

    # Filter to 2019-2022
    events_in_window = []
    for e in events:
        cal_year = None
        for key in ["calendar_year", "year"]:
            try:
                val = int(e.get(key, 0))
                if 2019 <= val <= 2022:
                    cal_year = val
                    break
            except (ValueError, TypeError):
                continue
        if cal_year:
            e["_year"] = cal_year
            events_in_window.append(e)

    print(f"  Events in 2019–2022: {len(events_in_window)}")

    out_path = os.path.join(OUT_DIR, "predictions_archive_2019_2022.csv")
    log_lines = []
    total_rows = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for e in events_in_window:
            event_id = e.get("event_id", e.get("id", ""))
            event_name = e.get("event_name", e.get("name", ""))
            cal_year = e["_year"]

            print(f"  {event_name} ({cal_year})...", end=" ", flush=True)

            url = (f"{BASE_URL}/preds/pre-tournament-archive"
                   f"?event_id={event_id}&year={cal_year}"
                   f"&odds_format=decimal&file_format=json&key={api_key}")

            data = fetch_json(url)
            if data is None or not isinstance(data, dict):
                print("SKIP")
                time.sleep(0.5)
                continue

            event_completed = data.get("event_completed", "")
            models = data.get("models_available", ["baseline", "baseline_history_fit"])
            event_rows = 0

            for model_name in models:
                players = data.get(model_name, [])
                if not isinstance(players, list):
                    continue

                for p in players:
                    if not isinstance(p, dict):
                        continue

                    row = {
                        "event_id": event_id,
                        "event_name": event_name,
                        "event_completed": event_completed,
                        "calendar_year": cal_year,
                        "model": model_name,
                        "dg_id": p.get("dg_id", ""),
                        "player_name": p.get("player_name", ""),
                        "fin_text": p.get("fin_text", ""),
                        "win": p.get("win", ""),
                        "top_3": p.get("top_3", ""),
                        "top_5": p.get("top_5", ""),
                        "top_10": p.get("top_10", ""),
                        "top_20": p.get("top_20", ""),
                        "top_30": p.get("top_30", ""),
                        "make_cut": p.get("make_cut", ""),
                        "first_round_leader": p.get("first_round_leader", ""),
                    }
                    writer.writerow(row)
                    event_rows += 1
                    total_rows += 1

            print(f"{event_rows} rows ({len(models)} models)")
            log_lines.append(f"{event_name} ({cal_year}): {event_rows} rows")
            time.sleep(0.7)

    log_path = os.path.join(LOG_DIR, "07_predictions.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\nTotal rows: {total_rows}\n")
        for line in log_lines: f.write(f"  {line}\n")

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Total rows : {total_rows:,}")
    print(f"  Output     : {out_path}")
    print(f"\n  NOTE: Values are decimal odds (win=9.57 → prob=1/9.57=0.1045)")
    print(f"\n  ✓ Script 07 complete. Next: run 08_pull_matchup_odds.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    main(args.key)
