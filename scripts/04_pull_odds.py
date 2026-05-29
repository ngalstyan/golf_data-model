"""
Script 04 — Pull Historical Outright Odds 2019–2022 (FIXED)
Endpoint: /historical-odds/outrights?tour=pga&event_id=all&year=YYYY&market=M&book=B
Output: data/raw/odds_2019_2022.csv

CONFIRMED from diagnostics:
- Response is a DICT keyed by event_id (not a list):
  { "14": { book, event_completed, event_id, event_name, market,
            odds: [ {bet_outcome_numeric, bet_outcome_text, close_odds,
                     close_time, dg_id, open_odds, open_time, outcome,
                     player_name}, ... ],
            season, year },
    "23": { ... }, ... }
- CRITICAL: odds can be a STRING when no data exists:
  "we did not track any bets from pinnacle at the Sony Open in Hawaii"
- Must check isinstance(event["odds"], list) before iterating

Run: python 04_pull_odds.py --key YOUR_API_KEY
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
MARKETS  = ["win", "top_5", "make_cut"]

FIELDNAMES = [
    "event_id", "event_name", "event_completed", "season", "calendar_year",
    "bookmaker", "market",
    "dg_id", "player_name", "outcome", "bet_outcome_numeric", "bet_outcome_text",
    "open_odds", "open_time", "close_odds", "close_time",
    "implied_prob_open", "implied_prob_close",
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
        print(f"HTTP {e.code}: {e.reason}" + (f" — {body}" if body else ""))
        return None
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        return None


def implied_prob(odds_decimal):
    try:
        od = float(odds_decimal)
        return round(1.0 / od, 6) if od > 0 else ""
    except (ValueError, TypeError):
        return ""


def main(api_key: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 04 — Historical Odds 2019–2022 [FIXED]")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Books: {BOOKS}  Markets: {MARKETS}")
    print("=" * 60)

    out_path = os.path.join(OUT_DIR, "odds_2019_2022.csv")
    log_lines = []
    total_rows = 0
    no_data_events = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for year in SEASONS:
            print(f"\n── {year} ─────────────────────────────────────────")

            for book in BOOKS:
                for market in MARKETS:
                    label = f"{year}/{book}/{market}"
                    print(f"  {label}...", end=" ", flush=True)

                    url = (f"{BASE_URL}/historical-odds/outrights"
                           f"?tour=pga&event_id=all&year={year}"
                           f"&market={market}&book={book}"
                           f"&odds_format=decimal&file_format=json"
                           f"&key={api_key}")

                    data = fetch_json(url)
                    if data is None:
                        print("FAILED")
                        log_lines.append(f"FAILED: {label}")
                        time.sleep(1.0)
                        continue

                    # Response is dict keyed by event_id
                    if not isinstance(data, dict):
                        print(f"unexpected type: {type(data).__name__}")
                        continue

                    combo_rows = 0
                    combo_skip = 0

                    for eid, event in data.items():
                        if not isinstance(event, dict):
                            continue

                        event_name = event.get("event_name", "")
                        event_completed = event.get("event_completed", "")
                        season = event.get("season", "")
                        cal_year = event.get("year", year)
                        odds_data = event.get("odds", [])

                        # CRITICAL: odds can be a string when no data
                        if not isinstance(odds_data, list):
                            combo_skip += 1
                            continue

                        for p in odds_data:
                            if not isinstance(p, dict):
                                continue

                            row = {
                                "event_id": eid,
                                "event_name": event_name,
                                "event_completed": event_completed,
                                "season": season,
                                "calendar_year": cal_year,
                                "bookmaker": book,
                                "market": market,
                                "dg_id": p.get("dg_id", ""),
                                "player_name": p.get("player_name", ""),
                                "outcome": p.get("outcome", ""),
                                "bet_outcome_numeric": p.get("bet_outcome_numeric", ""),
                                "bet_outcome_text": p.get("bet_outcome_text", ""),
                                "open_odds": p.get("open_odds", ""),
                                "open_time": p.get("open_time", ""),
                                "close_odds": p.get("close_odds", ""),
                                "close_time": p.get("close_time", ""),
                                "implied_prob_open": implied_prob(p.get("open_odds")),
                                "implied_prob_close": implied_prob(p.get("close_odds")),
                            }
                            writer.writerow(row)
                            combo_rows += 1
                            total_rows += 1

                    no_data_events += combo_skip
                    print(f"{combo_rows:,} rows ({combo_skip} events skipped — no data)")
                    log_lines.append(f"{label}: {combo_rows:,} rows, {combo_skip} skipped")
                    time.sleep(1.0)

    # ── Log ────────────────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "04_odds.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\nTotal rows: {total_rows}\n")
        f.write(f"Events with no data: {no_data_events}\n")
        for line in log_lines: f.write(f"  {line}\n")

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Total rows          : {total_rows:,}")
    print(f"  Events with no data : {no_data_events}")
    print(f"  Output              : {out_path}")
    print(f"\n  ✓ Script 04 complete. Next: run 06_pull_skill_ratings.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    main(args.key)
