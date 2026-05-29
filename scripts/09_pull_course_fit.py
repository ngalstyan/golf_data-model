"""
Script 09 — Course Fit Archive 2019–2022 (FIXED)
Endpoint: /preds/pre-tournament-archive
Output: data/raw/course_fit_archive_2019_2022.csv

CONFIRMED from diagnostics:
- Pre-tournament-archive returns TWO models: "baseline" and "baseline_history_fit"
- Course fit = difference between baseline_history_fit and baseline probabilities
- Same per-event iteration as Script 07, but we compute the fit delta

Each row: one player × one event with both model predictions and their difference.

Run: python 09_pull_course_fit.py --key YOUR_API_KEY
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

PROB_COLS = ["win", "top_3", "top_5", "top_10", "top_20", "top_30",
             "make_cut", "first_round_leader"]

FIELDNAMES = (
    ["event_id", "event_name", "event_completed", "calendar_year",
     "dg_id", "player_name", "fin_text"]
    + [f"baseline_{c}" for c in PROB_COLS]
    + [f"history_fit_{c}" for c in PROB_COLS]
    + [f"fit_delta_{c}" for c in PROB_COLS]
)


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
    url = (f"{BASE_URL}/historical-odds/event-list"
           f"?tour=pga&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data is None:
        return []
    return data if isinstance(data, list) else []


def safe_float(v):
    try: return float(v)
    except (ValueError, TypeError): return None


def main(api_key: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 09 — Course Fit Archive 2019–2022 [FIXED]")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n  Fetching event list...")
    events = get_odds_event_list(api_key)
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

    out_path = os.path.join(OUT_DIR, "course_fit_archive_2019_2022.csv")
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

            baseline_list = data.get("baseline", [])
            fit_list = data.get("baseline_history_fit", [])

            if not isinstance(baseline_list, list) or not isinstance(fit_list, list):
                print("no model data")
                time.sleep(0.5)
                continue

            # Index fit predictions by dg_id for fast lookup
            fit_by_id = {}
            for p in fit_list:
                if isinstance(p, dict):
                    fit_by_id[p.get("dg_id")] = p

            event_completed = data.get("event_completed", "")
            event_rows = 0

            for bp in baseline_list:
                if not isinstance(bp, dict):
                    continue

                dg_id = bp.get("dg_id")
                fp = fit_by_id.get(dg_id, {})

                row = {
                    "event_id": event_id,
                    "event_name": event_name,
                    "event_completed": event_completed,
                    "calendar_year": cal_year,
                    "dg_id": dg_id,
                    "player_name": bp.get("player_name", ""),
                    "fin_text": bp.get("fin_text", ""),
                }

                for col in PROB_COLS:
                    bv = safe_float(bp.get(col))
                    fv = safe_float(fp.get(col))
                    row[f"baseline_{col}"] = bp.get(col, "")
                    row[f"history_fit_{col}"] = fp.get(col, "")
                    if bv is not None and fv is not None and bv > 0 and fv > 0:
                        # Convert decimal odds to probability, compute delta
                        bp_prob = 1.0 / bv
                        fp_prob = 1.0 / fv
                        row[f"fit_delta_{col}"] = round(fp_prob - bp_prob, 6)
                    else:
                        row[f"fit_delta_{col}"] = ""

                writer.writerow(row)
                event_rows += 1
                total_rows += 1

            print(f"{event_rows} players ({len(fit_by_id)} with fit data)")
            log_lines.append(f"{event_name} ({cal_year}): {event_rows}")
            time.sleep(0.7)

    log_path = os.path.join(LOG_DIR, "09_course_fit.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\nTotal: {total_rows}\n")
        for line in log_lines: f.write(f"  {line}\n")

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Total rows : {total_rows:,}")
    print(f"  Output     : {out_path}")
    print(f"\n  fit_delta_win = P(win|course_fit) - P(win|baseline)")
    print(f"  Positive delta = course fit HELPS this player")
    print(f"\n  ✓ Script 09 complete. Next: run 10_pull_kft_euro_sg.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    main(args.key)
