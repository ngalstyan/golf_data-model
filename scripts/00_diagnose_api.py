"""
Script 00 — API Response Diagnostics
Run this FIRST before anything else.

It makes ONE call to each endpoint type and dumps the raw JSON
structure so we can see exact field names, nesting, and formats.

Run: python 00_diagnose_api.py --key YOUR_API_KEY
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "https://feeds.datagolf.com"
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "diagnostics")


def fetch_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:500]
        except:
            pass
        print(f"  HTTP {e.code}: {e.reason}")
        if body:
            print(f"  Response body: {body}")
        return None
    except urllib.error.URLError as e:
        print(f"  Network error: {e.reason}")
        return None


def describe_structure(data, label="", depth=0, max_items=2):
    """Recursively describe JSON structure."""
    indent = "  " * depth
    if isinstance(data, list):
        print(f"{indent}{label}LIST of {len(data)} items")
        if len(data) > 0:
            for i, item in enumerate(data[:max_items]):
                describe_structure(item, f"[{i}] ", depth + 1, max_items)
            if len(data) > max_items:
                print(f"{indent}  ... and {len(data) - max_items} more")
    elif isinstance(data, dict):
        print(f"{indent}{label}DICT with {len(data)} keys:")
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                describe_structure(v, f"{k}: ", depth + 1, max_items)
            else:
                vtype = type(v).__name__
                vstr = str(v)[:80]
                print(f"{indent}  {k}: {vtype} = {vstr}")
    else:
        vtype = type(data).__name__
        vstr = str(data)[:80]
        print(f"{indent}{label}{vtype} = {vstr}")


def dump_json(data, filename):
    """Save raw JSON to diagnostics dir."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved to: {path}")


def main(api_key: str):
    print("=" * 70)
    print("Script 00 — API Response Diagnostics")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Historical Raw Data — Event List ────────────────────────────
    print("\n" + "─" * 70)
    print("  1. historical-raw-data/event-list (PGA)")
    print("─" * 70)
    url = (f"{BASE_URL}/historical-raw-data/event-list"
           f"?tour=pga&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        describe_structure(data, "event-list: ")
        dump_json(data, "01_raw_event_list.json")

    # ── 2. Historical Raw Data — Rounds (single event) ─────────────────
    print("\n" + "─" * 70)
    print("  2. historical-raw-data/rounds (Masters 2020, event_id=14)")
    print("─" * 70)
    url = (f"{BASE_URL}/historical-raw-data/rounds"
           f"?tour=pga&event_id=14&year=2020"
           f"&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        describe_structure(data, "rounds: ")
        dump_json(data, "02_rounds_masters_2020.json")

    # ── 3. Historical Raw Data — Rounds with event_id=all ──────────────
    print("\n" + "─" * 70)
    print("  3. historical-raw-data/rounds (event_id=all, year=2020)")
    print("     NOTE: This may be large — fetching first 5 seconds only")
    print("─" * 70)
    # Just try the endpoint to see if event_id=all returns flat round data
    url = (f"{BASE_URL}/historical-raw-data/rounds"
           f"?tour=pga&event_id=all&year=2020"
           f"&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        if isinstance(data, list):
            print(f"  Returned list of {len(data)} items")
            describe_structure(data[:3], "rounds-all: ", max_items=3)
            # Save just first 10 rows for inspection
            dump_json(data[:10], "03_rounds_all_2020_sample.json")
        else:
            describe_structure(data, "rounds-all: ")
            dump_json(data, "03_rounds_all_2020_sample.json")

    # ── 4. Historical Odds — Event List ────────────────────────────────
    print("\n" + "─" * 70)
    print("  4. historical-odds/event-list (PGA)")
    print("─" * 70)
    url = (f"{BASE_URL}/historical-odds/event-list"
           f"?tour=pga&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        describe_structure(data, "odds-event-list: ")
        dump_json(data, "04_odds_event_list.json")

    # ── 5. Historical Odds — Outrights (single event) ──────────────────
    print("\n" + "─" * 70)
    print("  5. historical-odds/outrights (Masters 2020, Pinnacle, win)")
    print("─" * 70)
    url = (f"{BASE_URL}/historical-odds/outrights"
           f"?tour=pga&event_id=14&year=2020&market=win&book=pinnacle"
           f"&odds_format=decimal&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        describe_structure(data, "outrights: ")
        dump_json(data, "05_outrights_masters_2020.json")

    # ── 5b. Historical Odds — Outrights (event_id=all) ─────────────────
    print("\n" + "─" * 70)
    print("  5b. historical-odds/outrights (event_id=all, year=2020, Pinnacle, win)")
    print("─" * 70)
    url = (f"{BASE_URL}/historical-odds/outrights"
           f"?tour=pga&event_id=all&year=2020&market=win&book=pinnacle"
           f"&odds_format=decimal&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        if isinstance(data, list):
            print(f"  Returned list of {len(data)} items")
            describe_structure(data[:3], "outrights-all: ", max_items=2)
            dump_json(data[:10], "05b_outrights_all_2020_sample.json")
        else:
            describe_structure(data, "outrights-all: ")
            dump_json(data, "05b_outrights_all_2020_sample.json")

    # ── 6. Historical Odds — Matchups (single event) ───────────────────
    print("\n" + "─" * 70)
    print("  6. historical-odds/matchups (Masters 2020, Pinnacle)")
    print("─" * 70)
    url = (f"{BASE_URL}/historical-odds/matchups"
           f"?tour=pga&event_id=14&year=2020&book=pinnacle"
           f"&odds_format=decimal&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        describe_structure(data, "matchups: ")
        dump_json(data, "06_matchups_masters_2020.json")

    # ── 7. Predictions Archive (Masters 2020) ──────────────────────────
    print("\n" + "─" * 70)
    print("  7. preds/pre-tournament-archive (Masters 2020)")
    print("─" * 70)
    url = (f"{BASE_URL}/preds/pre-tournament-archive"
           f"?event_id=14&year=2020"
           f"&odds_format=decimal&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        describe_structure(data, "predictions: ")
        dump_json(data, "07_predictions_masters_2020.json")

    # ── 7b. Try predictions with no event_id (latest) ──────────────────
    print("\n" + "─" * 70)
    print("  7b. preds/pre-tournament-archive (no event_id, latest)")
    print("─" * 70)
    url = (f"{BASE_URL}/preds/pre-tournament-archive"
           f"?odds_format=decimal&file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        describe_structure(data, "predictions-latest: ")
        dump_json(data, "07b_predictions_latest.json")

    # ── 8. DG Rankings ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  8. preds/get-dg-rankings")
    print("─" * 70)
    url = (f"{BASE_URL}/preds/get-dg-rankings"
           f"?file_format=json&key={api_key}")
    data = fetch_json(url)
    if data:
        describe_structure(data, "rankings: ")
        dump_json(data, "08_dg_rankings.json")

    # ── 9. Historical Raw Data — Rounds CSV format ─────────────────────
    print("\n" + "─" * 70)
    print("  9. historical-raw-data/rounds CSV format (Masters 2020)")
    print("      This reveals the flat CSV column names")
    print("─" * 70)
    url = (f"{BASE_URL}/historical-raw-data/rounds"
           f"?tour=pga&event_id=14&year=2020"
           f"&file_format=csv&key={api_key}")
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            csv_text = resp.read().decode("utf-8")
            lines = csv_text.strip().split("\n")
            print(f"  CSV columns: {lines[0]}")
            print(f"  Total rows (including header): {len(lines)}")
            if len(lines) > 1:
                print(f"  First data row: {lines[1][:200]}")
            if len(lines) > 2:
                print(f"  Second data row: {lines[2][:200]}")
            # Save CSV sample
            sample_path = os.path.join(OUT_DIR, "09_rounds_csv_sample.csv")
            with open(sample_path, "w") as f:
                f.write("\n".join(lines[:20]))
            print(f"  Saved sample to: {sample_path}")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}")

    print("\n" + "=" * 70)
    print("  ✓ Diagnostics complete.")
    print(f"  Check {OUT_DIR} for raw JSON/CSV dumps.")
    print("  Share the terminal output with Claude to get fixed scripts.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose DataGolf API response structures")
    parser.add_argument("--key", required=True, help="DataGolf API key")
    args = parser.parse_args()
    main(args.key)
