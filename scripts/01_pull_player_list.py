"""
Script 01 — Pull Player List
Table: player_profile (Table 5 seed)
Endpoint: /get-player-list
Output: data/raw/player_list.csv

Run: python 01_pull_player_list.py --key YOUR_API_KEY
"""

import argparse
import csv
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "https://feeds.datagolf.com"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")

# ── Columns we keep (maps DataGolf field → our column name) ──────────────────
COLUMN_MAP = {
    "dg_id":        "player_id",
    "player_name":  "player_name",
    "country":      "country",
    "amateur":      "is_amateur",
}


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("  → Invalid API key. Check your Scratch PLUS subscription.")
        elif e.code == 403:
            print("  → Endpoint requires Scratch PLUS (not BASIC).")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"  Network error: {e.reason}")
        sys.exit(1)


def validate(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Basic validation — returns (clean_rows, warnings)."""
    warnings = []
    clean = []
    seen_ids = set()

    for i, r in enumerate(rows):
        pid = r.get("player_id")

        # Must have ID and name
        if not pid:
            warnings.append(f"Row {i}: missing player_id — skipped")
            continue
        if not r.get("player_name"):
            warnings.append(f"Row {i} (id={pid}): missing player_name — skipped")
            continue

        # Duplicate ID check
        if pid in seen_ids:
            warnings.append(f"Row {i} (id={pid}): duplicate player_id — skipped")
            continue
        seen_ids.add(pid)

        # is_amateur should be 0 or 1
        if r.get("is_amateur") not in (0, 1, "0", "1", True, False):
            warnings.append(f"Row {i} (id={pid}): unexpected is_amateur value '{r.get('is_amateur')}' — kept")

        clean.append(r)

    return clean, warnings


def main(api_key: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 01 — Player List")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Fetch ─────────────────────────────────────────────────────────────
    url = f"{BASE_URL}/get-player-list?file_format=json&key={api_key}"
    print(f"\nFetching: {BASE_URL}/get-player-list")
    data = fetch_json(url)

    # DataGolf returns a list directly
    raw_players = data if isinstance(data, list) else data.get("players", [])
    print(f"  Raw records received: {len(raw_players)}")

    # ── 2. Remap columns ─────────────────────────────────────────────────────
    rows = []
    for p in raw_players:
        row = {}
        for dg_field, our_field in COLUMN_MAP.items():
            row[our_field] = p.get(dg_field, "")
        rows.append(row)

    # ── 3. Validate ──────────────────────────────────────────────────────────
    clean_rows, warnings = validate(rows)

    if warnings:
        print(f"\n  Validation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    ! {w}")

    print(f"\n  Clean rows: {len(clean_rows)}")

    # ── 4. Write CSV ─────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "player_list.csv")
    fieldnames = list(COLUMN_MAP.values())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)

    print(f"\n  Written: {out_path}")
    print(f"  Rows: {len(clean_rows)}")

    # ── 5. Write log ─────────────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "01_player_list.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\n")
        f.write(f"Raw records: {len(raw_players)}\n")
        f.write(f"Clean rows: {len(clean_rows)}\n")
        f.write(f"Warnings: {len(warnings)}\n")
        for w in warnings:
            f.write(f"  {w}\n")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    amateurs = sum(1 for r in clean_rows if str(r.get("is_amateur")) in ("1", "True", "true"))
    countries = len(set(r["country"] for r in clean_rows if r["country"]))

    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  Total players : {len(clean_rows)}")
    print(f"  Professionals : {len(clean_rows) - amateurs}")
    print(f"  Amateurs      : {amateurs}")
    print(f"  Countries     : {countries}")
    print(f"\n  ✓ Script 01 complete. Next: run 02_pull_schedule.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull DataGolf player list")
    parser.add_argument("--key", required=True, help="DataGolf API key")
    args = parser.parse_args()
    main(args.key)
