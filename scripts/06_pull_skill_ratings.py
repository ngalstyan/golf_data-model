"""
Script 06 — Pull DataGolf Skill Ratings Archive
Endpoint: /preds/get-dg-rankings (current) + /preds/pre-tournament-archive
          to extract implied skill ratings historically
Output: data/raw/skill_ratings_current.csv

DataGolf publishes current DG rankings with skill estimates.
This gives us warm-start priors for player ability.

Run: python 06_pull_skill_ratings.py --key YOUR_API_KEY
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "https://feeds.datagolf.com"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")

# Column map for DG rankings response
# Fields: dg_id, player_name, country, owgr_rank, dg_skill_estimate,
#         sg_putt, sg_arg, sg_app, sg_ott, driving_dist, driving_acc
COLUMN_MAP = {
    "dg_id":             "player_id",
    "player_name":       "player_name",
    "country":           "country",
    "owgr_rank":         "owgr_rank",
    "dg_skill_estimate": "dg_skill_estimate",
    "sg_putt":           "sg_putt_rating",
    "sg_arg":            "sg_arg_rating",
    "sg_app":            "sg_app_rating",
    "sg_ott":            "sg_ott_rating",
    "driving_dist":      "driving_dist",
    "driving_acc":       "driving_acc",
}


def fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("  → Invalid API key.")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"  Network error: {e.reason}")
        sys.exit(1)


def main(api_key: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 06 — DataGolf Skill Ratings")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Current DG rankings (skill estimates for all active players) ──────────
    print("\nFetching current DG rankings...")
    url = f"{BASE_URL}/preds/get-dg-rankings?file_format=json&key={api_key}"
    data = fetch_json(url)

    # Response: {"last_updated": "...", "rankings": [...]}
    rankings = data.get("rankings", data) if isinstance(data, dict) else data
    print(f"  Players received: {len(rankings)}")

    rows = []
    for p in rankings:
        row = {}
        for dg_field, our_field in COLUMN_MAP.items():
            row[our_field] = p.get(dg_field, "")
        rows.append(row)

    # Validate
    warnings = []
    for i, r in enumerate(rows):
        if not r.get("player_id"):
            warnings.append(f"Row {i}: missing player_id")
        skill = r.get("dg_skill_estimate")
        if skill not in ("", None):
            try:
                fv = float(skill)
                if not (-5 <= fv <= 5):
                    warnings.append(f"Row {i} ({r.get('player_name')}): "
                                    f"skill_estimate={fv} outside [-5,5]")
            except (ValueError, TypeError):
                warnings.append(f"Row {i}: non-numeric skill_estimate '{skill}'")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"    ! {w}")

    # Write
    out_path = os.path.join(OUT_DIR, "skill_ratings_current.csv")
    fieldnames = list(COLUMN_MAP.values())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written: {out_path}  ({len(rows)} rows)")

    # Log
    log_path = os.path.join(LOG_DIR, "06_skill_ratings.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\n")
        f.write(f"Rows: {len(rows)}\n")
        f.write(f"Warnings: {len(warnings)}\n")
        for w in warnings:
            f.write(f"  {w}\n")

    # Summary
    skill_vals = []
    for r in rows:
        v = r.get("dg_skill_estimate")
        if v not in ("", None):
            try:
                skill_vals.append(float(v))
            except (ValueError, TypeError):
                pass

    if skill_vals:
        skill_vals.sort(reverse=True)
        print("\n── Skill estimate distribution ──────────────────────────")
        print(f"  Players with estimates : {len(skill_vals)}")
        print(f"  Max (best player)      : {skill_vals[0]:.3f}")
        print(f"  Median                 : {skill_vals[len(skill_vals)//2]:.3f}")
        print(f"  Min                    : {skill_vals[-1]:.3f}")
        print(f"  Top 10 range           : {skill_vals[9]:.3f} – {skill_vals[0]:.3f}")

    print(f"\n  ✓ Script 06 complete. Next: run 07_pull_predictions_archive.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull DataGolf skill ratings")
    parser.add_argument("--key", required=True, help="DataGolf API key")
    args = parser.parse_args()
    main(args.key)
