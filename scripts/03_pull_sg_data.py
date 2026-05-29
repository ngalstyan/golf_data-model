"""
Script 03 — Pull Historical Round-Level SG Data 2019–2022 (FIXED)
Endpoint: /historical-raw-data/rounds?tour=pga&event_id=all&year=YYYY&file_format=csv
Output: data/raw/sg_rounds_2019_2022.csv

CONFIRMED from diagnostics:
- CSV returns flat rows: tour,year,season,event_completed,event_name,event_id,
  player_name,dg_id,fin_text,round_num,course_name,course_num,course_par,
  start_hole,teetime,round_score,sg_putt,sg_arg,sg_app,sg_ott,sg_t2g,
  sg_total,driving_dist,driving_acc,gir,scrambling,prox_rgh,prox_fw,
  great_shots,poor_shots,eagles_or_better,birdies,pars,bogies,doubles_or_worse
- Masters 2020 = 303 rows (correct: ~92 players × ~3.3 avg rounds)
- SG components blank at non-ShotLink events (Augusta, Open Championship, etc.)
- round_num values: {1, 2, 3, 4}

Run: python 03_pull_sg_data.py --key YOUR_API_KEY
"""

import argparse
import csv
import io
import os
import time
import urllib.request
import urllib.error
from datetime import datetime

BASE_URL = "https://feeds.datagolf.com"
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
SEASONS  = [2019, 2020, 2021, 2022]


def fetch_csv(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "golf-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = ""
        try: body = e.read().decode("utf-8")[:300]
        except: pass
        print(f"\n  HTTP {e.code}: {e.reason}" + (f" — {body}" if body else ""))
        return None
    except urllib.error.URLError as e:
        print(f"\n  Network error: {e.reason}")
        return None


def main(api_key: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("Script 03 — Historical SG Rounds (PGA, 2019–2022) [FIXED]")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    out_path = os.path.join(OUT_DIR, "sg_rounds_2019_2022.csv")
    log_lines = []
    total_rows = 0
    fieldnames = None
    csv_file = None
    writer = None

    for year in SEASONS:
        print(f"\n── {year} ─────────────────────────────────────────")
        url = (f"{BASE_URL}/historical-raw-data/rounds"
               f"?tour=pga&event_id=all&year={year}"
               f"&file_format=csv&key={api_key}")

        print(f"  Fetching all PGA rounds for {year}...", end=" ", flush=True)
        csv_text = fetch_csv(url)

        if csv_text is None:
            log_lines.append(f"FAILED: {year}")
            continue

        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        if not rows:
            print("0 rows")
            log_lines.append(f"{year}: 0 rows")
            continue

        if fieldnames is None:
            fieldnames = reader.fieldnames
            csv_file = open(out_path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            print(f"\n  Columns ({len(fieldnames)}): {','.join(fieldnames)}")
            print(f"  {year}:", end=" ", flush=True)

        for row in rows:
            writer.writerow(row)
            total_rows += 1

        events = set(r.get("event_id", "") for r in rows)
        rounds = sorted(set(r.get("round_num", "") for r in rows))
        sg_present = sum(1 for r in rows if r.get("sg_ott", ""))
        print(f"{len(rows):,} rows | {len(events)} events | "
              f"rounds: {rounds} | SG: {sg_present}/{len(rows)}")
        log_lines.append(f"{year}: {len(rows):,} rows, {len(events)} events")
        time.sleep(2.0)

    if csv_file:
        csv_file.close()

    log_path = os.path.join(LOG_DIR, "03_sg_rounds.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\nTotal rows: {total_rows}\n")
        f.write(f"Columns: {fieldnames}\n")
        for line in log_lines: f.write(f"  {line}\n")

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Total rows : {total_rows:,}")
    print(f"  Output     : {out_path}")
    expected = "✓" if total_rows >= 50000 else "⚠ LOW"
    print(f"  Status     : {expected} (expected 50,000–120,000)")
    print(f"\n  ✓ Script 03 complete. Next: run 04_pull_odds.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    main(args.key)
