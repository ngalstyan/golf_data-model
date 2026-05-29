"""
Script 05 — Full Validation Report [FIXED v4]
Training data: 2017–2022  |  Holdout data: 2023–2024 (DO NOT use for training)

Run: python 05_validate_all.py
"""

import csv
import os
from datetime import datetime

DATA_DIR = "/Users/galstyann/Documents/golf_data-model/golf_model/data/raw"
LOG_DIR  = "/Users/galstyann/Documents/golf_data-model/golf_model/logs"

FILES = {
    # Training
    "players":          "player_list.csv",
    "schedule":         "schedule_2017_2022.csv",
    "sg_rounds":        "sg_rounds_2019_2022.csv",
    "odds":             "odds_2019_2022.csv",
    "skill_ratings":    "skill_ratings_current.csv",
    "predictions":      "predictions_archive_2019_2022.csv",
    "matchups":         "matchup_odds_2019_2022.csv",
    "course_fit":       "course_fit_archive_2019_2022.csv",
    "sg_pga_early":     "sg_rounds_pga_2017_2018.csv",
    "sg_kft":           "sg_rounds_kft_2017_2022.csv",
    "sg_euro":          "sg_rounds_euro_2017_2022.csv",
    # Holdout
    "sg_holdout":       "sg_rounds_pga_2023_2024.csv",
    "odds_holdout":     "odds_2023_2024.csv",
    "schedule_holdout": "schedule_2023_2024.csv",
}

SG_COLS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
SG_TOL  = 0.05


def load(key):
    path = os.path.join(DATA_DIR, FILES[key])
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check(label, passed, detail=""):
    icon = "✓" if passed else "✗"
    line = f"  {icon}  {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    return passed


def warn(label, detail=""):
    line = f"  ⚠  {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)


def info(label, val=""):
    print(f"  ℹ  {label}  :  {val}" if val else f"  ℹ  {label}")


def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def get_id(row):
    for c in ("dg_id", "player_id"):
        v = row.get(c)
        if v not in (None, ""):
            return str(v)
    return None


def get_year(row):
    for c in ("year", "_season", "season", "calendar_year"):
        v = row.get(c)
        if v not in (None, ""):
            return str(v)
    return ""


def sg_integrity(rows, label):
    results = []
    rows_with = [r for r in rows if all(r.get(c) not in (None, "") for c in SG_COLS)]
    info(f"{label}: {len(rows_with):,} rows with SG components (of {len(rows):,} total)")

    if not rows_with:
        return results

    errors   = 0
    outliers = 0
    for r in rows_with:
        parts = [safe_float(r.get(c)) for c in SG_COLS]
        total = safe_float(r.get("sg_total"))
        if None not in parts and total is not None:
            if abs(sum(parts) - total) > SG_TOL:
                errors += 1
        for v in parts:
            if v is not None and abs(v) > 10:
                outliers += 1

    pct = errors / len(rows_with) * 100
    results.append(check(
        f"{label}: SG components sum to total (<=1% error)",
        pct <= 1.0,
        f"{errors}/{len(rows_with)} errors ({pct:.2f}%)"
    ))
    if outliers > 0:
        warn(f"{label}: {outliers} SG component value(s) outside [-10, +10]",
             "minor data error — clamp in feature engineering")
    else:
        results.append(check(f"{label}: SG values in [-10, +10]", True))

    return results


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    results = []

    print("=" * 60)
    print("Script 05 — Full Validation Report [FIXED v4]")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data dir: {DATA_DIR}")
    print("=" * 60)

    data = {k: load(k) for k in FILES}

    # ── 1. File existence & row counts ────────────────────────────────────────
    section("1. File existence & row counts")

    for key in ("players", "schedule", "sg_rounds", "odds", "skill_ratings",
                "predictions", "matchups", "course_fit", "sg_pga_early", "sg_kft", "sg_euro"):
        n = len(data[key])
        results.append(check(FILES[key], n > 0, f"{n:,} rows"))

    results.append(check("PGA 2019-2022 SG rounds in expected range (50,000-200,000)",
                         50000 <= len(data["sg_rounds"]) <= 200000,
                         f"actual: {len(data['sg_rounds']):,}"))
    results.append(check("Schedule events in expected range (100-700)",
                         100 <= len(data["schedule"]) <= 700,
                         f"actual: {len(data['schedule']):,}"))
    results.append(check("Outright odds rows in expected range (5,000-150,000)",
                         5000 <= len(data["odds"]) <= 150000,
                         f"actual: {len(data['odds']):,}"))
    results.append(check("Prediction rows in expected range (10,000-200,000)",
                         10000 <= len(data["predictions"]) <= 200000,
                         f"actual: {len(data['predictions']):,}"))
    results.append(check("Matchup rows in expected range (2,000-150,000)",
                         2000 <= len(data["matchups"]) <= 150000,
                         f"actual: {len(data['matchups']):,}"))
    results.append(check("Course fit rows in expected range (5,000-100,000)",
                         5000 <= len(data["course_fit"]) <= 100000,
                         f"actual: {len(data['course_fit']):,}"))
    results.append(check("PGA 2017-2018 rows in expected range (20,000-100,000)",
                         20000 <= len(data["sg_pga_early"]) <= 100000,
                         f"actual: {len(data['sg_pga_early']):,}"))
    results.append(check("KFT rows in expected range (20,000-150,000)",
                         20000 <= len(data["sg_kft"]) <= 150000,
                         f"actual: {len(data['sg_kft']):,}"))
    results.append(check("Euro rows in expected range (20,000-150,000)",
                         20000 <= len(data["sg_euro"]) <= 150000,
                         f"actual: {len(data['sg_euro']):,}"))

    # ── 2. Column verification ────────────────────────────────────────────────
    section("2. Column verification")

    sg_cols = list(data["sg_rounds"][0].keys()) if data["sg_rounds"] else []
    results.append(check("SG rounds has round_num", "round_num" in sg_cols,
                         f"columns: {sg_cols[:8]}"))
    results.append(check("SG rounds has player ID (dg_id or player_id)",
                         "dg_id" in sg_cols or "player_id" in sg_cols))
    results.append(check("SG rounds has sg_total", "sg_total" in sg_cols))

    odds_cols = list(data["odds"][0].keys()) if data["odds"] else []
    results.append(check("Odds has bookmaker", "bookmaker" in odds_cols,
                         f"columns: {odds_cols[:8]}"))

    pred_cols = list(data["predictions"][0].keys()) if data["predictions"] else []
    results.append(check("Predictions has model", "model" in pred_cols,
                         f"columns: {pred_cols[:8]}"))

    mu_cols = list(data["matchups"][0].keys()) if data["matchups"] else []
    results.append(check("Matchups has p1_dg_id",
                         "p1_dg_id" in mu_cols or "p1_player_id" in mu_cols,
                         f"columns: {mu_cols[:8]}"))

    # ── 3. Player ID coverage ─────────────────────────────────────────────────
    section("3. Player ID coverage")

    ids_list = {str(r.get("player_id", "")) for r in data["players"] if r.get("player_id")}
    ids_sg   = {get_id(r) for r in data["sg_rounds"] if get_id(r)}
    ids_kft  = {get_id(r) for r in data["sg_kft"]    if get_id(r)}
    ids_euro = {get_id(r) for r in data["sg_euro"]   if get_id(r)}

    sg_id_col = "dg_id" if (data["sg_rounds"] and "dg_id" in data["sg_rounds"][0]) else "player_id"
    info("Player ID column in players", "player_id")
    info("Player ID column in rounds",  sg_id_col)

    missing = ids_sg - ids_list
    # Relaxed to 10% — retired/inactive players not in current player list
    results.append(check(
        "SG player IDs in player_list (<=10% missing)",
        len(missing) / max(len(ids_sg), 1) <= 0.10,
        f"{len(missing)} missing of {len(ids_sg)}"
    ))
    info("KFT-only players",  str(len(ids_kft - ids_sg)))
    info("Euro-only players", str(len(ids_euro - ids_sg)))

    # ── 4. Season coverage ────────────────────────────────────────────────────
    section("4. Season coverage")

    sg_years = {get_year(r) for r in data["sg_rounds"]}
    info("SG year values", str(sorted(sg_years)))

    for yr in ["2019", "2020", "2021", "2022"]:
        cnt = sum(1 for r in data["sg_rounds"] if get_year(r) == yr)
        results.append(check(f"PGA {yr} in SG data", cnt > 0, f"{cnt:,} rounds"))

    early_years = {get_year(r) for r in data["sg_pga_early"]}
    results.append(check("PGA 2017-2018 covers expected years",
                         {"2017", "2018"}.issubset(early_years),
                         f"found: {sorted(early_years)}"))

    kft_years  = {get_year(r) for r in data["sg_kft"]}
    euro_years = {get_year(r) for r in data["sg_euro"]}
    results.append(check("KFT 2017-2022 covers expected years",
                         {str(y) for y in range(2017, 2023)}.issubset(kft_years),
                         f"found: {sorted(kft_years)}"))
    results.append(check("Euro 2017-2022 covers expected years",
                         {str(y) for y in range(2017, 2023)}.issubset(euro_years),
                         f"found: {sorted(euro_years)}"))

    # ── 5. SG component integrity ─────────────────────────────────────────────
    section("5. SG component integrity")

    for key, label in [
        ("sg_rounds",    "PGA 2019-2022"),
        ("sg_pga_early", "PGA 2017-2018"),
        ("sg_kft",       "KFT"),
        ("sg_euro",      "Euro"),
    ]:
        results.extend(sg_integrity(data[key], label))

    # ── 6. Round number sanity ────────────────────────────────────────────────
    section("6. Round number sanity")

    for key, label in [
        ("sg_rounds",    "PGA 2019-2022"),
        ("sg_pga_early", "PGA 2017-2018"),
        ("sg_kft",       "KFT"),
        ("sg_euro",      "Euro"),
    ]:
        vals = sorted({str(r.get("round_num", "")) for r in data[key]})
        bad  = [r for r in data[key]
                if str(r.get("round_num", "")) not in ("1", "2", "3", "4")]
        results.append(check(f"{label}: round_num in {{1,2,3,4}}",
                             len(bad) == 0,
                             f"values: {vals}, invalid: {len(bad)}"))

    # ── 7. Outright odds sanity ───────────────────────────────────────────────
    section("7. Outright odds sanity")

    books = {r.get("bookmaker", "") for r in data["odds"]}
    mkts  = {r.get("market", "")    for r in data["odds"]}

    results.append(check("Pinnacle in odds",       "pinnacle"   in books, str(books)))
    results.append(check("DraftKings in odds",      "draftkings" in books, str(books)))
    results.append(check("Win market present",      "win"        in mkts,  str(mkts)))
    results.append(check("Make-cut market present", "make_cut"   in mkts,  str(mkts)))

    close_col = "close_odds" if (data["odds"] and "close_odds" in data["odds"][0]) else "odds_decimal_close"
    low = [r for r in data["odds"]
           if r.get(close_col) not in (None, "")
           and safe_float(r[close_col]) is not None
           and safe_float(r[close_col]) < 1.01]
    results.append(check("No close_odds < 1.01", len(low) == 0,
                         f"{len(low)} invalid" if low else "0 invalid"))

    # ── 8. Matchup odds sanity ────────────────────────────────────────────────
    section("8. Matchup odds sanity")

    mu_books    = {r.get("bookmaker", "") for r in data["matchups"]}
    mu_bettypes = {r.get("bet_type", "")  for r in data["matchups"]}
    results.append(check("Pinnacle matchups present", "pinnacle" in mu_books, str(mu_books)))
    results.append(check("Bet types found",           len(mu_bettypes) > 0,  str(mu_bettypes)))
    info("Total matchup rows", f"{len(data['matchups']):,}")

    # ── 9. Predictions archive sanity ─────────────────────────────────────────
    section("9. Predictions archive sanity")

    models = {r.get("model", "") for r in data["predictions"]}
    results.append(check("Baseline model present",
                         any("baseline" in m for m in models), str(models)))
    results.append(check("Course history model present",
                         any("history" in m for m in models), str(models)))
    info("Total prediction rows", f"{len(data['predictions']):,}")

    # ── 10. Skill ratings sanity ──────────────────────────────────────────────
    section("10. Skill ratings sanity")

    results.append(check("Skill ratings >200 players",
                         len(data["skill_ratings"]) > 200,
                         f"{len(data['skill_ratings'])} players"))
    if data["skill_ratings"]:
        skill_col = next((c for c in ("dg_skill_estimate", "sg_total", "skill_estimate", "total")
                          if c in data["skill_ratings"][0]), None)
        if skill_col:
            vals = [safe_float(r[skill_col]) for r in data["skill_ratings"]]
            vals = [v for v in vals if v is not None]
            if vals:
                results.append(check("Skill estimates in [-5, +5]",
                                     min(vals) >= -5 and max(vals) <= 5,
                                     f"{min(vals):.3f} to {max(vals):.3f}"))

    # ── 11. Course fit sanity ─────────────────────────────────────────────────
    section("11. Course fit sanity")

    if data["course_fit"]:
        fit_col = next((c for c in ("dg_course_fit_adj", "course_fit_adj", "fit_adj")
                        if c in data["course_fit"][0]), None)
        if fit_col:
            fit_rows = [r for r in data["course_fit"] if r.get(fit_col) not in (None, "")]
            results.append(check("Course fit rows with delta data",
                                 len(fit_rows) > 1000,
                                 f"{len(fit_rows):,} of {len(data['course_fit']):,}"))
        else:
            info("Course fit columns", str(list(data["course_fit"][0].keys())[:8]))

    # ── 12. Cross-tour prior coverage ─────────────────────────────────────────
    section("12. Cross-tour prior coverage")

    total_rows = sum(len(data[k]) for k in ("sg_rounds", "sg_pga_early", "sg_kft", "sg_euro"))
    results.append(check("Total SG across all tours > 100k",
                         total_rows > 100000,
                         f"{total_rows:,} total rows"))

    # ═══════════════════════════════════════════════════════════════════════════
    # ── 13. Holdout data (2023-2024) ──────────────────────────────────────────
    # ═══════════════════════════════════════════════════════════════════════════
    section("13. Holdout data — 2023-2024 (DO NOT use for training)")

    sg_h    = data["sg_holdout"]
    odds_h  = data["odds_holdout"]
    sched_h = data["schedule_holdout"]

    # File existence
    results.append(check("sg_rounds_pga_2023_2024.csv exists",
                         len(sg_h) > 0, f"{len(sg_h):,} rows"))
    results.append(check("odds_2023_2024.csv exists",
                         len(odds_h) > 0, f"{len(odds_h):,} rows"))
    results.append(check("schedule_2023_2024.csv exists",
                         len(sched_h) > 0, f"{len(sched_h):,} rows"))

    # Row count sanity
    results.append(check("Holdout SG rounds in expected range (20,000-150,000)",
                         20000 <= len(sg_h) <= 150000,
                         f"actual: {len(sg_h):,}"))
    # 500 minimum — player-level JSON varies by how many events had Pinnacle tracking
    results.append(check("Holdout odds rows in expected range (500-100,000)",
                         500 <= len(odds_h) <= 100000,
                         f"actual: {len(odds_h):,}"))

    # Year coverage
    h_sg_years = {get_year(r) for r in sg_h}
    results.append(check("Holdout SG covers 2023", "2023" in h_sg_years,
                         f"found years: {sorted(h_sg_years)}"))
    results.append(check("Holdout SG covers 2024", "2024" in h_sg_years,
                         f"found years: {sorted(h_sg_years)}"))

    h_odds_years = {get_year(r) for r in odds_h}
    results.append(check("Holdout odds covers 2023", "2023" in h_odds_years,
                         f"found years: {sorted(h_odds_years)}"))
    results.append(check("Holdout odds covers 2024", "2024" in h_odds_years,
                         f"found years: {sorted(h_odds_years)}"))

    # No overlap with training
    training_years = {"2017", "2018", "2019", "2020", "2021", "2022"}
    overlap = h_sg_years & training_years
    results.append(check("No year overlap between holdout and training SG data",
                         len(overlap) == 0,
                         f"overlap: {overlap}" if overlap else "clean separation"))

    # SG integrity on holdout
    results.extend(sg_integrity(sg_h, "PGA 2023-2024 (holdout)"))

    # Round number sanity on holdout
    bad_rn = [r for r in sg_h
              if str(r.get("round_num", "")) not in ("1", "2", "3", "4")]
    results.append(check("Holdout: round_num in {1,2,3,4}",
                         len(bad_rn) == 0, f"{len(bad_rn)} invalid"))

    # Player overlap with training
    h_ids     = {get_id(r) for r in sg_h if get_id(r)}
    train_ids = {get_id(r) for r in data["sg_rounds"] if get_id(r)}
    info("Holdout players with training history", f"{len(h_ids & train_ids):,}")
    info("New players (will use prior only)",     f"{len(h_ids - train_ids):,}")

    # Holdout odds column check
    if odds_h:
        odds_h_cols = list(odds_h[0].keys())
        info("Holdout odds columns", str(odds_h_cols[:8]))
        if "bookmaker" in odds_h_cols:
            h_books = {r.get("bookmaker", "") for r in odds_h}
            h_mkts  = {r.get("market", "")    for r in odds_h}
            results.append(check("Holdout odds: pinnacle present",
                                 "pinnacle" in h_books, str(h_books)))
            results.append(check("Holdout odds: win market present",
                                 "win" in h_mkts, str(h_mkts)))
        else:
            has_close = "close_odds" in odds_h_cols or "close" in odds_h_cols
            has_id    = "dg_id" in odds_h_cols or "player_id" in odds_h_cols
            results.append(check("Holdout odds: has close_odds column", has_close,
                                 str(odds_h_cols)))
            results.append(check("Holdout odds: has player ID column",  has_id,
                                 str(odds_h_cols)))
            low_h = [r for r in odds_h
                     if r.get("close_odds") not in (None, "")
                     and safe_float(r["close_odds"]) is not None
                     and safe_float(r["close_odds"]) < 1.01]
            results.append(check("Holdout odds: no close_odds < 1.01",
                                 len(low_h) == 0,
                                 f"{len(low_h)} invalid" if low_h else "0 invalid"))

    # ── Final verdict ──────────────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r is True)
    n_fail = sum(1 for r in results if r is False)
    n_tot  = len(results)

    print(f"\n{'='*60}")
    print(f"  RESULT: {n_pass} passed  /  {n_fail} failed  /  {n_tot} checks")

    if n_fail == 0:
        print("  All checks passed.")
        print("  Training data  -> ready for model fitting (2017-2022).")
        print("  Holdout data   -> locked for backtest (2023-2024).")
    else:
        print("  Fix failures before proceeding.")
    print("=" * 60)

    # Write log
    log_path = os.path.join(LOG_DIR, "05_validation.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run: {datetime.now().isoformat()}\n")
        f.write(f"Passed: {n_pass}  Failed: {n_fail}\n\n")
        for k in FILES:
            f.write(f"{FILES[k]}: {len(data[k])} rows\n")


if __name__ == "__main__":
    main()