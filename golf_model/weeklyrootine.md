## Weekly Routine (Notebook 09)

### Pre-Tournament (Tuesday/Wednesday)
1. Pull fresh data: `python scripts/14_pull_season_data.py --key YOUR_API_KEY`
2. Restart kernel, run cells 1–7 → generate predictions & identify edges
3. Run Step 6a (News Research Briefing) → review player intel for each matchup
4. Review bets + research briefing, decide which to place
5. Run Step 6b → log confirmed bets to persistent CSV
6. Place bets on your sportsbook (DraftKings, Pinnacle, BetMGM)

### Post-Tournament (Sunday/Monday)
7. Re-pull data: `python scripts/14_pull_season_data.py --key YOUR_API_KEY`
8. Run Step 7 (Dry Run) → check hypothetical P&L from tournament results
9. Run Step 7b → settle bets in persistent log
10. Run Step 9 → see lifetime performance dashboard
