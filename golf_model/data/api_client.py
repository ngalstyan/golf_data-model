# ==============================================================================
# golf_model/data/api_client.py
# ==============================================================================
#
# DATAGOLF API CLIENT (Fresh Build)
# -----------------------------------
# Clean interface to all DataGolf API endpoints needed for Method 1.
# Built from scratch for the new architecture — does NOT reuse old scripts.
#
# Features:
#   - Automatic retry with exponential backoff (tenacity)
#   - Rate limiting (respects DataGolf's limits)
#   - Response caching to disk (avoid redundant API calls)
#   - Schema validation on every response
#   - Comprehensive logging
#
# DataGolf API docs: https://datagolf.com/api-access
#
# Endpoints used:
#   1. /historical-raw-data/rounds     → Round-level SG data (core dataset)
#   2. /historical-raw-data/event-list → Tournament metadata
#   3. /historical-odds/outrights      → Historical betting odds
#   4. /preds/player-list              → Player metadata
#   5. /preds/pre-tournament           → DataGolf's own predictions (benchmark)
#
# Usage:
#   from data.api_client import DataGolfClient
#   from config.settings import Settings
#
#   client = DataGolfClient(Settings())
#   rounds = client.get_historical_rounds(tour="pga", season=2023)
#   odds = client.get_historical_odds(tour="pga", season=2023, market="win")
#
# ==============================================================================

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DataGolfClient:
    """
    Client for the DataGolf API.
    
    Handles authentication, request formatting, retry logic,
    rate limiting, and response caching.
    
    Parameters
    ----------
    settings : Settings
        Project configuration (provides API key, base URL, cache dir).
        
    cache_responses : bool, default True
        If True, cache API responses to disk as JSON. Subsequent calls
        with the same parameters return cached data without hitting the API.
        Cache is stored in {DATA_DIR}/../cache/api/.
        
    Attributes
    ----------
    base_url : str
        DataGolf API base URL.
    session : requests.Session
        Persistent HTTP session (connection pooling).
    """

    # DataGolf rate limit: ~60 requests/minute for Scratch PLUS
    RATE_LIMIT_DELAY: float = 1.0  # seconds between requests

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache_responses: bool = True,
    ):
        self.settings = settings or Settings()
        self.base_url = self.settings.DATAGOLF_BASE_URL.rstrip("/")
        self.api_key = self.settings.DATAGOLF_API_KEY
        self.cache_responses = cache_responses

        # Cache directory
        self.cache_dir = self.settings.DATA_DIR.parent / "cache" / "api"
        if cache_responses:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Persistent session for connection pooling
        self.session = requests.Session()
        self.session.params = {"key": self.api_key}  # type: ignore
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "GolfBettingModel/1.0",
        })

        # Rate limiting state
        self._last_request_time: float = 0.0

        # Validation
        if not self.api_key:
            logger.warning(
                "DataGolf API key not set. Set GOLF_DATAGOLF_API_KEY env var "
                "or pass to Settings(). API calls will fail."
            )
        else:
            logger.info(
                "DataGolfClient initialized | base_url=%s | cache=%s",
                self.base_url,
                cache_responses,
            )

    # ==========================================================================
    # PUBLIC API — DATA ENDPOINTS
    # ==========================================================================

    def get_historical_rounds(
        self,
        tour: str = "pga",
        season: Optional[int] = None,
        event_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch round-level strokes-gained data.
        
        DataGolf endpoint: /historical-raw-data/rounds
        
        Parameters
        ----------
        tour : str, default "pga"
            Tour identifier: "pga", "kft", "euro".
        season : int, optional
            PGA Tour season year. If None, returns most recent.
        event_id : int, optional
            Specific event ID. If None, returns all events in the season.
            
        Returns
        -------
        pd.DataFrame
            Round-level data with SG components.
            
        Notes
        -----
        DataGolf returns data one event at a time for this endpoint.
        If season is specified without event_id, we need to:
          1. Get the event list for that season
          2. Loop through each event and fetch rounds
          3. Concatenate results
        """
        params: Dict[str, Any] = {
            "tour": tour,
            "file_format": "json",
        }

        if event_id is not None:
            # Single event fetch
            params["event_id"] = event_id
            data = self._get("/historical-raw-data/rounds", params)
            return self._rounds_response_to_df(data)

        if season is not None:
            # Fetch all events in the season, then get rounds for each
            logger.info(
                "Fetching all rounds for %s %d (multi-event fetch)...",
                tour.upper(), season,
            )
            events = self.get_event_list(tour=tour, season=season)

            if events.empty:
                logger.warning("No events found for %s %d", tour, season)
                return pd.DataFrame()

            frames = []
            event_ids = events["event_id"].unique()
            for i, eid in enumerate(event_ids):
                logger.debug(
                    "Fetching rounds for event %d/%d (id=%d)",
                    i + 1, len(event_ids), eid,
                )
                try:
                    params_evt = {**params, "event_id": int(eid)}
                    data = self._get("/historical-raw-data/rounds", params_evt)
                    df_evt = self._rounds_response_to_df(data)
                    if not df_evt.empty:
                        frames.append(df_evt)
                except Exception as e:
                    logger.warning("Failed to fetch event %d: %s", eid, e)
                    continue

            if not frames:
                return pd.DataFrame()

            result = pd.concat(frames, ignore_index=True)
            logger.info(
                "Fetched %d rounds across %d events for %s %d",
                len(result), len(frames), tour.upper(), season,
            )
            return result

        # No season or event_id → fetch current/latest
        data = self._get("/historical-raw-data/rounds", params)
        return self._rounds_response_to_df(data)

    def get_event_list(
        self,
        tour: str = "pga",
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch tournament/event metadata.
        
        DataGolf endpoint: /historical-raw-data/event-list
        
        Parameters
        ----------
        tour : str, default "pga"
        season : int, optional
            
        Returns
        -------
        pd.DataFrame
            Event metadata including event_id, name, dates, course info.
        """
        params: Dict[str, Any] = {
            "tour": tour,
            "file_format": "json",
        }
        if season is not None:
            params["season"] = season

        data = self._get("/historical-raw-data/event-list", params)

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and "events" in data:
            return pd.DataFrame(data["events"])
        else:
            logger.warning("Unexpected event list response format")
            return pd.DataFrame()

    def get_historical_odds(
        self,
        tour: str = "pga",
        season: Optional[int] = None,
        event_id: Optional[int] = None,
        market: str = "win",
        book: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical bookmaker odds.
        
        DataGolf endpoint: /historical-odds/outrights
        
        Parameters
        ----------
        tour : str, default "pga"
        season : int, optional
        event_id : int, optional
        market : str, default "win"
            Market type: "win", "top_5", "top_10", "top_20", "make_cut".
        book : str, optional
            Specific sportsbook. If None, returns all available books.
            
        Returns
        -------
        pd.DataFrame
            Historical odds data.
        """
        params: Dict[str, Any] = {
            "tour": tour,
            "market": market,
            "file_format": "json",
        }
        if season is not None:
            params["season"] = season
        if event_id is not None:
            params["event_id"] = event_id
        if book is not None:
            params["book"] = book

        data = self._get("/historical-odds/outrights", params)

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # DataGolf may nest odds under various keys
            for key in ["odds", "data", "results"]:
                if key in data:
                    return pd.DataFrame(data[key])
            return pd.DataFrame(data, index=[0]) if data else pd.DataFrame()
        return pd.DataFrame()

    def get_player_list(self) -> pd.DataFrame:
        """
        Fetch player metadata.
        
        DataGolf endpoint: /preds/player-list
        
        Returns
        -------
        pd.DataFrame
            Player IDs, names, countries, amateur status.
        """
        params = {"file_format": "json"}
        data = self._get("/preds/player-list", params)

        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_pre_tournament_predictions(
        self,
        tour: str = "pga",
        event_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch DataGolf's own pre-tournament predictions.
        
        Useful as a benchmark: if our model can't beat DataGolf's 
        public predictions, we have no edge.
        
        DataGolf endpoint: /preds/pre-tournament
        
        Returns
        -------
        pd.DataFrame
            DataGolf's predicted win probabilities per player.
        """
        params: Dict[str, Any] = {
            "tour": tour,
            "file_format": "json",
        }
        if event_id is not None:
            params["event_id"] = event_id

        data = self._get("/preds/pre-tournament", params)

        if isinstance(data, dict) and "baseline_history_fit" in data:
            return pd.DataFrame(data["baseline_history_fit"])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    def get_live_matchup_odds(
        self,
        tour: str = "pga",
        market: str = "tournament_matchups",
    ) -> pd.DataFrame:
        """
        Fetch current matchup odds from all available bookmakers.

        DataGolf endpoint: /betting-tools/matchups

        Returns one row per (matchup, book) pair so callers can compare
        odds across books and pick the best line.

        Parameters
        ----------
        tour : str, default "pga"
        market : str, default "tournament_matchups"

        Returns
        -------
        pd.DataFrame
            Columns: event_name, p1_dg_id, p1_player_name, p2_dg_id,
            p2_player_name, book, p1_odds, p2_odds, last_updated.
            Empty DataFrame if no matchups are available.
        """
        params = {
            "tour": tour,
            "market": market,
            "odds_format": "decimal",
            "file_format": "json",
        }
        data = self._get("/betting-tools/matchups", params)
        if not data or not isinstance(data, dict):
            return pd.DataFrame()

        event_name = data.get("event_name", "")
        last_updated = data.get("last_updated", "")
        match_list = data.get("match_list", data.get("matchups", []))

        rows: List[Dict[str, Any]] = []
        for m in match_list:
            if not isinstance(m, dict):
                continue
            odds_by_book = m.get("odds", {})
            if isinstance(odds_by_book, dict):
                for book_name, book_odds in odds_by_book.items():
                    if not isinstance(book_odds, dict):
                        continue
                    p1 = book_odds.get("p1") or book_odds.get("p1_odds")
                    p2 = book_odds.get("p2") or book_odds.get("p2_odds")
                    if p1 is None or p2 is None:
                        continue
                    rows.append({
                        "event_name": event_name,
                        "p1_dg_id": m.get("p1_dg_id"),
                        "p1_player_name": m.get("p1_player_name", ""),
                        "p2_dg_id": m.get("p2_dg_id"),
                        "p2_player_name": m.get("p2_player_name", ""),
                        "book": book_name,
                        "p1_odds": float(p1),
                        "p2_odds": float(p2),
                        "last_updated": last_updated,
                    })
            else:
                # Legacy format: offerings list
                for off in m.get("offerings", []):
                    if not isinstance(off, dict):
                        continue
                    rows.append({
                        "event_name": event_name,
                        "p1_dg_id": m.get("p1_dg_id"),
                        "p1_player_name": m.get("p1_player_name", ""),
                        "p2_dg_id": m.get("p2_dg_id"),
                        "p2_player_name": m.get("p2_player_name", ""),
                        "book": off.get("book", ""),
                        "p1_odds": float(off.get("p1_odds", 0)),
                        "p2_odds": float(off.get("p2_odds", 0)),
                        "last_updated": last_updated,
                    })

        if rows:
            logger.info(
                "Fetched %d matchup lines across %d books for %s",
                len(rows),
                len({r["book"] for r in rows}),
                event_name,
            )
        else:
            logger.warning("No live matchup odds returned")

        return pd.DataFrame(rows)

    # ==========================================================================
    # BULK DOWNLOAD — Multi-Season Fetch
    # ==========================================================================

    def download_all_seasons(
        self,
        tour: str = "pga",
        seasons: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download rounds, events, and odds for multiple seasons.
        Saves each as a CSV to output_dir.
        
        Parameters
        ----------
        tour : str, default "pga"
        seasons : list of int, optional
            Seasons to download. If None, uses Settings.TRAIN_SEASONS + HOLDOUT_SEASONS.
        output_dir : Path, optional
            Where to save CSVs. If None, uses Settings.DATA_DIR.
            
        Returns
        -------
        dict
            {"rounds": DataFrame, "events": DataFrame, "odds": DataFrame}
        """
        if seasons is None:
            seasons = self.settings.TRAIN_SEASONS + self.settings.HOLDOUT_SEASONS
        if output_dir is None:
            output_dir = self.settings.DATA_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %s data for seasons %s → %s",
            tour.upper(), seasons, output_dir,
        )

        # --- Events ---
        all_events = []
        for season in seasons:
            logger.info("Fetching events for %s %d...", tour, season)
            df = self.get_event_list(tour=tour, season=season)
            if not df.empty:
                df["season"] = season
                all_events.append(df)

        events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        if not events_df.empty:
            events_path = output_dir / f"events_{tour}.csv"
            events_df.to_csv(events_path, index=False)
            logger.info("Saved %d events → %s", len(events_df), events_path)

        # --- Rounds ---
        all_rounds = []
        for season in seasons:
            logger.info("Fetching rounds for %s %d...", tour, season)
            df = self.get_historical_rounds(tour=tour, season=season)
            if not df.empty:
                df["season"] = season
                all_rounds.append(df)

        rounds_df = pd.concat(all_rounds, ignore_index=True) if all_rounds else pd.DataFrame()
        if not rounds_df.empty:
            rounds_path = output_dir / f"rounds_{tour}.csv"
            rounds_df.to_csv(rounds_path, index=False)
            logger.info("Saved %d rounds → %s", len(rounds_df), rounds_path)

        # --- Odds ---
        all_odds = []
        for season in seasons:
            logger.info("Fetching odds for %s %d...", tour, season)
            df = self.get_historical_odds(tour=tour, season=season, market="win")
            if not df.empty:
                df["season"] = season
                all_odds.append(df)

        odds_df = pd.concat(all_odds, ignore_index=True) if all_odds else pd.DataFrame()
        if not odds_df.empty:
            odds_path = output_dir / f"odds_win_{tour}.csv"
            odds_df.to_csv(odds_path, index=False)
            logger.info("Saved %d odds rows → %s", len(odds_df), odds_path)

        return {
            "events": events_df,
            "rounds": rounds_df,
            "odds": odds_df,
        }

    # ==========================================================================
    # PRIVATE — HTTP & CACHING
    # ==========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make a GET request to the DataGolf API with retry and caching.
        
        Parameters
        ----------
        endpoint : str
            API endpoint path (e.g., "/historical-raw-data/rounds").
        params : dict, optional
            Query parameters.
            
        Returns
        -------
        Any
            Parsed JSON response (dict or list).
        """
        # Check cache first
        if self.cache_responses:
            cached = self._load_from_cache(endpoint, params)
            if cached is not None:
                logger.debug("Cache hit: %s", endpoint)
                return cached

        # Rate limiting
        self._rate_limit()

        # Build URL
        url = f"{self.base_url}{endpoint}"

        logger.debug("GET %s | params=%s", endpoint, params)

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
        except requests.HTTPError as e:
            if response.status_code == 401:
                logger.error("Authentication failed. Check your API key.")
            elif response.status_code == 429:
                logger.warning("Rate limited. Waiting 60s...")
                time.sleep(60)
                raise  # Will be retried by tenacity
            elif response.status_code == 404:
                logger.warning("Endpoint not found: %s", endpoint)
                return {}
            else:
                logger.error("HTTP %d: %s", response.status_code, e)
            raise

        data = response.json()

        # Save to cache
        if self.cache_responses:
            self._save_to_cache(endpoint, params, data)

        return data

    def _rate_limit(self):
        """Enforce minimum delay between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate a unique cache key for an API request."""
        key_data = f"{endpoint}|{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _load_from_cache(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> Optional[Any]:
        """Load a cached API response from disk."""
        cache_file = self.cache_dir / f"{self._cache_key(endpoint, params)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_to_cache(
        self, endpoint: str, params: Optional[Dict], data: Any
    ):
        """Save an API response to disk cache."""
        cache_file = self.cache_dir / f"{self._cache_key(endpoint, params)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except (TypeError, IOError) as e:
            logger.debug("Failed to cache response: %s", e)

    def clear_cache(self):
        """Delete all cached API responses."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            logger.info("API cache cleared")

    # ==========================================================================
    # PRIVATE — RESPONSE PARSING
    # ==========================================================================

    def _rounds_response_to_df(self, data: Any) -> pd.DataFrame:
        """
        Parse the DataGolf rounds endpoint response into a DataFrame.
        
        DataGolf returns round-level data in various nested formats.
        This normalizes everything into a flat DataFrame matching RoundsSchema.
        """
        if not data:
            return pd.DataFrame()

        # DataGolf may return list of dicts or nested structure
        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            # Check common nesting patterns
            for key in ["rounds", "data", "results", "scores"]:
                if key in data:
                    return pd.DataFrame(data[key])

            # Single event response may have event metadata + player rounds
            if "event_id" in data and "players" in data:
                players = data["players"]
                rows = []
                for player in players:
                    player_id = player.get("player_id") or player.get("dg_id")
                    player_name = player.get("player_name", "")
                    for round_data in player.get("rounds", []):
                        row = {
                            "event_id": data["event_id"],
                            "player_id": player_id,
                            "player_name": player_name,
                            **round_data,
                        }
                        rows.append(row)
                return pd.DataFrame(rows)

        logger.warning(
            "Unexpected rounds response format: %s",
            type(data).__name__,
        )
        return pd.DataFrame()
