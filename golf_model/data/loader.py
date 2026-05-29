# ==============================================================================
# golf_model/data/loader.py
# ==============================================================================
#
# DATA LOADER
# ------------
# Loads CSV files from disk, validates against schemas, and returns 
# clean, typed DataFrames ready for the feature engineering pipeline.
#
# Design principles:
#   1. Raw CSVs are NEVER modified. All transformations create new DataFrames.
#   2. Every load validates against the corresponding schema.
#   3. Type casting is explicit and logged.
#   4. Missing data is flagged, not silently dropped.
#
# Usage:
#   from data.loader import DataLoader
#   from config.settings import Settings
#
#   loader = DataLoader(Settings())
#   rounds_df = loader.load_rounds()
#   events_df = loader.load_events()
#   odds_df   = loader.load_odds()
#
# File discovery:
#   The loader searches DATA_DIR for CSV files matching expected patterns.
#   It handles both single-file and multi-file (per-season) layouts:
#     - Single file:  rounds.csv, events.csv, odds.csv
#     - Multi-file:   rounds_2019.csv, rounds_2020.csv, ...
#
# ==============================================================================

import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config.settings import Settings
from data.schemas import (
    RoundsSchema,
    EventsSchema,
    OddsSchema,
    CourseSchema,
    PlayerSchema,
    validate_dataframe,
)
from utils.logger import get_logger
from utils.helpers import summarize_dataframe

logger = get_logger(__name__)


class DataLoader:
    """
    Loads and validates golf model data from CSV files.
    
    This is the single entry point for all data into the system.
    Every DataFrame returned by this class has been:
      1. Read from disk
      2. Validated against its schema
      3. Type-cast to correct dtypes
      4. Logged with summary statistics
    
    Parameters
    ----------
    settings : Settings
        Project configuration (provides DATA_DIR path and other params).
        
    Attributes
    ----------
    data_dir : Path
        Root directory containing CSV files.
    settings : Settings
        Full project configuration.
        
    Examples
    --------
    >>> cfg = Settings(DATA_DIR=Path("/path/to/my/csvs"))
    >>> loader = DataLoader(cfg)
    >>> rounds = loader.load_rounds()
    >>> print(f"Loaded {len(rounds):,} rounds")
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.data_dir = self.settings.DATA_DIR

        # Cache loaded DataFrames to avoid re-reading
        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info("DataLoader initialized | data_dir=%s", self.data_dir)

        if not self.data_dir.exists():
            logger.warning(
                "DATA_DIR does not exist: %s. "
                "Create it and place your CSV files there.",
                self.data_dir,
            )

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def load_rounds(
        self,
        seasons: Optional[List[int]] = None,
        tours: Optional[List[str]] = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load round-level strokes-gained data.
        
        This is the primary dataset for the entire model.
        Each row = one player's performance in one round of one tournament.
        
        Parameters
        ----------
        seasons : list of int, optional
            Filter to specific PGA Tour seasons. 
            If None, loads all available data.
            
        tours : list of str, optional
            Filter to specific tours ("pga", "kft", "euro").
            If None, loads all tours.
            
        force_reload : bool, default False
            If True, bypass cache and re-read from disk.
            
        Returns
        -------
        pd.DataFrame
            Validated rounds DataFrame with columns per RoundsSchema.
            
        Raises
        ------
        FileNotFoundError
            If no rounds CSV files are found in DATA_DIR.
        """
        cache_key = "rounds"
        if cache_key in self._cache and not force_reload:
            logger.debug("Returning cached rounds data")
            return self._apply_filters(self._cache[cache_key], seasons, tours)

        logger.info("Loading rounds data from %s", self.data_dir)

        # Discover CSV files matching rounds pattern
        df = self._discover_and_load(
            patterns=[
                "rounds*.csv",
                "sg_rounds*.csv",
                "*round*data*.csv",
                "*historical*round*.csv",
            ],
            dataset_name="rounds",
        )

        # Type casting
        df = self._cast_rounds_types(df)

        # Schema validation
        issues = validate_dataframe(df, RoundsSchema())
        self._report_issues(issues, "rounds")

        # Cache and log
        self._cache[cache_key] = df
        logger.info("Rounds loaded | %s", summarize_dataframe(df, "Rounds"))

        return self._apply_filters(df, seasons, tours)

    def load_events(
        self,
        seasons: Optional[List[int]] = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load tournament/event metadata.
        
        Parameters
        ----------
        seasons : list of int, optional
            Filter to specific seasons.
        force_reload : bool, default False
            Bypass cache.
            
        Returns
        -------
        pd.DataFrame
            Validated events DataFrame with columns per EventsSchema.
        """
        cache_key = "events"
        if cache_key in self._cache and not force_reload:
            logger.debug("Returning cached events data")
            df = self._cache[cache_key]
            if seasons:
                df = df[df["calendar_year"].isin(seasons)]
            return df

        logger.info("Loading events data from %s", self.data_dir)

        df = self._discover_and_load(
            patterns=[
                "events*.csv",
                "event_list*.csv",
                "tournament*list*.csv",
                "*tournaments*.csv",
                "schedule*.csv",        # ← add this
            ],
            dataset_name="events",
        )

        df = self._cast_events_types(df)

        issues = validate_dataframe(df, EventsSchema())
        self._report_issues(issues, "events")

        self._cache[cache_key] = df
        logger.info("Events loaded | %s", summarize_dataframe(df, "Events"))

        if seasons:
            df = df[df["calendar_year"].isin(seasons)]
        return df

    def load_odds(
        self,
        seasons: Optional[List[int]] = None,
        books: Optional[List[str]] = None,
        market: str = "win",
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load historical bookmaker odds.
        
        Parameters
        ----------
        seasons : list of int, optional
            Filter to specific seasons.
        books : list of str, optional
            Filter to specific sportsbooks (e.g., ["pinnacle", "draftkings"]).
        market : str, default "win"
            Market type to load ("win", "top_5", "top_10", "make_cut").
        force_reload : bool, default False
            Bypass cache.
            
        Returns
        -------
        pd.DataFrame
            Validated odds DataFrame with columns per OddsSchema.
        """
        cache_key = f"odds_{market}"
        if cache_key in self._cache and not force_reload:
            logger.debug("Returning cached odds data (%s)", market)
            df = self._cache[cache_key]
            return self._filter_odds(df, seasons, books)

        logger.info("Loading odds data (market=%s) from %s", market, self.data_dir)

        df = self._discover_and_load(
            patterns=[
                f"odds*{market}*.csv",
                f"*odds*outright*.csv",
                f"*historical*odds*.csv",
                "odds*.csv",
            ],
            dataset_name=f"odds ({market})",
        )

        df = self._cast_odds_types(df)

        issues = validate_dataframe(df, OddsSchema())
        self._report_issues(issues, f"odds ({market})")

        self._cache[cache_key] = df
        logger.info("Odds loaded | %s", summarize_dataframe(df, f"Odds ({market})"))

        return self._filter_odds(df, seasons, books)

    def load_course_features(
        self,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load course characteristic features (γ_c vector).
        
        Returns
        -------
        pd.DataFrame
            Course features DataFrame with columns per CourseSchema.
        """
        cache_key = "course_features"
        if cache_key in self._cache and not force_reload:
            return self._cache[cache_key]

        logger.info("Loading course features from %s", self.data_dir)

        df = self._discover_and_load(
            patterns=[
                "course_features*.csv",
                "course*char*.csv",
                "courses*.csv",
            ],
            dataset_name="course_features",
        )

        issues = validate_dataframe(df, CourseSchema())
        self._report_issues(issues, "course_features")

        self._cache[cache_key] = df
        logger.info("Course features loaded | %s",
                     summarize_dataframe(df, "Courses"))

        return df

    def load_players(
        self,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load player metadata and ratings.
        
        Returns
        -------
        pd.DataFrame
            Player metadata DataFrame.
        """
        cache_key = "players"
        if cache_key in self._cache and not force_reload:
            return self._cache[cache_key]

        logger.info("Loading player data from %s", self.data_dir)

        df = self._discover_and_load(
            patterns=[
                "players*.csv",
                "player*list*.csv",
                "*player*ratings*.csv",
            ],
            dataset_name="players",
        )

        issues = validate_dataframe(df, PlayerSchema())
        self._report_issues(issues, "players")

        self._cache[cache_key] = df
        logger.info("Players loaded | %s", summarize_dataframe(df, "Players"))

        return df

    def list_available_files(self) -> List[str]:
        """List all CSV files found in DATA_DIR."""
        if not self.data_dir.exists():
            return []
        files = sorted(self.data_dir.glob("*.csv"))
        return [f.name for f in files]

    def clear_cache(self):
        """Clear all cached DataFrames. Forces next load to read from disk."""
        self._cache.clear()
        logger.info("DataLoader cache cleared")

    def get_data_summary(self) -> Dict[str, str]:
        """
        Generate a summary of all loaded datasets.
        
        Returns
        -------
        dict
            Dataset name → summary string.
        """
        return {
            name: summarize_dataframe(df, name)
            for name, df in self._cache.items()
        }

    # ==========================================================================
    # PRIVATE HELPERS
    # ==========================================================================

    def _discover_and_load(
        self,
        patterns: List[str],
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Search for CSV files matching any of the given glob patterns,
        then concatenate and return as a single DataFrame.
        
        Handles both single-file and multi-file (per-season) layouts.
        """
        all_files: List[Path] = []

        for pattern in patterns:
            found = sorted(self.data_dir.glob(pattern))
            all_files.extend(found)

        # Deduplicate while preserving order
        seen = set()
        unique_files = []
        for f in all_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        if not unique_files:
            available = self.list_available_files()
            raise FileNotFoundError(
                f"No {dataset_name} CSV files found in {self.data_dir}.\n"
                f"Searched patterns: {patterns}\n"
                f"Available files: {available}"
            )

        logger.info(
            "Found %d file(s) for %s: %s",
            len(unique_files),
            dataset_name,
            [f.name for f in unique_files],
        )

        # Read and concatenate all matching files
        frames = []
        for filepath in unique_files:
            try:
                df_chunk = pd.read_csv(filepath, low_memory=False)
                df_chunk["_source_file"] = filepath.name  # Track provenance
                frames.append(df_chunk)
                logger.debug(
                    "Read %s: %d rows × %d cols",
                    filepath.name,
                    len(df_chunk),
                    len(df_chunk.columns),
                )
            except Exception as e:
                logger.error("Failed to read %s: %s", filepath.name, e)
                raise

        df = pd.concat(frames, ignore_index=True)

        # Standardize column names: lowercase, strip whitespace, replace spaces
        df.columns = (
            df.columns.str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        return df

    def _cast_rounds_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast rounds DataFrame columns to correct types.
        
        Handles common issues:
            - player_id as float (due to NaN) → int
            - date as various formats → datetime
            - SG columns as string → float
        """
        df = df.copy()

        # Integer columns (handle NaN → float issue)
        for col in ["player_id", "event_id", "round_num", "course_id"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Don't convert to int yet if NaN present — keep float
                n_null = df[col].isna().sum()
                if n_null > 0:
                    logger.warning(
                        "Column '%s' has %d null values (%.1f%%)",
                        col, n_null, 100 * n_null / len(df),
                    )

        # Float columns (SG components)
        sg_cols = ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt"]
        for col in sg_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Date parsing
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            n_bad_dates = df["date"].isna().sum()
            if n_bad_dates > 0:
                logger.warning(
                    "Column 'date' has %d unparseable dates", n_bad_dates
                )

        # Season inference (if not present)
        if "season" not in df.columns and "date" in df.columns:
            from utils.helpers import season_from_date
            df["season"] = df["date"].apply(
                lambda d: season_from_date(d) if pd.notna(d) else np.nan
            )
            logger.info("Inferred 'season' column from dates")

        return df

    def _cast_events_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast events DataFrame columns to correct types."""
        df = df.copy()

        for col in ["event_id", "calendar_year", "season", "course_id", "field_size"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["start_date", "end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "purse" in df.columns:
            df["purse"] = pd.to_numeric(
                df["purse"].astype(str).str.replace(",", "").str.replace("$", ""),
                errors="coerce",
            )

        return df

    def _cast_odds_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast odds DataFrame columns to correct types."""
        df = df.copy()

        for col in ["event_id", "dg_id"]:           # was: player_id
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["open_odds", "close_odds"]:      # was: decimal_odds
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "bookmaker" in df.columns:               # was: book
            df["bookmaker"] = df["bookmaker"].astype(str).str.lower().str.strip()

        if "market" in df.columns:
            df["market"] = df["market"].astype(str).str.lower().str.strip()

        return df

    def _apply_filters(
        self,
        df: pd.DataFrame,
        seasons: Optional[List[int]] = None,
        tours: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply season and tour filters to a DataFrame."""
        result = df

        if seasons and "season" in df.columns:
            result = result[result["season"].isin(seasons)]
            logger.debug("Filtered to seasons %s: %d rows", seasons, len(result))

        if tours and "tour" in df.columns:
            tours_lower = [t.lower() for t in tours]
            result = result[result["tour"].str.lower().isin(tours_lower)]
            logger.debug("Filtered to tours %s: %d rows", tours, len(result))

        return result

    def _filter_odds(
        self,
        df: pd.DataFrame,
        seasons: Optional[List[int]] = None,
        books: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply season and book filters to odds DataFrame."""
        result = df

        if seasons and "season" in df.columns:
            result = result[result["season"].isin(seasons)]

        if books and "bookmaker" in df.columns:         # was: "book"
            books_lower = [b.lower() for b in books]
            result = result[result["bookmaker"].isin(books_lower)]
            logger.debug("Filtered to books %s: %d rows", books, len(result))

        return result

    def _report_issues(self, issues: List[str], dataset_name: str):
        """Log validation issues as warnings or raise if critical."""
        if not issues:
            logger.info("Schema validation passed for %s", dataset_name)
            return

        logger.warning(
            "Schema validation found %d issue(s) for %s:",
            len(issues),
            dataset_name,
        )
        for issue in issues:
            logger.warning("  ⚠ %s", issue)
