# ==============================================================================
# golf_model/utils/logger.py
# ==============================================================================
#
# STRUCTURED LOGGING
# -------------------
# Provides a consistent logging interface across all modules.
# Each module gets its own named logger for easy filtering.
#
# Usage:
#   from utils.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("Loading 2023 tournament data...")
#   logger.warning("Player %s has only %d rounds", player_id, n_rounds)
#   logger.error("Schema validation failed: missing column '%s'", col_name)
#
# Output (console + file):
#   2026-02-27 14:30:01 | data.loader               | INFO    | Loading 2023 tournament data...
#
# ==============================================================================

import logging
import sys
from pathlib import Path
from typing import Optional


# Module-level flag to prevent duplicate handler attachment
_initialized_loggers: set = set()
_cached_settings = None


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Get or create a named logger with console and optional file output.
    
    Parameters
    ----------
    name : str
        Logger name. Convention: use __name__ to match module path.
        Example: "data.loader" from data/loader.py.
        
    level : str, optional
        Override log level. If None, uses Settings.LOG_LEVEL.
        Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        
    log_to_file : bool, optional
        Override file logging. If None, uses Settings.LOG_TO_FILE.
        
    log_dir : Path, optional
        Override log directory. If None, uses Settings.LOGS_DIR.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
        
    Notes
    -----
    - Loggers are cached by name — calling get_logger("data.loader") twice
      returns the same logger instance (standard Python logging behavior).
    - Handlers are only added once per logger to prevent duplicate messages.
    """

    # Lazy import to avoid circular dependency with config
    from config.settings import Settings

    logger = logging.getLogger(name)

    # Only configure each logger once
    if name in _initialized_loggers:
        return logger

    # Load defaults from settings (cache to avoid repeated __post_init__ side effects)
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = Settings()
    cfg = _cached_settings
    _level = getattr(logging, (level or cfg.LOG_LEVEL).upper(), logging.INFO)
    _log_to_file = log_to_file if log_to_file is not None else cfg.LOG_TO_FILE
    _log_dir = log_dir or cfg.LOGS_DIR
    _log_format = cfg.LOG_FORMAT

    logger.setLevel(_level)

    # --- Console Handler ---
    # Writes to stderr (standard for logging; stdout reserved for output)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(_level)
    console_handler.setFormatter(logging.Formatter(_log_format))
    logger.addHandler(console_handler)

    # --- File Handler (optional) ---
    # Single rotating log file for the entire project
    if _log_to_file:
        _log_dir.mkdir(parents=True, exist_ok=True)
        log_file = _log_dir / "golf_model.log"

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(_level)
        file_handler.setFormatter(logging.Formatter(_log_format))
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    # Mark as initialized
    _initialized_loggers.add(name)

    return logger
