# ==============================================================================
# golf_model/data/weather.py
# ==============================================================================
#
# WEATHER DATA INTEGRATION
# --------------------------
# Historical weather data from Open-Meteo (free tier, no API key needed).
#
# Purpose: AM/PM wave adjustments for fair SG comparison.
# If the AM wave plays in calm conditions and the PM wave plays in 30mph 
# wind, comparing their raw SG is misleading. Weather data enables 
# wave-condition normalization.
#
# Data source: Open-Meteo Archive API
#   - Hourly historical weather at any lat/lon
#   - Variables: wind speed, temperature, precipitation, humidity
#   - Free tier: unlimited historical queries
#
# Usage:
#   from data.weather import WeatherClient
#   client = WeatherClient()
#   weather = client.get_tournament_weather(
#       latitude=33.45, longitude=-111.95,
#       start_date="2024-01-25", end_date="2024-01-28"
#   )
#
# ==============================================================================

from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class WeatherClient:
    """
    Client for Open-Meteo historical weather data.
    
    Parameters
    ----------
    settings : Settings
        Project configuration.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.base_url = self.settings.OPENMETEO_BASE_URL
        logger.info("WeatherClient initialized")

    def get_tournament_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch hourly weather for a tournament venue and date range.
        
        Parameters
        ----------
        latitude : float
            Course latitude.
        longitude : float
            Course longitude.
        start_date : str
            Tournament start date (YYYY-MM-DD).
        end_date : str
            Tournament end date (YYYY-MM-DD).
            
        Returns
        -------
        pd.DataFrame
            Hourly weather with columns:
                datetime, temperature_c, wind_speed_mph, wind_gust_mph,
                precipitation_mm, humidity_pct, pressure_hpa
        """
        import requests

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join([
                "temperature_2m",
                "wind_speed_10m",
                "wind_gusts_10m",
                "precipitation",
                "relative_humidity_2m",
                "surface_pressure",
            ]),
            "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("Weather API request failed: %s", e)
            return pd.DataFrame()

        if "hourly" not in data:
            logger.warning("No hourly data in weather response")
            return pd.DataFrame()

        hourly = data["hourly"]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            "temperature_f": hourly.get("temperature_2m", []),
            "wind_speed_mph": hourly.get("wind_speed_10m", []),
            "wind_gust_mph": hourly.get("wind_gusts_10m", []),
            "precipitation_mm": hourly.get("precipitation", []),
            "humidity_pct": hourly.get("relative_humidity_2m", []),
            "pressure_hpa": hourly.get("surface_pressure", []),
        })

        logger.info(
            "Weather data: %s to %s | %d hourly records | "
            "avg wind=%.1f mph, avg temp=%.1f°F",
            start_date, end_date, len(df),
            df["wind_speed_mph"].mean() if len(df) > 0 else 0,
            df["temperature_f"].mean() if len(df) > 0 else 0,
        )

        return df

    def compute_wave_conditions(
        self,
        weather_df: pd.DataFrame,
        am_hours: tuple = (7, 12),
        pm_hours: tuple = (12, 17),
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute average conditions for AM and PM tee-time waves.
        
        PGA Tour typically has two waves:
            AM wave: ~7:00 AM to 12:00 PM
            PM wave: ~12:00 PM to 5:00 PM
            
        Waves flip on day 2 (AM players go PM, PM goes AM).
        
        Parameters
        ----------
        weather_df : pd.DataFrame
            Hourly weather from get_tournament_weather().
        am_hours : tuple
            (start_hour, end_hour) for AM wave.
        pm_hours : tuple
            (start_hour, end_hour) for PM wave.
            
        Returns
        -------
        dict with keys "am" and "pm", each containing:
            avg_wind, avg_temp, avg_precip, difficulty_index
        """
        if weather_df.empty:
            return {"am": {}, "pm": {}}

        df = weather_df.copy()
        df["hour"] = df["datetime"].dt.hour

        results = {}
        for wave_name, (h_start, h_end) in [("am", am_hours), ("pm", pm_hours)]:
            wave_data = df[(df["hour"] >= h_start) & (df["hour"] < h_end)]

            if len(wave_data) == 0:
                results[wave_name] = {}
                continue

            avg_wind = wave_data["wind_speed_mph"].mean()
            avg_temp = wave_data["temperature_f"].mean()
            avg_precip = wave_data["precipitation_mm"].mean()

            # Difficulty index: higher wind + rain + extreme temp = harder
            # Normalized 0-1 scale (rough heuristic)
            wind_factor = min(avg_wind / 30.0, 1.0)
            rain_factor = min(avg_precip / 5.0, 1.0)
            temp_factor = max(0, abs(avg_temp - 72) / 30.0)  # Deviation from ideal 72°F
            difficulty = (0.5 * wind_factor + 0.3 * rain_factor + 0.2 * temp_factor)

            results[wave_name] = {
                "avg_wind_mph": round(avg_wind, 1),
                "avg_temp_f": round(avg_temp, 1),
                "avg_precip_mm": round(avg_precip, 2),
                "difficulty_index": round(difficulty, 3),
            }

        return results

    def compute_wave_adjustment(
        self,
        am_conditions: Dict[str, float],
        pm_conditions: Dict[str, float],
    ) -> float:
        """
        Compute SG adjustment between AM and PM waves.
        
        Positive = AM wave had easier conditions (AM players should be 
        adjusted downward / PM players adjusted upward).
        
        Returns
        -------
        float
            Adjustment in SG units. Add to PM wave scores, subtract from AM.
        """
        if not am_conditions or not pm_conditions:
            return 0.0

        am_diff = am_conditions.get("difficulty_index", 0)
        pm_diff = pm_conditions.get("difficulty_index", 0)

        # Empirical scaling: ~0.5 SG per 0.1 difficulty index difference
        # This is a rough estimate — should be calibrated from data
        adjustment = (pm_diff - am_diff) * 5.0

        return round(adjustment, 3)
