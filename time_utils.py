"""
Timezone helpers for user-local and NBA-local date logic.
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from nbautils import log_warning

DEFAULT_USER_TIMEZONE = "Africa/Accra"
DEFAULT_NBA_TIMEZONE = "America/New_York"


def _safe_zoneinfo(name: str, fallback: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:
        log_warning(f"Invalid timezone '{name}'. Falling back to '{fallback}'.")
        return ZoneInfo(fallback)


def get_user_timezone() -> ZoneInfo:
    tz_name = os.getenv("USER_TIMEZONE", DEFAULT_USER_TIMEZONE)
    return _safe_zoneinfo(tz_name, "UTC")


def get_nba_timezone() -> ZoneInfo:
    tz_name = os.getenv("NBA_TIMEZONE", DEFAULT_NBA_TIMEZONE)
    return _safe_zoneinfo(tz_name, DEFAULT_NBA_TIMEZONE)


def now_in_tz(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)
