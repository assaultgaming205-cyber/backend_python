import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def today_in_tz(time_zone: str = "Asia/Kolkata") -> str:
    tz = ZoneInfo(time_zone)
    return datetime.now(tz).strftime("%Y-%m-%d")


def weekday_in_tz(time_zone: str = "Asia/Kolkata") -> str:
    tz = ZoneInfo(time_zone)
    return datetime.now(tz).strftime("%A")


def is_valid_iso_date(s: str) -> bool:
    if not s or not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return False
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d") == s
    except Exception:
        return False


def plus_days(day_str: str, n: int) -> str:
    dt = datetime.strptime(day_str, "%Y-%m-%d")
    return (dt + timedelta(days=n)).strftime("%Y-%m-%d")


def format_day_of_week(date_str: str, time_zone: str = "Asia/Kolkata") -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # e.g., 'Wed, Oct 28'
        return dt.strftime("%a, %b %-d")
    except Exception:
        return date_str
