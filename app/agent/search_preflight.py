"""Deterministic date preflight run BEFORE any date-dependent supplier call is scheduled.

Natural-language parsing stays in the existing intent parser / spoken normalizer; this
module only validates and resolves already-parsed ISO dates. Clock is injectable.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from ..common.dates import is_valid_iso_date

IST = "Asia/Kolkata"
MAX_RANGE_DAYS = 7  # bounded fan-out for "next N days" requests
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def today_ist(now_fn: Optional[Callable[[], datetime]] = None, tz: str = IST) -> date:
    now = now_fn() if now_fn else datetime.now(ZoneInfo(tz))
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo(tz))
    return now.astimezone(ZoneInfo(tz)).date()


@dataclass(frozen=True)
class DateDecision:
    status: str  # "ok" | "clarify"
    date: Optional[str] = None
    inferred: bool = False  # True only when we defaulted to tomorrow
    source: str = ""  # "explicit" | "context" | "default"
    message: str = ""


def _pretty(d: date) -> str:
    return f"{d.day} {_MONTHS[d.month - 1]}"


def check_future_iso(value: Optional[str], today: date) -> Optional[str]:
    """None if `value` is a real, non-past ISO date, else a short reason."""
    if not value or not is_valid_iso_date(value):
        return "invalid"
    return "past" if date.fromisoformat(value) < today else None


def resolve_flight_date(
    explicit: Optional[str],
    context_date: Optional[str],
    today: date,
    ambiguous: bool = False,
) -> DateDecision:
    """Pick the single travel date for a flight/train search, or ask for clarification."""
    if ambiguous:
        return DateDecision("clarify", message="Kaunsi date? Please pick the travel date.")
    if explicit:
        why = check_future_iso(explicit, today)
        if why:
            return DateDecision("clarify", message=f"That date is {why}; please give a valid upcoming travel date.")
        return DateDecision("ok", explicit, False, "explicit")
    if context_date and check_future_iso(context_date, today) is None:
        return DateDecision("ok", context_date, False, "context")
    tomorrow = today + timedelta(days=1)
    return DateDecision(
        "ok", tomorrow.isoformat(), True, "default",
        f"Aapne date nahi batayi, toh main kal, {_pretty(tomorrow)} ke flights check kar raha hoon.",
    )


def expand_range(start: str, days: int, today: date) -> Optional[List[str]]:
    """Exact calendar dates for an N-day window starting at `start`; None if unusable."""
    if not 1 <= days <= MAX_RANGE_DAYS or check_future_iso(start, today):
        return None
    s = date.fromisoformat(start)
    return [(s + timedelta(days=i)).isoformat() for i in range(days)]


def check_hotel_dates(check_in: Optional[str], check_out: Optional[str], today: date) -> Optional[str]:
    """None when both dates are valid and ordered; else a reason. Never invents a stay length."""
    if not check_in or not check_out:
        return "missing_dates"
    if check_future_iso(check_in, today) or check_future_iso(check_out, today):
        return "invalid_dates"
    return None if date.fromisoformat(check_out) > date.fromisoformat(check_in) else "checkout_not_after_checkin"


def resolve_next_week_range(today: date) -> List[str]:
    """Deterministic next calendar week policy: Monday to Sunday (up to 7 days)."""
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # next Monday
    next_monday = today + timedelta(days=days_until_monday)
    return [(next_monday + timedelta(days=i)).isoformat() for i in range(7)]


def resolve_multi_date_range(
    text: str,
    context_text: Optional[str] = None,
    session_dates: Optional[Any] = None,
    today: Optional[date] = None,
) -> Optional[List[str]]:
    """Resolves date ranges from user instruction, conversation context, or calendar-week policy."""
    import re
    if today is None:
        today = today_ist()

    norm_text = text.lower()
    month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    # 1. Explicit range in text, e.g. "26 se 31 october", "26-31 oct", "26 to 31 oct"
    range_match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:se|to|-|–|through|til|till|aur|and|&)\s*(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?(" + "|".join(month_names) + r")[a-z]*",
        norm_text,
        re.IGNORECASE,
    )
    if range_match:
        d1 = int(range_match.group(1))
        d2 = int(range_match.group(2))
        month_idx = month_names.index(range_match.group(3).lower()[:3]) + 1
        year = today.year
        if d1 <= d2:
            out = []
            for d in range(d1, min(d2 + 1, d1 + MAX_RANGE_DAYS)):
                try:
                    dt = date(year, month_idx, d)
                    if dt >= today:
                        out.append(dt.isoformat())
                except ValueError:
                    continue
            if out:
                return out

    # 2. Multi-date request confirmation: "next week sara he dhond ke bata do", "all of next week", etc.
    multi_intent = bool(
        re.search(
            r"\b(next week|agle hafte|sara\s+he|saara\s+hi|sara\s+dhond|saara\s+dhund|dhond\s+ke\s+bata|dhund\s+ke\s+bata|all\s+of\s+next|whole\s+week|entire\s+week|pure\s+hafte|saare\s+din)\b",
            norm_text,
            re.IGNORECASE,
        )
    )
    if multi_intent:
        # Check if conversation context previously described a range (e.g. "26–31 October")
        if context_text:
            ctx_match = re.search(
                r"\b(\d{1,2})\s*[-–to aur se]+\s*(\d{1,2})\s+(" + "|".join(month_names) + r")[a-z]*",
                context_text,
                re.IGNORECASE,
            )
            if ctx_match:
                d1 = int(ctx_match.group(1))
                d2 = int(ctx_match.group(2))
                m_idx = month_names.index(ctx_match.group(3).lower()[:3]) + 1
                year = today.year
                out = []
                for d in range(d1, min(d2 + 1, d1 + MAX_RANGE_DAYS)):
                    try:
                        dt = date(year, m_idx, d)
                        if dt >= today:
                            out.append(dt.isoformat())
                    except ValueError:
                        continue
                if out:
                    return out

        # Check existing session range
        if session_dates and getattr(session_dates, "dateRangeStart", None) and getattr(session_dates, "dateRangeEnd", None):
            try:
                s_dt = date.fromisoformat(session_dates.dateRangeStart)
                e_dt = date.fromisoformat(session_dates.dateRangeEnd)
                if s_dt >= today and s_dt <= e_dt:
                    diff = (e_dt - s_dt).days + 1
                    return [(s_dt + timedelta(days=i)).isoformat() for i in range(min(diff, MAX_RANGE_DAYS))]
            except Exception:
                pass

        return resolve_next_week_range(today)

    return None


@dataclass(frozen=True)
class PreflightDecision:
    ready: bool
    missing_field: Optional[str] = None  # "origin" | "destination" | "date"
    clarification_card: Optional[dict] = None
    resolved_date: Optional[str] = None
    resolved_dates: Optional[List[str]] = None
    inferred_date: bool = False
    status_message: str = ""


def validate_flight_preflight(
    session: Any,
    today: date,
    last_user_text: str = "",
    turn_id: str = "",
) -> PreflightDecision:
    """Validate route, date, and passenger preconditions before flight/train search is dispatched."""
    import re
    from .clarification import normalize_clarification, date_options

    is_train = getattr(session, "mode", "flight") == "train"

    # 1. Check for unrecognized candidate locations (e.g. 'Lise')
    pending = getattr(session, "pendingClarification", "") or ""
    if pending.startswith("unrecognized_origin:"):
        cand = pending.split(":", 1)[1]
        options = (
            [
                {"label": "New Delhi (NDLS)", "value": "New Delhi"},
                {"label": "Mumbai Central (MMCT)", "value": "Mumbai"},
                {"label": "Bengaluru (SBC)", "value": "Bengaluru"},
            ]
            if is_train
            else [
                {"label": "Delhi (DEL)", "value": "Delhi"},
                {"label": "Mumbai (BOM)", "value": "Mumbai"},
                {"label": "Bengaluru (BLR)", "value": "Bengaluru"},
            ]
        )
        card = normalize_clarification(
            {
                "field": "origin",
                "question": f"'{cand.capitalize()}' koi jaani-pehchaani city nahi lag rahi. Aap kahan se travel karna chahte ho?",
                "options": options[:3],
                "allowCustom": True,
            },
            turn_id,
        )
        return PreflightDecision(ready=False, missing_field="origin", clarification_card=card)

    if pending.startswith("unrecognized_destination:"):
        cand = pending.split(":", 1)[1]
        options = (
            [
                {"label": "Jaipur (JP)", "value": "Jaipur"},
                {"label": "Mumbai Central (MMCT)", "value": "Mumbai"},
                {"label": "Goa (MAO)", "value": "Goa"},
            ]
            if is_train
            else [
                {"label": "Jaipur (JAI)", "value": "Jaipur"},
                {"label": "Mumbai (BOM)", "value": "Mumbai"},
                {"label": "Goa (GOI)", "value": "Goa"},
            ]
        )
        card = normalize_clarification(
            {
                "field": "destination",
                "question": f"'{cand.capitalize()}' koi jaani-pehchaani destination nahi lag rahi. Aap kahan travel karna chahte ho?",
                "options": options[:3],
                "allowCustom": True,
            },
            turn_id,
        )
        return PreflightDecision(ready=False, missing_field="destination", clarification_card=card)

    # 2. Missing destination
    if not getattr(session, "destination", None):
        options = (
            [
                {"label": "Jaipur (JP)", "value": "Jaipur"},
                {"label": "Mumbai (MMCT)", "value": "Mumbai"},
                {"label": "Goa (MAO)", "value": "Goa"},
            ]
            if is_train
            else [
                {"label": "Goa (GOI)", "value": "Goa"},
                {"label": "Mumbai (BOM)", "value": "Mumbai"},
                {"label": "Jaipur (JAI)", "value": "Jaipur"},
            ]
        )
        orig_val = getattr(getattr(session, "origin", None), "name", None) or ""
        filtered_opts = [o for o in options if o["value"].lower() != orig_val.lower()][:3]
        card = normalize_clarification(
            {
                "field": "destination",
                "question": "Aap kahan travel karna chahte ho?",
                "options": filtered_opts,
                "allowCustom": True,
            },
            turn_id,
        )
        return PreflightDecision(ready=False, missing_field="destination", clarification_card=card)

    # 3. Missing origin
    if not getattr(session, "origin", None):
        hubs = (
            [
                {"label": "New Delhi (NDLS)", "value": "New Delhi"},
                {"label": "Mumbai (MMCT)", "value": "Mumbai"},
                {"label": "Bengaluru (SBC)", "value": "Bengaluru"},
                {"label": "Kolkata (HWH)", "value": "Kolkata"},
            ]
            if is_train
            else [
                {"label": "Delhi (DEL)", "value": "Delhi"},
                {"label": "Mumbai (BOM)", "value": "Mumbai"},
                {"label": "Bengaluru (BLR)", "value": "Bengaluru"},
                {"label": "Kolkata (CCU)", "value": "Kolkata"},
            ]
        )
        dest_val = getattr(getattr(session, "destination", None), "name", None) or ""
        filtered_hubs = [h for h in hubs if h["value"].lower() != dest_val.lower()][:3]
        card = normalize_clarification(
            {
                "field": "origin",
                "question": "Aap kahan se travel karna chahte ho?",
                "options": filtered_hubs,
                "allowCustom": True,
            },
            turn_id,
        )
        return PreflightDecision(ready=False, missing_field="origin", clarification_card=card)

    # 4. Origin and destination identical
    orig_name = (getattr(session.origin, "name", "") or getattr(session.origin, "code", "")).strip().lower()
    dest_name = (getattr(session.destination, "name", "") or getattr(session.destination, "code", "")).strip().lower()
    if orig_name and dest_name and orig_name == dest_name:
        card = normalize_clarification(
            {
                "field": "destination",
                "question": "Origin aur destination alag hone chahiye. Kripya doosri destination batayein.",
                "options": [],
                "allowCustom": True,
            },
            turn_id,
        )
        return PreflightDecision(ready=False, missing_field="destination", clarification_card=card)

    # 5. Date validation
    multi_dates = resolve_multi_date_range(
        last_user_text,
        getattr(session, "lastAskedQuestion", None),
        getattr(session, "dates", None),
        today,
    )
    if multi_dates:
        if getattr(session, "dates", None):
            session.dates.departureDate = multi_dates[0]
            session.dates.dateRangeStart = multi_dates[0]
            session.dates.dateRangeEnd = multi_dates[-1]
            session.dates.isDateFlexible = True
            session.dates.flexibleWindowDays = len(multi_dates)
            session.dates.isExplicit = True
            session.lastAskedSlot = "none"
        date_dec = DateDecision(
            "ok",
            multi_dates[0],
            False,
            "multi_date_range",
            f"Checking flights across {multi_dates[0]} to {multi_dates[-1]}…",
        )
    else:
        month_pattern = r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
        has_month = bool(re.search(month_pattern, last_user_text, re.IGNORECASE))
        has_day_digit = bool(re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b", last_user_text, re.IGNORECASE))
        is_ambiguous = bool(
            re.search(r"\b(purchase\s*october|end\s*of|middle\s*of|aaspas|around|flexible)\b", last_user_text, re.IGNORECASE)
            or (has_month and not has_day_digit and not getattr(getattr(session, "dates", None), "departureDate", None))
        )

        dep_date = getattr(getattr(session, "dates", None), "departureDate", None)
        date_dec = resolve_flight_date(dep_date, None, today, ambiguous=is_ambiguous)
        if date_dec.status == "clarify":
            card = normalize_clarification(
                {
                    "field": "date",
                    "question": date_dec.message or "Aap kis date ko travel karna chahte ho?",
                    "options": date_options(today),
                    "allowCustom": True,
                },
                turn_id,
            )
            return PreflightDecision(ready=False, missing_field="date", clarification_card=card)

    # 6. Passenger count validation (when relevant / booking intent / explicitly missing)
    pax_state = getattr(session, "passengers", None)
    pax_explicit = getattr(pax_state, "isExplicit", False) if pax_state else False
    is_pax_slot = getattr(session, "lastAskedSlot", "none") == "passengers"
    is_booking_intent = bool(re.search(r"\b(book|booking|tickets?|seats?|family|group|passengers?|pax|log|yatri|fare|kiraya|prices?|kitne)\b", last_user_text, re.IGNORECASE))
    if not pax_explicit and (is_pax_slot or is_booking_intent):
        session.lastAskedSlot = "passengers"
        card = normalize_clarification(
            {
                "field": "passengers",
                "question": "Kitne log travel karenge?",
                "options": [
                    {"label": "1 passenger", "value": "1"},
                    {"label": "2 passengers", "value": "2"},
                    {"label": "3 passengers", "value": "3"},
                ],
                "allowCustom": True,
            },
            turn_id,
        )
        return PreflightDecision(ready=False, missing_field="passengers", clarification_card=card)

    return PreflightDecision(
        ready=True,
        resolved_date=date_dec.date,
        resolved_dates=multi_dates if multi_dates else ([date_dec.date] if date_dec.date else None),
        inferred_date=date_dec.inferred,
        status_message=date_dec.message,
    )

