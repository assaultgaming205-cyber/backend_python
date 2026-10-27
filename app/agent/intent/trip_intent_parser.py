import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from ...common.dates import today_in_tz, plus_days, is_valid_iso_date
from ..session import TravelSession, SlotName, create_initial_session
from .turn_intent import classify_turn
from .spoken_normalizer import normalize_spoken_travel_text
from ..resolvers.location_resolver import LocationResolver

MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


class ParseResult:
    def __init__(self, session: TravelSession, next_missing_slot: SlotName = "none", ready_for_search: bool = False, validation_error: Optional[str] = None):
        self.updatedSession = session
        self.nextMissingSlot = next_missing_slot
        self.readyForSearch = ready_for_search
        self.validationError = validation_error


class TripIntentParser:
    @staticmethod
    def parse_turn(text: str, session: TravelSession, time_zone: str = "Asia/Kolkata") -> ParseResult:
        raw = text.strip()
        lower = raw.lower()
        normalized_spoken = normalize_spoken_travel_text(raw, time_zone)
        norm_lower = normalized_spoken.lower()
        session.turnId += 1

        prev_search_sig = (
            getattr(session.origin, "code", None),
            getattr(session.destination, "code", None),
            session.dates.departureDate,
            session.dates.returnDate,
            session.dates.dateRangeStart,
            session.dates.dateRangeEnd,
            session.tripType,
            session.passengers.adults,
            session.passengers.children,
            session.passengers.infantsOnLap,
            session.cabinClass,
            session.mode,
        )

        # Language detection
        if session.turnId == 1 or re.search(r"[\u0900-\u097f]", raw) or re.search(r"\b(hai|he|hain|bhai|yaar|karo|batao|jana|jaana|se|kal|chalega)\b", raw, re.IGNORECASE):
            session.userLanguage = "hindi" if re.search(r"[\u0900-\u097f]", raw) and not re.search(r"[a-zA-Z]", raw) else "hinglish"
        if re.search(r"\b(in english|english mein|english please)\b", lower):
            session.userLanguage = "english"
        elif re.search(r"\b(hindi mein|hindi me|हिंदी में)\b", lower):
            session.userLanguage = "hindi"

        # Abandonment / Task Switching ("forget travel. help me write a birthday message", "stop the search", etc.)
        is_abandon = bool(
            re.search(
                r"\b(forget\s+travel|forget\s+(?:the\s+)?(?:flight|trip|search)|stop\s+(?:the\s+)?(?:travel|search|flight|workflow)|"
                r"leave\s+(?:the\s+)?(?:flight|search|it)|not\s+interested\s+anymore|nahi\s+chahiye|mat\s+dhoondo|rehne\s+do|"
                r"cancel\s+(?:the\s+)?(?:search|trip|plan)|let'?s\s+do\s+something\s+else|i\s+don'?t\s+want\s+to\s+travel|"
                r"drop\s+the\s+trip|abandon\s+(?:the\s+)?(?:search|trip)|help\s+me\s+write\s+a)\b",
                lower,
                re.IGNORECASE,
            )
        )
        if is_abandon:
            session.workflowState = "FLOW_ABANDONED"
            session.abandoned = True
            session.lastUserIntent = "abandon"
            session.lastAskedSlot = "none"
            session.pendingClarification = None
            return ParseResult(session, next_missing_slot="none", ready_for_search=False)

        # Hard session reset ("new trip", "start over", "reset", "shuru se")
        if re.search(r"\b(new\s+trip|start\s+over|reset|clear\s+session|shuru\s+se)\b|रद्द|नया ट्रिप", lower):
            session_id = session.sessionId
            lang = session.userLanguage
            new_session = create_initial_session(session_id)
            new_session.userLanguage = lang
            new_session.turnId = session.turnId
            new_session.lastUserIntent = "cancel_reset"
            return ParseResult(new_session, next_missing_slot="destination", ready_for_search=False)

        # Pure off-topic / distraction detection (so weather/capital cities never overwrite travel state)
        has_travel_action = bool(
            re.search(
                r"\b(flight|flights|train|trains|hotel|hotels|fly|flying|airfare|ticket|tickets|book|booking|"
                r"fare|seat|seats|seatmap|rule|rules|cancellation|refund|baggage|luggage|cabin|economy|business|"
                r"first\s+class|adult|adults|passenger|passengers|child|children|infant|infants|nonstop|non-stop|"
                r"direct|connecting|cheapest|fastest|origin|destination|depart|departure|return|roundtrip|round\s+trip|"
                r"one\s*way|one-way|change|replace|switch|modify|instead\s+of|select|choose|review|option)\b|"
                r"फ्लाइट|विमान|ट्रेन|रेल|होटल|शोध",
                norm_lower,
                re.IGNORECASE,
            )
        )
        if session.lastUserIntent == "distraction":
            session.lastUserIntent = None
        is_pure_distraction = bool(
            not has_travel_action
            and re.search(
                r"\b(what\s+is\s+the\s+(?:weather|capital|population|currency|time|temperature|meaning|formula|score)|"
                r"how\s+is\s+the\s+weather|weather\s+(?:like\s+)?in|capital\s+of|who\s+(?:won|is\s+the\s+president|is\s+the\s+prime\s+minister|is\s+the\s+ceo|invented)|"
                r"tell\s+me\s+a\s+joke|joke\s+sunao|recipe\s+for|how\s+to\s+(?:cook|make|bake)|write\s+a\s+(?:poem|code|script|function|birthday|story|essay)|"
                r"translate\s+this|solve\s+this|what\s+is\s+\d+\s*[\+\-\*\/x]\s*\d+|how\s+much\s+is\s+\d+%\s*gst|"
                r"explain\s+quantum|quantum\s+computing|stock\s+price|bitcoin\s+price|horoscope)\b",
                lower,
                re.IGNORECASE,
            )
        )
        if is_pure_distraction:
            session.lastUserIntent = "distraction"
            return ParseResult(session, next_missing_slot=session.lastAskedSlot, ready_for_search=False)

        # If session was previously abandoned and this turn has no travel action or city, stay abandoned
        if session.abandoned and not has_travel_action:
            orig_probe, dest_probe, _ = LocationResolver.extract_route(normalized_spoken, mode=session.mode)
            if not orig_probe and not dest_probe:
                session.lastUserIntent = "abandon_followup"
                return ParseResult(session, next_missing_slot="none", ready_for_search=False)
        if has_travel_action:
            session.abandoned = False

        # 1. Mode determination
        flexible_mode = bool(re.search(r"\b(any|either|both|koi bhi|kuch bhi|chalega|koi sa bhi|kisi se bhi|kisi se chalega|koi bhi chalega)\b", norm_lower))
        compare = (
            session.lastUserIntent == "compare_modes" or
            re.search(r"\b(?:train|rail|railway).*(?:ya|or|versus|vs|better|se|phir|sasta|sasti|cheapest|cheap|me se|mein se).*(?:flight|flights|plane|planes|air|airways)\b", norm_lower) or
            re.search(r"\b(?:flight|flights|plane|planes|air|airways).*(?:ya|or|versus|vs|better|se|phir|sasta|sasti|cheapest|cheap|me se|mein se).*(?:train|rail|railway)\b", norm_lower) or
            re.search(r"\b(compare both|dono compare|both modes|train ya plane|plane ya train|train ya phir plane|plane ya phir train|train ya flight|flight ya train|dono mein se sasta|dono me se sasta|kisse jau|kis se jau)\b", norm_lower)
        )

        if compare:
            session.mode = "any"
            session.modePolicy = "both"
            session.lastUserIntent = "compare_modes"
        elif flexible_mode and (session.lastAskedSlot == "mode" or re.search(r"chaleg|travel|mode", norm_lower)):
            session.mode = "any"
            session.modePolicy = "flight_first"
        elif re.search(r"\b(flight|flights|plane|planes|airplane|airplanes|airline|airlines|fly|flying|airfare)\b|फ्लाइट|विमान", norm_lower):
            session.mode = "flight"
            session.railClass = None
            session.railClassExplicit = False
            if session.currentTask not in ["hotel", "full_trip"]:
                session.currentTask = "transport"
        elif re.search(r"\b(train|trains|rail|railway|railways|irctc|rajdhani|shatabdi|vande\s*bharat|sleeper|3ac|2ac|1ac)\b|ट्रेन|रेल", norm_lower):
            session.mode = "train"
            if session.currentTask not in ["hotel", "full_trip"]:
                session.currentTask = "transport"
        elif re.search(r"\b(hotel|hotels|resort|resorts|room|rooms|stay|stays|homestay|villa)\b|होटल", norm_lower):
            session.currentTask = "hotel"

        # 2. Route extraction (with contextual replacement support e.g. "Actually, change Delhi to Bengaluru")
        orig, dest, reg = None, None, None
        contextual_replace_matched = False
        if session.origin or session.destination:
            rep_m = re.search(
                r"\b(?:change|switch|replace|modify|update)\s+(?:from\s+)?([a-zA-Z\u0900-\u097f]+(?:\s+[a-zA-Z\u0900-\u097f]+)?)\s+(?:to|with|into|by)\s+([a-zA-Z\u0900-\u097f]+(?:\s+[a-zA-Z\u0900-\u097f]+)?)\b",
                normalized_spoken,
                re.IGNORECASE,
            )
            if not rep_m:
                rep_m = re.search(
                    r"\binstead\s+of\s+([a-zA-Z\u0900-\u097f]+(?:\s+[a-zA-Z\u0900-\u097f]+)?)\s*,?\s*(?:go\s+to|make\s+it|fly\s+to|use|take|let'?s\s+do|chalo|kar\s+do)?\s*([a-zA-Z\u0900-\u097f]+(?:\s+[a-zA-Z\u0900-\u097f]+)?)\b",
                    normalized_spoken,
                    re.IGNORECASE,
                )
            if rep_m and not re.search(r"\b(origin|destination|departure|arrival)\b", rep_m.group(1), re.IGNORECASE):
                old_cand = rep_m.group(1).strip()
                new_cand = rep_m.group(2).strip()
                old_loc = LocationResolver.resolve_city(old_cand)
                new_loc = LocationResolver.resolve_city(new_cand) or LocationResolver.resolve_city(new_cand.split()[0])
                if old_loc and new_loc:
                    if session.destination and (
                        old_loc.code == session.destination.code or old_loc.name.lower() == session.destination.name.lower()
                    ):
                        dest = new_loc
                        contextual_replace_matched = True
                    elif session.origin and (
                        old_loc.code == session.origin.code or old_loc.name.lower() == session.origin.name.lower()
                    ):
                        orig = new_loc
                        contextual_replace_matched = True

        if not contextual_replace_matched:
            orig, dest, reg = LocationResolver.extract_route(normalized_spoken, mode=session.mode)
        dest_name_val = getattr(getattr(session, "destination", None), "name", "") or ""
        orig_name_val = getattr(getattr(session, "origin", None), "name", "") or ""
        if not contextual_replace_matched:
            if not orig and dest and (session.lastAskedSlot == "origin" or (session.destination and not session.origin and dest.name.lower() != dest_name_val.lower())):
                if not re.search(r"(?:to|towards|visit|jana|jaana|jane|jaane|ko|ke\s*liye|k\s*liye|for|destination|arrival)\b", norm_lower):
                    orig = dest
                    dest = None
            elif not dest and orig and (session.lastAskedSlot == "destination" or (session.origin and not session.destination and orig.name.lower() != orig_name_val.lower())):
                if not re.search(r"(?:from|se\b|say\b|से|origin|departure)", norm_lower):
                    dest = orig
                    orig = None

        # Check for unrecognized candidate location (e.g. 'Lise se Mumbai' or replying 'Lise')
        if not orig and not session.origin:
            m_orig = re.search(r"\b([a-zA-Z\u0900-\u097F]{2,25})\s+(?:se|say|से)\b", norm_lower)
            if m_orig:
                cand = m_orig.group(1).strip()
                stopwords = {"aaj", "kal", "parso", "subah", "shaam", "raat", "delhi", "mumbai", "train", "flight", "bus", "car", "kaha", "kahan", "yahan", "wahan", "flight", "plane", "seat"}
                if cand not in stopwords and not LocationResolver.resolve_city(cand):
                    session.pendingClarification = f"unrecognized_origin:{cand}"
            elif session.lastAskedSlot == "origin" and len(norm_lower.split()) <= 4:
                clean_cand = re.sub(r"[.!?]", "", norm_lower).strip()
                if clean_cand and not LocationResolver.resolve_city(clean_cand):
                    session.pendingClarification = f"unrecognized_origin:{clean_cand}"

        if not dest and not session.destination:
            m_dest = re.search(r"\b([a-zA-Z\u0900-\u097F]{2,25})\s+(?:ke\s*liye|k\s*liye|ko|jana|jaana)\b", norm_lower)
            if m_dest:
                cand = m_dest.group(1).strip()
                stopwords = {"ghoomne", "aaj", "kal", "parso", "travel", "chutti", "flight", "train"}
                if cand not in stopwords and not LocationResolver.resolve_city(cand):
                    session.pendingClarification = f"unrecognized_destination:{cand}"
            elif session.lastAskedSlot == "destination" and len(norm_lower.split()) <= 4:
                clean_cand = re.sub(r"[.!?]", "", norm_lower).strip()
                if clean_cand and not LocationResolver.resolve_city(clean_cand):
                    session.pendingClarification = f"unrecognized_destination:{clean_cand}"

        if orig:
            session.origin = orig
            if session.pendingClarification and session.pendingClarification.startswith("unrecognized_origin"):
                session.pendingClarification = None
        if dest:
            session.destination = dest
            if session.pendingClarification and session.pendingClarification.startswith("unrecognized_destination"):
                session.pendingClarification = None
        if reg:
            session.destinationRegion = reg

        # 3. Dates extraction
        today_str = today_in_tz(time_zone)
        today_dt = datetime.strptime(today_str, "%Y-%m-%d")
        month_pattern = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

        # Check for past relative keywords ("yesterday", "beeta hua kal", "last week")
        if re.search(r"\b(yesterday|beeta\s+hua\s+kal|last\s+week|last\s+month|pichle\s+hafte)\b", norm_lower):
            session.dates.departureDate = None
            session.dates.isExplicit = False
            session.lastAskedSlot = "date"
            session.invalidate_search_state()
            val_err = "Past dates are not allowed. Please choose today or a future travel date."
            return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=val_err)

        # Trip type & return date extraction
        if re.search(r"\b(round\s*trip|roundtrip|return\s*trip|two[\s-]*way|both\s+ways|aane\s+jaane)\b", norm_lower):
            session.tripType = "roundtrip"
        elif re.search(r"\b(one[\s-]*way|1[\s-]*way|ek\s+taraf|single\s+way)\b", norm_lower):
            session.tripType = "oneway"
            session.dates.returnDate = None

        ret_iso_m = re.search(
            r"\b(?:return(?:ing)?(?:\s+date|\s+on)?|back\s+on|wapas(?:\s+wali\s+date)?)\s*(?:on|is|to|ko|:)?\s*(\d{4}-\d{2}-\d{2})\b",
            norm_lower,
            re.IGNORECASE,
        )
        if ret_iso_m and is_valid_iso_date(ret_iso_m.group(1)):
            session.dates.returnDate = ret_iso_m.group(1)
            session.tripType = "roundtrip"

        ret_m = re.search(
            rf"\b(?:return(?:ing)?(?:\s+date|\s+on)?|back\s+on|wapas(?:\s+wali\s+date)?)\s*(?:on|is|to|ko|:)?\s*(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{month_pattern}\b",
            norm_lower,
            re.IGNORECASE,
        )
        if not ret_m:
            ret_m = re.search(
                rf"\b(?:return(?:ing)?(?:\s+date|\s+on)?|back\s+on|wapas(?:\s+wali\s+date)?)\s*(?:on|is|to|ko|:)?\s*{month_pattern}\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b",
                norm_lower,
                re.IGNORECASE,
            )
            if ret_m:
                r_month_str = ret_m.group(1)[:3].lower()
                r_day_num = int(ret_m.group(2))
            else:
                r_month_str, r_day_num = "", 0
        else:
            r_day_num = int(ret_m.group(1))
            r_month_str = ret_m.group(2)[:3].lower()

        if r_month_str in MONTHS and 1 <= r_day_num <= 31:
            try:
                r_dt = datetime(today_dt.year, MONTHS.index(r_month_str) + 1, r_day_num)
                if r_dt < today_dt:
                    r_dt = datetime(today_dt.year + 1, MONTHS.index(r_month_str) + 1, r_day_num)
                session.dates.returnDate = r_dt.strftime("%Y-%m-%d")
                session.tripType = "roundtrip"
            except ValueError:
                pass
        ret_m = ret_m or ret_iso_m

        # Strip return-date clause from norm_lower_dep so return-only updates don't overwrite departureDate!
        norm_lower_dep = re.sub(
            rf"\b(?:change\s+|set\s+|make\s+|update\s+)?(?:the\s+)?(?:return(?:ing)?(?:\s+date|\s+on)?|back\s+on|wapas(?:\s+wali\s+date)?)\s*(?:on|is|to|ko|:)?\s*(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{month_pattern}|{month_pattern}\s+(?:the\s+)?\d{{1,2}}(?:st|nd|rd|th)?)\b",
            " ",
            norm_lower,
            flags=re.IGNORECASE,
        )

        # Hotel dates extraction (Check-in, Check-out, Nights)
        is_hotel_task = (session.currentTask == "hotel") or bool(re.search(r"\b(hotel|hotels|resort|resorts|room|rooms|stay|stays|homestay|villa|check\s*in|check\s*out)\b|होटल", norm_lower))
        if is_hotel_task:
            session.currentTask = "hotel"
            if not session.destination and not session.hotel.destination:
                c = LocationResolver.resolve_city(norm_lower)
                if c:
                    session.destination = c
                    session.hotel.destination = c.name
            elif session.destination and not session.hotel.destination:
                session.hotel.destination = session.destination.name

            # 1. Affirmative response to previous hotel question/confirmation
            is_affirmative = bool(re.search(r"^(?:haa?|haan|yes|yup|yeah|sure|ok|okay|theek\s+hai|kar\s+do|chale\s*ga|confirm)[.!?]*$", norm_lower))
            if is_affirmative:
                if session.hotel.checkIn and session.hotel.checkOut:
                    session.hotel.datesExplicit = True
                elif session.lastAskedQuestion:
                    q_dates = list(re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{month_pattern}(?:\s+(\d{{4}}))?\b", session.lastAskedQuestion, re.IGNORECASE))
                    if len(q_dates) >= 2:
                        y_val = next((int(m.group(3)) for m in q_dates if m.group(3)), today_dt.year)
                        m1_str = q_dates[0].group(2)[:3].lower()
                        m2_str = q_dates[1].group(2)[:3].lower()
                        if m1_str in MONTHS and m2_str in MONTHS:
                            d1 = int(q_dates[0].group(1))
                            d2 = int(q_dates[1].group(1))
                            m1_idx = MONTHS.index(m1_str) + 1
                            m2_idx = MONTHS.index(m2_str) + 1
                            dt1 = datetime(y_val, m1_idx, d1)
                            dt2 = datetime(y_val, m2_idx, d2)
                            if dt2 < dt1:
                                dt2 = datetime(y_val + 1, m2_idx, d2)
                            session.hotel.checkIn = dt1.strftime("%Y-%m-%d")
                            session.hotel.checkOut = dt2.strftime("%Y-%m-%d")
                            session.hotel.nights = max(1, (dt2 - dt1).days)
                            session.hotel.datesExplicit = True

            # 2. Date Range: e.g. "20 se 28 october 2027" or "20 to 28 oct"
            range_m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:se|to|-)\s*(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{month_pattern}(?:\s+(\d{{4}}))?\b", norm_lower, re.IGNORECASE)
            if range_m:
                d1 = int(range_m.group(1))
                d2 = int(range_m.group(2))
                m_str = range_m.group(3)[:3].lower()
                y_val = int(range_m.group(4)) if range_m.group(4) else today_dt.year
                if m_str in MONTHS and 1 <= d1 <= 31 and 1 <= d2 <= 31:
                    m_idx = MONTHS.index(m_str) + 1
                    dt1 = datetime(y_val, m_idx, d1)
                    dt2 = datetime(y_val, m_idx, d2)
                    if dt2 < dt1:
                        dt2 = datetime(y_val + 1, m_idx, d2)
                    session.hotel.checkIn = dt1.strftime("%Y-%m-%d")
                    session.hotel.checkOut = dt2.strftime("%Y-%m-%d")
                    session.hotel.nights = max(1, (dt2 - dt1).days)
                    session.hotel.datesExplicit = True

            # 3. Two explicit dates: e.g. "check in 20 october 2027 ... check out 28 october"
            all_dates = list(re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{month_pattern}(?:\s+(\d{{4}}))?\b", norm_lower, re.IGNORECASE))
            if len(all_dates) >= 2 and not session.hotel.datesExplicit:
                y_val = next((int(m.group(3)) for m in all_dates if m.group(3)), today_dt.year)
                m1_str = all_dates[0].group(2)[:3].lower()
                m2_str = all_dates[1].group(2)[:3].lower()
                if m1_str in MONTHS and m2_str in MONTHS:
                    d1 = int(all_dates[0].group(1))
                    d2 = int(all_dates[1].group(1))
                    m1_idx = MONTHS.index(m1_str) + 1
                    m2_idx = MONTHS.index(m2_str) + 1
                    dt1 = datetime(y_val, m1_idx, d1)
                    dt2 = datetime(y_val, m2_idx, d2)
                    if dt2 < dt1:
                        dt2 = datetime(y_val + 1, m2_idx, d2)
                    session.hotel.checkIn = dt1.strftime("%Y-%m-%d")
                    session.hotel.checkOut = dt2.strftime("%Y-%m-%d")
                    session.hotel.nights = max(1, (dt2 - dt1).days)
                    session.hotel.datesExplicit = True

            # 4. One date + explicit nights: "20 october for 8 nights"
            nights_m = re.search(r"\b(\d+)\s*(?:nights?|raat|din|days?)\b", norm_lower)
            if len(all_dates) == 1 and nights_m and not session.hotel.datesExplicit:
                y_val = int(all_dates[0].group(3)) if all_dates[0].group(3) else today_dt.year
                m_str = all_dates[0].group(2)[:3].lower()
                if m_str in MONTHS:
                    d1 = int(all_dates[0].group(1))
                    m_idx = MONTHS.index(m_str) + 1
                    n_val = int(nights_m.group(1))
                    dt1 = datetime(y_val, m_idx, d1)
                    dt2 = dt1 + timedelta(days=n_val)
                    session.hotel.checkIn = dt1.strftime("%Y-%m-%d")
                    session.hotel.checkOut = dt2.strftime("%Y-%m-%d")
                    session.hotel.nights = n_val
                    session.hotel.datesExplicit = True

        # Relative keywords
        date_extracted_this_turn = False
        if re.search(r"\b(today|aaj)\b|आज", norm_lower_dep):
            session.dates.departureDate = today_str
            session.dates.isExplicit = True
            date_extracted_this_turn = True
        elif re.search(r"\b(day after tomorrow|parso|parson)\b|परसों", norm_lower_dep):
            session.dates.departureDate = plus_days(today_str, 2)
            session.dates.isExplicit = True
            date_extracted_this_turn = True
        elif re.search(r"\b(tomorrow|kal)\b|कल", norm_lower_dep):
            session.dates.departureDate = plus_days(today_str, 1)
            session.dates.isExplicit = True
            date_extracted_this_turn = True

        # Weekday expressions: "next friday", "this monday", "coming wednesday", "on friday"
        weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        wd_m = re.search(
            r"\b(?:next|this|coming|agle|is|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            norm_lower_dep,
            re.IGNORECASE,
        )
        if not wd_m and session.lastAskedSlot == "date":
            wd_m = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", norm_lower_dep, re.IGNORECASE)
        if wd_m and not date_extracted_this_turn:
            target_wd = weekday_names.index(wd_m.group(1).lower())
            days_ahead = (target_wd - today_dt.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            session.dates.departureDate = plus_days(today_str, days_ahead)
            session.dates.isExplicit = True
            date_extracted_this_turn = True

        # Explicit ISO date: YYYY-MM-DD
        iso_m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", norm_lower_dep)
        if iso_m:
            iso_val = iso_m.group(1)
            if not is_valid_iso_date(iso_val):
                session.dates.departureDate = None
                session.dates.isExplicit = False
                session.lastAskedSlot = "date"
                session.invalidate_search_state()
                return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=f"Invalid calendar date '{iso_val}'.")
            if iso_val < today_str:
                session.dates.departureDate = None
                session.dates.isExplicit = False
                session.lastAskedSlot = "date"
                session.invalidate_search_state()
                val_err = f"{iso_val} to beet chuki hai (past date hai). Kripya aaj ya aage ki kisi date ke liye batayein."
                return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=val_err)
            session.dates.departureDate = iso_val
            session.dates.isExplicit = True
            date_extracted_this_turn = True

        # Strip "from <old date> to " when changing dates so "change date from 10 Nov to 15 Nov" extracts 15 Nov
        norm_lower_dep_clean = re.sub(
            rf"\bfrom\s+\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{month_pattern}\s+to\s+",
            "to ",
            norm_lower_dep,
            flags=re.IGNORECASE,
        )
        norm_lower_dep_clean = re.sub(
            rf"\bfrom\s+{month_pattern}\s+(?:the\s+)?\d{{1,2}}(?:st|nd|rd|th)?\s+to\s+",
            "to ",
            norm_lower_dep_clean,
            flags=re.IGNORECASE,
        )

        # Explicit Day + Month: e.g. "28 October", "28th of Oct", "Oct 28", "November 15"
        day_month = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{month_pattern}\b", norm_lower_dep_clean, re.IGNORECASE)
        month_day = re.search(rf"\b{month_pattern}\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b", norm_lower_dep_clean, re.IGNORECASE)

        m_match = day_month or month_day
        if m_match:
            if day_month:
                day_num = int(m_match.group(1))
                month_str = m_match.group(2)[:3].lower()
            else:
                month_str = m_match.group(1)[:3].lower()
                day_num = int(m_match.group(2))

            if month_str in MONTHS:
                month_idx = MONTHS.index(month_str) + 1
                year = today_dt.year
                explicit_past_year = re.search(r"\b(201\d|202[0-5])\b", norm_lower_dep_clean)
                has_explicit_future_year = bool(re.search(r"\b(202[7-9]|203\d|next year|agle saal)\b", norm_lower_dep_clean))
                if explicit_past_year:
                    year = int(explicit_past_year.group(1))
                elif has_explicit_future_year:
                    year_m = re.search(r"\b(202\d|203\d)\b", norm_lower_dep_clean)
                    year = int(year_m.group(1)) if year_m else year + 1

                try:
                    target_dt = datetime(year, month_idx, day_num)
                except ValueError:
                    session.dates.departureDate = None
                    session.dates.isExplicit = False
                    session.lastAskedSlot = "date"
                    session.invalidate_search_state()
                    return ParseResult(
                        session,
                        next_missing_slot="date",
                        ready_for_search=False,
                        validation_error=f"'{day_num} {month_str.capitalize()}' is not a valid calendar date. Please provide a valid travel date.",
                    )

                if target_dt < today_dt:
                    session.dates.departureDate = None
                    session.dates.isExplicit = False
                    session.lastAskedSlot = "date"
                    session.invalidate_search_state()
                    val_err = f"{day_num} {month_str.capitalize()} to beet chuka hai (past date hai). Kripya aaj ya aage ki kisi date ke liye flight search karein (jaise kal ya agle hafte)."
                    return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=val_err)

                session.dates.departureDate = target_dt.strftime("%Y-%m-%d")
                session.dates.isExplicit = True
                session.dates.dateRangeStart = None
                session.dates.dateRangeEnd = None
                session.dates.isDateFlexible = False
                date_extracted_this_turn = True
        elif not date_extracted_this_turn and not ret_m:
            # Check for day-only (e.g. "25", "25th", "25 ko", "25 tarikh", "of 28", "on 28")
            day_scan_text = re.sub(
                r"(?:#\d+|\(\d+\)|\b(?:trip|journey|case|audit|ref|attempt|filter|book|abandon|zero|timeout|revfail|empty_ds|option|flight|travel)\s*#?\s*\d+\b)",
                "",
                norm_lower_dep,
                flags=re.IGNORECASE,
            ).strip()
            if not re.search(r"\b\d+(?:st|nd|rd|th)?\s*(?:class|flights?|planes?|options?|tier|ac|people|adults?|children|child|kids?|infants?|persons?|passengers?|log|yatri|rs|inr|₹|bucks)\b", day_scan_text, re.IGNORECASE):
                day_m = re.search(r"\b(?:on\s+|of\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s*(?:tarikh|taarikh|date|ko))?\b", day_scan_text, re.IGNORECASE)
                if day_m:
                    day_val = int(day_m.group(1))
                    is_date_slot = session.lastAskedSlot in ["date", "hotel_dates"]
                    is_explicit_marker = bool(re.search(r"(?:tarikh|taarikh|date|ko|\bst\b|\bnd\b|\brd\b|\bth\b|\bof\s+\d|\bon\s+\d)", day_scan_text, re.IGNORECASE))
                    is_short = len(day_scan_text.split()) <= 4
                    has_current = bool(session.dates.departureDate)
                    if 1 <= day_val <= 31 and (
                        (not has_current and (is_date_slot or is_explicit_marker or is_short))
                        or (has_current and is_explicit_marker)
                    ):
                        if has_current:
                            try:
                                curr_dt = datetime.strptime(session.dates.departureDate, "%Y-%m-%d")
                                month_idx = curr_dt.month
                                year = curr_dt.year
                            except Exception:
                                month_idx = today_dt.month
                                year = today_dt.year
                        else:
                            month_idx = today_dt.month
                            year = today_dt.year
                        try:
                            target_day_dt = datetime(year, month_idx, day_val)
                        except ValueError:
                            session.dates.departureDate = None
                            session.dates.isExplicit = False
                            session.lastAskedSlot = "date"
                            session.invalidate_search_state()
                            return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error="Invalid calendar day.")
                        if target_day_dt < today_dt and not has_current:
                            session.dates.departureDate = None
                            session.dates.isExplicit = False
                            session.lastAskedSlot = "date"
                            session.invalidate_search_state()
                            val_err = f"{day_val} {MONTHS[month_idx - 1].capitalize()} to beet chuka hai (past date hai). Kripya aage ki kisi date ke liye batayein."
                            return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=val_err)
                        session.dates.departureDate = target_day_dt.strftime("%Y-%m-%d")
                        session.dates.isExplicit = True
                        session.dates.dateRangeStart = None
                        session.dates.dateRangeEnd = None
                        session.dates.isDateFlexible = False
                        date_extracted_this_turn = True

        # Date change requested without providing a new date (e.g. "nahi yaar date change kar", "change the travel date")
        if (
            not date_extracted_this_turn
            and not ret_m
            and re.search(
                r"\b(date\s+change(?:\s+kar(?:o|na)?)?|change\s+(?:the\s+|my\s+)?(?:travel\s+|departure\s+)?date|date\s+badal\s+do|date\s+badlo|different\s+date)\b",
                norm_lower,
                re.IGNORECASE,
            )
            and not re.search(r"\b(fee|fees|charges?|rules?|policy)\b", norm_lower, re.IGNORECASE)
        ):
            session.dates.departureDate = None
            session.dates.isExplicit = False
            session.lastAskedSlot = "date"
            session.invalidate_search_state()
            return ParseResult(session, next_missing_slot="date", ready_for_search=False)

        # Ambiguous date (month name without day)
        month_found = bool(re.search(month_pattern, norm_lower, re.IGNORECASE))
        has_day = bool(re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b", norm_lower, re.IGNORECASE))
        is_ambiguous_date = bool(
            re.search(r"\b(purchase\s*october|end\s*of\s*month|middle\s*of\s*month)\b", norm_lower, re.IGNORECASE)
            or (month_found and not has_day and not session.dates.departureDate and session.lastAskedSlot in ["date", "hotel_dates", "none"])
        )
        if is_ambiguous_date:
            session.dates.departureDate = None
            session.dates.isExplicit = False
            session.lastAskedSlot = "date"
            return ParseResult(session, next_missing_slot="date", ready_for_search=False)

        # Multi-day / flexible / date ranges
        from ..search_preflight import resolve_multi_date_range
        multi_dates = resolve_multi_date_range(norm_lower, getattr(session, "lastAskedQuestion", None), getattr(session, "dates", None), today_dt.date())
        if multi_dates:
            session.dates.isDateFlexible = True
            session.dates.flexibleWindowDays = len(multi_dates)
            session.dates.dateRangeStart = multi_dates[0]
            session.dates.dateRangeEnd = multi_dates[-1]
            session.dates.departureDate = multi_dates[0]
            session.dates.isExplicit = True
            if session.lastAskedSlot in ["date", "hotel_dates"]:
                session.lastAskedSlot = "none"
        elif re.search(r"\b(flexible|any day|tum suggest karo|you suggest|next week|agle hafte)\b", norm_lower):
            session.dates.isDateFlexible = True
            session.dates.flexibleWindowDays = 7
            if not session.dates.departureDate:
                session.dates.departureDate = plus_days(today_str, 1)

        # 4. Passengers (Adults, Children, Infants + bounds validation)
        pax_text = re.sub(
            r"\bfrom\s+\d+\s*(?:people|adults?|persons?|passengers?|log|yatri|pax)\s+to\s+",
            "to ",
            norm_lower,
            flags=re.IGNORECASE,
        )
        child_m = re.search(r"\b(\d+)\s*(?:children|child|kids?|bacch[aeo]|बच्चे|बच्चा)\b", pax_text, re.IGNORECASE)
        if child_m:
            c_count = int(child_m.group(1))
            if c_count > 8:
                session.invalidate_search_state()
                return ParseResult(session, next_missing_slot="passengers", ready_for_search=False, validation_error="Maximum 8 children are supported per search.")
            session.passengers.children = c_count
            session.passengers.isExplicit = True

        infant_m = re.search(r"\b(\d+)\s*(?:infants?|bab(?:y|ies)|shishu|नवजात)\b", pax_text, re.IGNORECASE)
        if infant_m:
            i_count = int(infant_m.group(1))
            if i_count > 4:
                session.invalidate_search_state()
                return ParseResult(session, next_missing_slot="passengers", ready_for_search=False, validation_error="Maximum 4 infants are supported per search.")
            session.passengers.infantsOnLap = i_count
            session.passengers.isExplicit = True

        pax_m = re.search(r"\b(\d+)\s*(?:people|adults?|persons?|passengers?|log|yatri|jana|bande|hum|seats?|tickets?|pax)\b", pax_text)
        if pax_m:
            raw_adults = int(pax_m.group(1))
            if raw_adults < 1 or raw_adults > 9:
                session.invalidate_search_state()
                return ParseResult(
                    session,
                    next_missing_slot="passengers",
                    ready_for_search=False,
                    validation_error="Passenger count must be between 1 and 9 adults per booking.",
                )
            session.passengers.adults = raw_adults
            session.passengers.isExplicit = True
            if session.lastAskedSlot == "passengers":
                session.lastAskedSlot = "none"
        elif re.search(r"\b(akela|alone|only\s*me|sirf\s*main|single|solo)\b", pax_text):
            session.passengers.adults = 1
            session.passengers.isExplicit = True
            if session.lastAskedSlot == "passengers":
                session.lastAskedSlot = "none"
        elif re.search(r"\b(dono|both|couple|do\s*log|hum\s*do|hum\s*dono)\b", pax_text):
            session.passengers.adults = 2
            session.passengers.isExplicit = True
            if session.lastAskedSlot == "passengers":
                session.lastAskedSlot = "none"
        elif session.lastAskedSlot == "passengers" or (not session.passengers.isExplicit and re.match(r"^\s*(\d+|one|two|three|four|five|ek|do|teen|char|chaar)\s*$", norm_lower)):
            # Direct answers to 'Kitne log travel karenge?'
            num_only = re.search(r"\b(\d+)\b", norm_lower)
            if num_only:
                raw_adults = int(num_only.group(1))
                if raw_adults < 1 or raw_adults > 9:
                    return ParseResult(session, next_missing_slot="passengers", ready_for_search=False, validation_error="Passenger count must be between 1 and 9 adults.")
                session.passengers.adults = raw_adults
                session.passengers.isExplicit = True
                session.lastAskedSlot = "none"
            elif re.search(r"\b(one|ek)\b", norm_lower):
                session.passengers.adults = 1
                session.passengers.isExplicit = True
                session.lastAskedSlot = "none"
            elif re.search(r"\b(two|do)\b", norm_lower):
                session.passengers.adults = 2
                session.passengers.isExplicit = True
                session.lastAskedSlot = "none"
            elif re.search(r"\b(three|teen)\b", norm_lower):
                session.passengers.adults = 3
                session.passengers.isExplicit = True
                session.lastAskedSlot = "none"
            elif re.search(r"\b(four|char|chaar)\b", norm_lower):
                session.passengers.adults = 4
                session.passengers.isExplicit = True
                session.lastAskedSlot = "none"

        # 5. Budget & Preferences (timeOfDay, stops, rankingPreference)
        budget_m = re.search(r"(?:under|budget|max|limit|around|approx|upto)\s*(?:rs\.?|inr|₹)?\s*(\d+)", norm_lower)
        if budget_m:
            session.constraints.maxBudget = float(budget_m.group(1))

        if re.search(r"\b(morning|subah|early\s+morning|savere)\b", norm_lower):
            session.constraints.timeOfDay = "morning"
        elif re.search(r"\b(afternoon|dopahar)\b", norm_lower):
            session.constraints.timeOfDay = "afternoon"
        elif re.search(r"\b(evening|shaam)\b", norm_lower):
            session.constraints.timeOfDay = "evening"
        elif re.search(r"\b(night|raat|late\s+night)\b", norm_lower) and not re.search(r"\b\d+\s*nights?\b", norm_lower):
            session.constraints.timeOfDay = "night"

        if re.search(r"\b(allow\s+connecting|connecting\s+(?:flights?\s+)?(?:ok|allowed|chalega)|with\s+stops|any\s+stops|remove\s+nonstop|change\s+nonstop\s+only\s+to\s+allow\s+connecting)\b", norm_lower):
            session.constraints.stops = None
            session.nonstopOnly = False
        elif re.search(r"\b(non[\s-]*stop|direct\s+flights?(?:\s+only)?|nonstop\s+only|bina\s+stop|seedhi\s+flight)\b", norm_lower):
            session.constraints.stops = 0
            session.nonstopOnly = True

        if re.search(r"\b(fastest|shortest\s+duration|quickest|sabse\s+fast|least\s+time|minimum\s+duration)\b", norm_lower):
            session.rankingPreference = "fastest"
        elif re.search(r"\b(cheapest|lowest\s+(?:fare|price)|sabse\s+sasta|sasti|budget)\b", norm_lower):
            session.rankingPreference = "cheapest"

        # 6. Cabin / Rail class (strip "from <old_cabin> to " so "change Economy to Business" extracts Business)
        cabin_text = re.sub(
            r"\bfrom\s+(?:economy|business|first(?:\s+class)?|premium\s+economy)(?:\s+class)?\s+to\s+",
            "to ",
            norm_lower,
            flags=re.IGNORECASE,
        )
        if (
            re.search(r"\b(?:sabhi|sabki|sabka|saari|saare|all|har\s*(?:ek)?)\s*(?:cabin|cabins|class|classes)\b", cabin_text, re.IGNORECASE)
            or re.search(r"\bcompare\s*(?:all|classes|cabins)\b", cabin_text, re.IGNORECASE)
            or re.search(r"\b(?:sabhi|saari|saare|all)\s*(?:classes|cabins)\s*(?:compare|batao|dikhao|dekhna)?\b", cabin_text, re.IGNORECASE)
            or cabin_text.strip() in ["sabhi classes", "sabki classes", "saare classes", "all classes"]
            or re.search(r"\b(?:first\s*(?:aur|and|ya|or|dono)\s*business|business\s*(?:aur|and|ya|or|dono)\s*first)\b", cabin_text, re.IGNORECASE)
        ):
            session.cabinClass = "all"
            session.compareAllCabins = True
            session.cabinClassExplicit = True
        elif re.search(r"\b(premium\s*economy|pe)\b", cabin_text, re.IGNORECASE):
            session.cabinClass = "premium_economy"
            session.compareAllCabins = False
            session.cabinClassExplicit = True
        elif re.search(
            r"(?:^|[\s\W])(?:"
            r"busineded|busined|businesd|busnid|busineds|bizness|"
            r"business|biz|buisness|bussiness|busines|bussnes|"
            r"bisnes|bijnes|bijnis|biytnes|biyetnes|executive|"
            r"बिजनेस|बिज़नेस|बिज़नेस"
            r")(?:\s*class)?(?:$|[\s\W])",
            cabin_text,
            re.IGNORECASE,
        ):
            session.cabinClass = "business"
            session.compareAllCabins = False
            session.cabinClassExplicit = True
        elif re.search(
            r"(?:^|[\s\W])(?:"
            r"first\s*class|1st\s*class|first\s*cabin|1st\s*cabin|first\s*flights|1st\s*flights|pharst\s*class|pharst|"
            r"farst|फ़र्स्ट\s*क्लास|फर्स्ट\s*क्लास|फ़र्स्ट|फर्स्ट"
            r")(?:$|[\s\W])",
            cabin_text,
            re.IGNORECASE,
        ) or (session.lastAskedSlot in ("class", "cabin") and re.search(r"\b(first|1st)\b", cabin_text, re.IGNORECASE)):
            session.cabinClass = "first"
            session.compareAllCabins = False
            session.cabinClassExplicit = True
        elif re.search(
            r"(?:^|[\s\W])(?:"
            r"economy|eco|normal|standard|"
            r"ikonaami|ikonomi|इकोनॉमी|इकॉनमी|इकॉनामी"
            r")(?:\s*class)?(?:$|[\s\W])",
            cabin_text,
            re.IGNORECASE,
        ):
            session.cabinClass = "economy"
            session.compareAllCabins = False
            session.cabinClassExplicit = True

        if re.search(r"\b(3ac|3rd\s*ac|3\s*ac|3a)\b", norm_lower):
            session.railClass = "3A"
            session.railClassExplicit = True
        elif re.search(r"\b(2ac|2nd\s*ac|2\s*ac|2a)\b", norm_lower):
            session.railClass = "2A"
            session.railClassExplicit = True
        elif re.search(r"\b(sleeper|sl)\b", norm_lower):
            session.railClass = "SL"
            session.railClassExplicit = True

        # Invalidate downstream search/review/seatmap/farerules state if core search parameters changed
        curr_search_sig = (
            getattr(session.origin, "code", None),
            getattr(session.destination, "code", None),
            session.dates.departureDate,
            session.dates.returnDate,
            session.dates.dateRangeStart,
            session.dates.dateRangeEnd,
            session.tripType,
            session.passengers.adults,
            session.passengers.children,
            session.passengers.infantsOnLap,
            session.cabinClass,
            session.mode,
        )
        if curr_search_sig != prev_search_sig and (
            session.lastOffers or session.selectedFareId or session.bookingSessionId or session.activeSearchContextId
        ):
            session.invalidate_search_state()

        # Classify turn intent if not already set
        if not session.lastUserIntent or session.lastUserIntent == "answer_question":
            classified = classify_turn(norm_lower, session.lastUserIntent, has_explicit_route=bool(orig and dest))
            if classified != "answer_question":
                session.lastUserIntent = classified

        return TripIntentParser.evaluate_next_action(session, time_zone)

    @staticmethod
    def evaluate_next_action(session: TravelSession, time_zone: str = "Asia/Kolkata") -> ParseResult:
        today_str = today_in_tz(time_zone)
        if session.dates.departureDate and session.dates.departureDate < today_str:
            session.dates.departureDate = None
            session.dates.isExplicit = False
            session.lastAskedSlot = "date"
            val_err = "Chuni hui tareekh beet chuki hai (past date hai). Kripya aaj ya aage ki koi tareekh batayein."
            return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=val_err)

        if session.currentTask == "hotel":
            if session.hotel.checkIn and session.hotel.checkIn < today_str:
                session.hotel.checkIn = None
                session.hotel.checkOut = None
                session.lastAskedSlot = "hotel_dates"
                val_err = "Hotel check-in tareekh beet chuki hai. Kripya aage ki tareekh batayein."
                return ParseResult(session, next_missing_slot="hotel_dates", ready_for_search=False, validation_error=val_err)
            has_dest = bool(session.destination or session.hotel.destination)
            has_dates = bool(session.hotel.checkIn and session.hotel.checkOut)
            is_ready = has_dest and has_dates
            next_missing = "none" if is_ready else ("destination" if not has_dest else "hotel_dates")
            session.lastAskedSlot = next_missing
            return ParseResult(session, next_missing_slot=next_missing, ready_for_search=is_ready)

        is_flexible_multi_day = bool(
            (session.dates.isDateFlexible or session.lastUserIntent == "flexible_date_inquiry") and
            session.origin and session.destination and
            session.mode in ["train", "flight", "any"]
        )

        has_orig = bool(session.origin)
        has_dest = bool(session.destination)
        has_mode = session.mode != "unknown"

        if session.lastAskedSlot == "date" and not session.dates.departureDate and not session.dates.isDateFlexible:
            return ParseResult(session, next_missing_slot="date", ready_for_search=False)

        # Ready for search as soon as route is known (defaults date to tomorrow and class to Economy if not explicit)
        is_ready = has_orig and has_dest

        next_missing = "none"
        if not has_dest:
            next_missing = "destination"
        elif not has_orig:
            next_missing = "origin"
        elif not has_mode:
            next_missing = "mode"

        session.lastAskedSlot = next_missing
        return ParseResult(session, next_missing_slot=next_missing, ready_for_search=is_ready)
