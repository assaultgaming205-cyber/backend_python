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

        # Language detection
        if session.turnId == 1 or re.search(r"[\u0900-\u097f]", raw) or re.search(r"\b(hai|he|hain|bhai|yaar|karo|batao|jana|jaana|se|kal|chalega)\b", raw, re.IGNORECASE):
            session.userLanguage = "hindi" if re.search(r"[\u0900-\u097f]", raw) and not re.search(r"[a-zA-Z]", raw) else "hinglish"
        if re.search(r"\b(in english|english mein|english please)\b", lower):
            session.userLanguage = "english"
        elif re.search(r"\b(hindi mein|hindi me|हिंदी में)\b", lower):
            session.userLanguage = "hindi"

        # Cancel / Reset
        if re.search(r"\b(new trip|start over|reset|clear|shuru se|cancel|chodo|stop)\b|रद्द|नया ट्रिप", lower):
            session_id = session.sessionId
            lang = session.userLanguage
            new_session = create_initial_session(session_id)
            new_session.userLanguage = lang
            new_session.turnId = session.turnId
            new_session.lastUserIntent = "cancel_reset"
            return ParseResult(new_session, next_missing_slot="destination", ready_for_search=False)

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

        # 2. Route extraction
        orig, dest, reg = LocationResolver.extract_route(normalized_spoken, mode=session.mode)
        dest_name_val = getattr(getattr(session, "destination", None), "name", "") or ""
        orig_name_val = getattr(getattr(session, "origin", None), "name", "") or ""
        if not orig and dest and (session.lastAskedSlot == "origin" or (session.destination and not session.origin and dest.name.lower() != dest_name_val.lower())):
            if not re.search(r"(?:to|towards|visit|jana|jaana|jane|jaane|ko|ke\s*liye|k\s*liye|for)\b", norm_lower):
                orig = dest
                dest = None
        elif not dest and orig and (session.lastAskedSlot == "destination" or (session.origin and not session.destination and orig.name.lower() != orig_name_val.lower())):
            if not re.search(r"(?:from|se\b|say\b|से)", norm_lower):
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
        if re.search(r"\b(today|aaj)\b|आज", norm_lower):
            session.dates.departureDate = today_str
            session.dates.isExplicit = True
        elif re.search(r"\b(tomorrow|kal)\b|कल", norm_lower):
            session.dates.departureDate = plus_days(today_str, 1)
            session.dates.isExplicit = True
        elif re.search(r"\b(day after tomorrow|parso|parson)\b|परसों", norm_lower):
            session.dates.departureDate = plus_days(today_str, 2)
            session.dates.isExplicit = True

        # Explicit ISO date: YYYY-MM-DD
        iso_m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", norm_lower)
        if iso_m and is_valid_iso_date(iso_m.group(1)):
            iso_val = iso_m.group(1)
            if iso_val < today_str:
                session.dates.departureDate = None
                session.dates.isExplicit = False
                session.lastAskedSlot = "date"
                val_err = f"{iso_val} to beet chuki hai (past date hai). Kripya aaj ya aage ki kisi date ke liye batayein."
                return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=val_err)
            session.dates.departureDate = iso_val
            session.dates.isExplicit = True

        # Explicit Day + Month: e.g. "28 October", "28th of Oct", "Oct 28", "25th of october"
        month_pattern = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        day_month = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{month_pattern}\b", norm_lower, re.IGNORECASE)
        month_day = re.search(rf"\b{month_pattern}\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b", norm_lower, re.IGNORECASE)

        m_match = day_month or month_day
        if m_match:
            if day_month:
                day_num = int(m_match.group(1))
                month_str = m_match.group(2)[:3].lower()
            else:
                month_str = m_match.group(1)[:3].lower()
                day_num = int(m_match.group(2))

            if 1 <= day_num <= 31 and month_str in MONTHS:
                month_idx = MONTHS.index(month_str) + 1
                year = today_dt.year
                has_explicit_future_year = bool(re.search(r"\b(202[7-9]|203\d|next year|agle saal)\b", norm_lower))
                if has_explicit_future_year:
                    year_m = re.search(r"\b(202\d|203\d)\b", norm_lower)
                    year = int(year_m.group(1)) if year_m else year + 1

                target_dt = datetime(year, month_idx, day_num)
                if not has_explicit_future_year and target_dt < today_dt:
                    session.dates.departureDate = None
                    session.dates.isExplicit = False
                    session.lastAskedSlot = "date"
                    val_err = f"{day_num} {month_str.capitalize()} to beet chuka hai (past date hai). Kripya aaj ya aage ki kisi date ke liye flight search karein (jaise kal ya agle hafte)."
                    return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=val_err)

                session.dates.departureDate = target_dt.strftime("%Y-%m-%d")
                session.dates.isExplicit = True
                session.dates.dateRangeStart = None
                session.dates.dateRangeEnd = None
                session.dates.isDateFlexible = False
        else:
            # Check for day-only (e.g. "25", "25th", "25 ko", "25 tarikh", "of 28", "on 28")
            if not re.search(r"\b\d+(?:st|nd|rd|th)?\s*(?:class|flights?|planes?|tier|ac|people|adults?|persons?|passengers?|log|yatri|rs|inr|₹|bucks)\b", norm_lower, re.IGNORECASE):
                day_m = re.search(r"\b(?:on\s+|of\s+)?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s*(?:tarikh|taarikh|date|ko))?\b", norm_lower, re.IGNORECASE)
                if day_m:
                    day_val = int(day_m.group(1))
                    is_date_slot = session.lastAskedSlot in ["date", "hotel_dates"]
                    is_explicit_marker = bool(re.search(r"(?:tarikh|taarikh|date|ko|\bst\b|\bnd\b|\brd\b|\bth\b|\bof\s+\d|\bon\s+\d)", norm_lower, re.IGNORECASE))
                    is_short = len(norm_lower.split()) <= 4
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
                        target_day_dt = datetime(year, month_idx, day_val)
                        if target_day_dt < today_dt and not has_current:
                            session.dates.departureDate = None
                            session.dates.isExplicit = False
                            session.lastAskedSlot = "date"
                            val_err = f"{day_val} {MONTHS[month_idx - 1].capitalize()} to beet chuka hai (past date hai). Kripya aage ki kisi date ke liye batayein."
                            return ParseResult(session, next_missing_slot="date", ready_for_search=False, validation_error=val_err)
                        session.dates.departureDate = target_day_dt.strftime("%Y-%m-%d")
                        session.dates.isExplicit = True
                        session.dates.dateRangeStart = None
                        session.dates.dateRangeEnd = None
                        session.dates.isDateFlexible = False

        # Ambiguous date transcriptr month name without day)
        # if today is Oct 26, "26 tarikh" should mean Oct 26, not Oct 2026
        
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

        # 4. Passengers
        pax_m = re.search(r"\b(\d+)\s*(?:people|adults?|persons?|passengers?|log|yatri|jana|bande|hum|seats?|tickets?)\b", norm_lower)
        if pax_m:
            session.passengers.adults = max(1, int(pax_m.group(1)))
            session.passengers.isExplicit = True
            if session.lastAskedSlot == "passengers":
                session.lastAskedSlot = "none"
        elif re.search(r"\b(akela|alone|only\s*me|sirf\s*main|single)\b", norm_lower):
            session.passengers.adults = 1
            session.passengers.isExplicit = True
            if session.lastAskedSlot == "passengers":
                session.lastAskedSlot = "none"
        elif re.search(r"\b(dono|both|couple|do\s*log|hum\s*do|hum\s*dono)\b", norm_lower):
            session.passengers.adults = 2
            session.passengers.isExplicit = True
            if session.lastAskedSlot == "passengers":
                session.lastAskedSlot = "none"
        elif session.lastAskedSlot == "passengers" or (not session.passengers.isExplicit and re.match(r"^\s*(\d+|one|two|three|four|five|ek|do|teen|char|chaar)\s*$", norm_lower)):
            # Direct answers to 'Kitne log travel karenge?'
            num_only = re.search(r"\b(\d+)\b", norm_lower)
            if num_only:
                session.passengers.adults = max(1, int(num_only.group(1)))
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

        # 5. Budget
        budget_m = re.search(r"(?:under|budget|max|limit|around|approx|upto)\s*(?:rs\.?|inr|₹)?\s*(\d+)", norm_lower)
        if budget_m:
            session.constraints.maxBudget = float(budget_m.group(1))

        # 6. Cabin / Rail class
        if (
            re.search(r"\b(?:sabhi|sabki|sabka|saari|saare|all|har\s*(?:ek)?)\s*(?:cabin|cabins|class|classes)\b", norm_lower, re.IGNORECASE)
            or re.search(r"\bcompare\s*(?:all|classes|cabins)\b", norm_lower, re.IGNORECASE)
            or re.search(r"\b(?:sabhi|saari|saare|all)\s*(?:classes|cabins)\s*(?:compare|batao|dikhao|dekhna)?\b", norm_lower, re.IGNORECASE)
            or norm_lower.strip() in ["sabhi classes", "sabki classes", "saare classes", "all classes"]
            or re.search(r"\b(?:first\s*(?:aur|and|ya|or|dono)\s*business|business\s*(?:aur|and|ya|or|dono)\s*first)\b", norm_lower, re.IGNORECASE)
        ):
            session.cabinClass = "all"
            session.compareAllCabins = True
            session.cabinClassExplicit = True
        elif re.search(r"\b(premium\s*economy|pe)\b", norm_lower, re.IGNORECASE):
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
            norm_lower,
            re.IGNORECASE,
        ):
            session.cabinClass = "business"
            session.compareAllCabins = False
            session.cabinClassExplicit = True
        elif re.search(
            r"(?:^|[\s\W])(?:"
            r"first\s*class|1st\s*class|pharst\s*class|pharst|first|1st|"
            r"farst|फ़र्स्ट|फर्स्ट"
            r")(?:$|[\s\W])",
            norm_lower,
            re.IGNORECASE,
        ):
            session.cabinClass = "first"
            session.compareAllCabins = False
            session.cabinClassExplicit = True
        elif re.search(
            r"(?:^|[\s\W])(?:"
            r"economy|eco|sasta|normal|standard|"
            r"ikonaami|ikonomi|इकोनॉमी|इकॉनमी|इकॉनामी"
            r")(?:\s*class)?(?:$|[\s\W])",
            norm_lower,
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
