import re
from datetime import date
import time
import uuid
import logging
import asyncio
from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel

from ..schemas.chat import (
    ChatResults,
    ToolCallSummary,
    NarrationSection,
    empty_results,
)
from ..schemas.flight import FlightOffer
from ..schemas.train import TrainOption
from ..schemas.hotel import HotelOffer
from ..ai.ai_service import AiService, AiMessage
from ..travel.flights.bankg_flight_provider import BankGFlightProvider
from ..travel.trains.makemytrip_rail import MakeMyTripRailProvider
from ..travel.hotels.makemytrip_hotel import MakeMyTripHotelProvider
from .session import TravelSession, create_initial_session, ResolvedLocation
from .session_store import SessionStoreService, session_store
from .resolvers.location_resolver import LocationResolver
from .intent.spoken_normalizer import normalize_spoken_travel_text
from .intent.trip_intent_parser import TripIntentParser, ParseResult
from .intent.turn_intent import classify_turn
from .intent.guard import TravelDomainGuard
from .prompts.travel_buddy_prompt import build_system_prompt_with_context
from .grounding.grounded_reply import build_facts, publish_reply, clean_reply_text, REPLY_SCHEMA
from .grounding.draft_validator import validate_draft
from .grounding.narration import build_narration
from .grounding.sentence_buffer import SentenceBuffer
from .grounding.grounding_gate import GroundingGate
from .tool_executor import ToolExecutor
from ..common.dates import today_in_tz, plus_days
from .search_preflight import resolve_flight_date, validate_flight_preflight, PreflightDecision, resolve_multi_date_range
from .clarification import clarification_store, normalize_clarification
from .flight_selection import get_ledger, select_diverse, _dedupe, offer_key, build_search_context_id
from ..config import settings

logger = logging.getLogger("agent_service")

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_TWO_DATES = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:,|aur|and|ya|or|&|/|-|to)\s*(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")[a-z]*",
    re.IGNORECASE,
)


def _requested_dates(text: str, tz: str) -> List[str]:
    """Dates from '26 aur 27 october' (max 2); [] if the user gave none or one. shortcut: same-month pairs only."""
    for candidate in (normalize_spoken_travel_text(text, tz), text):
        m = _TWO_DATES.search(candidate)
        if m:
            break
    else:
        return []
    today = date.fromisoformat(today_in_tz(tz))
    month = _MONTHS.index(m.group(3).lower()) + 1
    has_explicit_future = bool(re.search(r"\b(202[7-9]|203\d|next year|agle saal)\b", text, re.IGNORECASE))
    year = today.year
    if has_explicit_future:
        year_m = re.search(r"\b(202\d|203\d)\b", text)
        year = int(year_m.group(1)) if year_m else year + 1
    out: List[str] = []
    for d in (m.group(1), m.group(2)):
        try:
            dt = date(year, month, int(d))
        except ValueError:
            continue
        if not has_explicit_future and dt < today:
            return []
        out.append(dt.isoformat())
    return out if len(out) == 2 else []


class AgentResult(BaseModel):
    content: str
    toolCalls: List[ToolCallSummary] = []
    results: ChatResults
    model: str
    sessionId: Optional[str] = None
    narration: List[NarrationSection] = []
    telemetry: Optional[Dict[str, Any]] = None


class AgentService:
    def __init__(
        self,
        ai: Optional[AiService] = None,
        executor: Optional[ToolExecutor] = None,
        store: Optional[SessionStoreService] = None,
        timezone: str = "Asia/Kolkata",
    ):
        self.ai = ai or AiService()
        self.executor = executor or ToolExecutor()
        self.store = store or session_store
        self.timezone = timezone
        self.snapshots: Dict[str, Dict[str, Any]] = {}

    def build_session_from_history(
        self,
        history: List[AiMessage],
        session_id: Optional[str] = None,
    ) -> tuple[TravelSession, ParseResult]:
        stored = self.store.get(session_id) if session_id else None
        session = stored.model_copy(deep=True) if stored else create_initial_session(session_id)

        parse_result = ParseResult(session, next_missing_slot="destination", ready_for_search=False)

        for m in history:
            if m.role == "assistant":
                session.lastAskedQuestion = m.content
            elif m.role == "user":
                content = m.content or ""
                # Skip past trip stories
                if re.search(r"\b(?:last year|last time|visited|went to|gaya tha|pichli baar)\b", content, re.IGNORECASE) and not re.search(r"\b(?:book|search|find|plan|want|chahiye|jana hai)\b", content, re.IGNORECASE):
                    continue
                cleaned = re.sub(r"\b(?:kal|kaal)\s+(?:bhool|bhul|forgot)(?:\s+gaya)?|कल\s+भूल(?:\s+गया)?", "past mistake", content, flags=re.IGNORECASE)
                normalized = normalize_spoken_travel_text(cleaned, self.timezone)
                parse_result = TripIntentParser.parse_turn(normalized, session, self.timezone)

        # Context preservation across turns: extract origin/destination if still missing
        if not session.origin or not session.destination:
            for m in history:
                orig, dest, reg = LocationResolver.extract_route(m.content or "")
                dest_name_val = getattr(getattr(session, "destination", None), "name", "") or ""
                if not orig and dest and session.destination and not session.origin and dest.name.lower() != dest_name_val.lower():
                    if not re.search(r"(?:to|towards|visit|jana|jaana|jane|jaane|ko|ke\s*liye|k\s*liye|for)\b", m.content or "", re.IGNORECASE):
                        orig = dest
                        dest = None
                if not session.origin and orig:
                    session.origin = orig
                if not session.destination and dest:
                    session.destination = dest
                if reg and not session.destinationRegion:
                    session.destinationRegion = reg

        # Mode inference from history if unknown
        if session.mode == "unknown":
            has_train = any(re.search(r"\b(train|trains|rail|railway|irctc|gaadi|gadi)\b|ट्रेन|रेल", m.content or "", re.IGNORECASE) for m in history)
            has_flight = any(re.search(r"\b(flight|flights|plane|planes|aeroplane|airline)\b|फ्लाइट", m.content or "", re.IGNORECASE) for m in history)
            if has_train and not has_flight:
                session.mode = "train"
            elif has_flight and not has_train:
                session.mode = "flight"

        if parse_result and parse_result.validationError:
            return session, parse_result

        parse_result = TripIntentParser.evaluate_next_action(session, self.timezone)
        return session, parse_result

    async def run(
        self,
        history: List[AiMessage],
        session_id: Optional[str] = None,
        model_provider: Optional[str] = None,
        interactive_mode: bool = False,
        character: str = "meera",
        model: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_results: Optional[Callable[[ChatResults], None]] = None,
        on_card: Optional[Callable[[str, Any], None]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
        on_clarification: Optional[Callable[[Dict[str, Any]], None]] = None,
        clarification_answer: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        start_time = time.time()
        results = empty_results()
        tool_calls: List[ToolCallSummary] = []
        narration: List[NarrationSection] = []
        supplier_durations: Dict[str, int] = {}
        llm_ttft_ms: Optional[int] = None

        if not history:
            msg = "Hello! Main aapki travel buddy hoon. Kahan chalne ka plan hai?"
            return AgentResult(
                content=msg,
                toolCalls=[],
                results=results,
                model=model or self.ai.model_name,
                sessionId=session_id,
                narration=[],
            )

        # 1. Clean history & detect noise
        last_msg = next((m for m in reversed(history) if m.role == "user"), None)
        last_user = normalize_spoken_travel_text((last_msg.content or "").strip(), self.timezone) if last_msg else ""

        if not last_user or (len(last_user) <= 1 and not last_user.isdigit() and not clarification_answer):
            msg = (
                "Sunai nahi diya, ek baar phir bolenge?"
                if character == "kabir"
                else "Mujhe sunai nahi diya, ek baar phir bolengi?"
            )
            return AgentResult(
                content=msg,
                toolCalls=[],
                results=results,
                model=self.ai.model_name,
                sessionId=session_id,
                narration=[],
            )

        session, parse_result = self.build_session_from_history(history, session_id)
        today_val = date.fromisoformat(today_in_tz(self.timezone))

        # Handle structured clarification answer if present
        ans_payload = clarification_answer or (getattr(last_msg, "clarification_answer", None) if last_msg else None)
        if ans_payload and isinstance(ans_payload, dict):
            q_id = str(ans_payload.get("questionId") or "")
            t_id = str(ans_payload.get("turnId") or "")
            c_val = str(ans_payload.get("value") or "")
            ans_res = clarification_store.answer(session.sessionId, q_id, t_id, c_val, today_val)
            f_slot = ans_res.get("field") or ans_payload.get("field")
            f_val = ans_res.get("value") or ans_payload.get("value")
            if f_slot and f_val:
                if f_slot == "origin":
                    c_loc = LocationResolver.resolve_city(f_val)
                    if c_loc:
                        session.origin = c_loc
                        session.lastAskedSlot = "none"
                        session.pendingClarification = None
                    else:
                        session.origin = None
                        session.lastAskedSlot = "origin"
                        session.pendingClarification = f"unrecognized_origin:{f_val}"
                elif f_slot == "destination":
                    c_loc = LocationResolver.resolve_city(f_val)
                    if c_loc:
                        session.destination = c_loc
                        session.lastAskedSlot = "none"
                        session.pendingClarification = None
                    else:
                        session.destination = None
                        session.lastAskedSlot = "destination"
                        session.pendingClarification = f"unrecognized_destination:{f_val}"
                elif f_slot == "date":
                    session.dates.departureDate = f_val
                    session.dates.isExplicit = True
                    session.lastAskedSlot = "none"
                elif f_slot == "passengers":
                    try:
                        p_m = re.search(r"\d+", str(f_val))
                        session.passengers.adults = max(1, int(p_m.group(0))) if p_m else 1
                        session.passengers.isExplicit = True
                        session.lastAskedSlot = "none"
                    except Exception:
                        pass
                elif f_slot in ["cabin", "class"]:
                    clean_c = str(f_val).lower().strip()
                    if clean_c in ["economy", "premium_economy", "business", "first", "all"]:
                        session.cabinClass = clean_c
                        session.cabinClassExplicit = True
                        session.lastAskedSlot = "none"

        # 2. Scope Guard Check
        user_lang = session.userLanguage
        has_active_loc = bool(session.destination or session.origin)
        scope_check = TravelDomainGuard.evaluate(
            last_user,
            user_language=user_lang,
            character=character,
            has_active_session_with_location=has_active_loc,
        )
        is_out_of_scope = scope_check.isOutOfScope

        # 3. Detect Mode Comparison (train vs flight / cheapest)
        is_mode_comparison = (
            session.lastUserIntent == "compare_modes" or
            classify_turn(last_user) == "compare_modes" or
            bool(re.search(r"\b(?:train\s*(?:ya|or|vs|aur|phir)\s*(?:flight|plane)|flight\s*(?:ya|or|vs|aur|phir)\s*train|cheapest.*(?:train|flight|plane)|sasta.*(?:train|flight|plane)|compare.*(?:train|flight)|dono.*compare)\b", last_user, re.IGNORECASE))
        )

        if is_mode_comparison:
            session.mode = "any"
            session.modePolicy = "both"
            session.lastUserIntent = "compare_modes"
        elif session.mode == "unknown" and session.origin and session.destination:
            session.mode = "flight"

        # 4. Search Execution Preflight
        turn_id = str(uuid.uuid4())
        preflight = validate_flight_preflight(session, today_val, last_user, turn_id)
        is_transport_turn = (session.mode in ["flight", "train", "any"] or session.currentTask in ["transport", "full_trip"])

        if is_transport_turn and not is_out_of_scope and not preflight.ready:
            if session.sessionId and preflight.clarification_card:
                clarification_store.open(session.sessionId, preflight.clarification_card)
            if on_clarification and preflight.clarification_card:
                on_clarification(preflight.clarification_card)

        actionable = TripIntentParser.evaluate_next_action(session, self.timezone)
        if actionable.validationError and not parse_result.validationError:
            parse_result = actionable
        is_suggest_requested = bool(re.search(r"\b(suggest|sasta kab|flexible date|best day|cheapest day)\b", last_user, re.IGNORECASE))
        is_multi_day = bool((session.dates.isDateFlexible or is_suggest_requested) and session.origin and session.destination)

        outcome = ""
        if parse_result.validationError:
            session.pendingClarification = parse_result.validationError
            outcome = f"INPUT_INVALID: {parse_result.validationError}"

        attempted_hotel = False
        attempted_flight = False
        attempted_train = False
        zero_results_reason = ""
        zero_results_suggestion = ""

        if not is_out_of_scope and (actionable.readyForSearch or is_multi_day) and not parse_result.validationError:
            if on_status:
                on_status("Searching live flights and trains…")

            search_tasks = []
            req_dates = _requested_dates(last_user, self.timezone)
            multi_dates = (
                preflight.resolved_dates
                or resolve_multi_date_range(last_user, session.lastAskedQuestion, session.dates, today_val)
            )
            dep_dates: List[str] = []
            dates_with_offers: List[str] = []
            dates_zero_offers: List[str] = []
            dates_failed: List[str] = []
            date_failures: set[str] = set()


            # Flight Search (BankG airSearch): queries each resolved date independently with honest cabin class
            if preflight.ready and (session.mode in ["flight", "any"] or session.currentTask == "full_trip"):
                dep_dates = multi_dates or req_dates or [preflight.resolved_date or session.dates.departureDate or session.dates.dateRangeStart or plus_days(today_in_tz(self.timezone), 1)]
                if preflight.inferred_date and on_status and preflight.status_message:
                    on_status(preflight.status_message)
                flight_orig = LocationResolver.get_flight_code(session.origin.code or session.origin.name)
                flight_dest = LocationResolver.get_flight_code(session.destination.code or session.destination.name)

                search_ctx_id = build_search_context_id(
                    flight_orig,
                    flight_dest,
                    departure_date=session.dates.departureDate,
                    date_range_start=session.dates.dateRangeStart,
                    date_range_end=session.dates.dateRangeEnd,
                    cabin=session.cabinClass,
                    adults=session.passengers.adults,
                    dates=dep_dates if len(dep_dates) > 1 else None,
                )
                session.activeSearchContextId = search_ctx_id
                results.searchContextId = search_ctx_id

                if session.cabinClass == "premium_economy":
                    attempted_flight = True
                    zero_results_reason = (
                        "BankG API Gateway integration documents ECONOMY, BUSINESS, and FIRST cabin classes only. "
                        "Premium Economy is not supported by this supplier."
                    )
                    zero_results_suggestion = "Aap Economy, Business Class ya First Class me se select kar sakte hain."
                    alt_card = normalize_clarification(
                        {
                            "field": "cabin",
                            "question": "Premium Economy is not supported by BankG. Which cabin class would you like?",
                            "options": [
                                {"label": "Economy", "value": "economy"},
                                {"label": "Business Class", "value": "business"},
                                {"label": "First Class", "value": "first"},
                            ],
                            "allowCustom": False,
                        },
                        turn_id,
                    )
                    if session.sessionId and alt_card:
                        clarification_store.open(session.sessionId, alt_card)
                    if on_clarification and alt_card:
                        on_clarification(alt_card)
                elif session.cabinClass == "all" or session.compareAllCabins:
                    cabins = ["economy", "business", "first"]
                    if on_status:
                        on_status("Checking Economy, Business & First Class flights…")

                    async def run_cabin_search(c: str, d: str):
                        t0 = time.monotonic()
                        try:
                            t = await self.executor.execute(
                                "search_flights",
                                {
                                    "origin": flight_orig,
                                    "destination": flight_dest,
                                    "departureDate": d,
                                    "adults": session.passengers.adults,
                                    "cabin": c,
                                },
                                results,
                            )
                            tool_calls.append(t)
                            supplier_durations["flights"] = int((time.monotonic() - t0) * 1000)
                            if not t.success:
                                date_failures.add(d)
                            for f in results.flights:
                                if not getattr(f, "searchContextId", None):
                                    f.searchContextId = search_ctx_id
                            if on_card and results.flights:
                                on_card("flight", results.flights)
                        except Exception as e:
                            logger.error(f"Cabin flight search error ({c}, {d}): {e}")
                            date_failures.add(d)

                    search_tasks += [run_cabin_search(c, d) for c in cabins for d in dep_dates]
                    attempted_flight = True
                else:
                    async def run_flight_search(d: str):
                        if on_status:
                            on_status(f"Checking flights for {d}…")
                        t0 = time.monotonic()
                        try:
                            t = await self.executor.execute(
                                "search_flights",
                                {
                                    "origin": flight_orig,
                                    "destination": flight_dest,
                                    "departureDate": d,
                                    "adults": session.passengers.adults,
                                    "cabin": session.cabinClass,
                                },
                                results,
                            )
                            tool_calls.append(t)
                            supplier_durations[f"flights_{d}"] = int((time.monotonic() - t0) * 1000)
                            if not t.success:
                                date_failures.add(d)
                            for f in results.flights:
                                if not getattr(f, "searchContextId", None):
                                    f.searchContextId = search_ctx_id
                            if on_card and results.flights:
                                on_card("flight", results.flights)
                        except Exception as e:
                            logger.error(f"Flight search error ({d}): {e}")
                            date_failures.add(d)

                    search_tasks += [run_flight_search(d) for d in dep_dates]
                    attempted_flight = True

            # Train Search (MMT Rail), one search per requested date
            if preflight.ready and session.origin and session.destination and (session.mode in ["train", "any"] or session.currentTask == "full_trip"):
                train_orig = LocationResolver.get_train_station(session.origin.station or session.origin.code or session.origin.name)
                train_dest = LocationResolver.get_train_station(session.destination.station or session.destination.code or session.destination.name)
                train_dates = req_dates or [session.dates.departureDate or today_in_tz(self.timezone)]

                async def run_train_search(d: str):
                    if on_status:
                        on_status(f"Checking trains for {d}…")
                    t0 = time.monotonic()
                    try:
                        t = await self.executor.execute(
                            "search_trains",
                            {
                                "origin": train_orig,
                                "destination": train_dest,
                                "date": d,
                            },
                            results,
                        )
                        tool_calls.append(t)
                        supplier_durations["trains"] = int((time.monotonic() - t0) * 1000)
                        if on_card and results.trains:
                            on_card("train", results.trains)
                    except Exception as e:
                        logger.error(f"Train search error ({d}): {e}")

                search_tasks += [run_train_search(d) for d in train_dates]
                attempted_train = True

            # Hotel Search (MMT Hotel)
            is_hotel_ready = bool(
                (session.currentTask in ["hotel", "full_trip"]) and
                (session.destination or session.hotel.destination) and
                session.hotel.checkIn and
                session.hotel.checkOut
            )
            if is_hotel_ready:
                hotel_dest = session.destination.name if session.destination else (session.hotel.destination or "")

                async def run_hotel_search():
                    t0 = time.monotonic()
                    try:
                        t = await self.executor.execute(
                            "search_hotels",
                            {
                                "destination": hotel_dest,
                                "checkIn": session.hotel.checkIn,
                                "checkOut": session.hotel.checkOut,
                                "adults": session.hotel.adults,
                                "rooms": session.hotel.rooms,
                            },
                            results,
                        )
                        tool_calls.append(t)
                        supplier_durations["hotels"] = int((time.monotonic() - t0) * 1000)
                        if on_card and results.hotels:
                            on_card("hotel", results.hotels)
                    except Exception as e:
                        logger.error(f"Hotel search error: {e}")

                search_tasks.append(run_hotel_search())
                attempted_hotel = True

            if search_tasks:
                await asyncio.gather(*search_tasks, return_exceptions=True)

            # Filter & rank
            self._filter_and_rank(session, results, last_user)
            if attempted_flight:
                date_statuses: Dict[str, str] = {}
                for d in dep_dates:
                    d_offers = [f for f in results.flights if (f.outbound.departureLocal or "")[:10] == d]
                    if d_offers:
                        date_statuses[d] = "success"
                    elif d in date_failures:
                        date_statuses[d] = "failed"
                    else:
                        date_statuses[d] = "empty"

                results.dateStatuses = date_statuses
                dates_with_offers = [d for d in dep_dates if date_statuses.get(d) == "success"]
                dates_zero_offers = [d for d in dep_dates if date_statuses.get(d) == "empty"]
                dates_failed = [d for d in dep_dates if date_statuses.get(d) == "failed"]

            parts = [f"{n} {k}" for k, n in (("flights", len(results.flights)), ("trains", len(results.trains)), ("hotels", len(results.hotels))) if n]
            if on_status:
                on_status(f"Found {', '.join(parts)}" if parts else "No live results found")
            if on_results:
                on_results(results)  # cards appear while the model is still writing

            if attempted_hotel and not results.hotels:
                dest_name = session.destination.name if session.destination else (session.hotel.destination or "is destination")
                zero_results_reason = f"Maine {dest_name} ke hotels check kiye yaar, par in dates ke liye abhi koi rooms/properties available nahi dikh rahe hain (online portal pe bookings open nahi hui hain ya sold out dikha raha hai)."
                zero_results_suggestion = "Kya aap dates thodi aage-peeche karna chahenge ya nearby kisi area me dekhein?"
            elif attempted_flight and not results.flights:
                orig_name = session.origin.name if session.origin else "origin"
                dest_name = session.destination.name if session.destination else "destination"
                d_str = session.dates.departureDate or "in dates"
                if session.cabinClass == "first":
                    zero_results_reason = (
                        f"Maine {orig_name} se {dest_name} ke liye First Class check kiya dost, "
                        f"par {d_str} ke liye is route par koi First Class flights available nahi hain."
                    )
                    zero_results_suggestion = "Kya aap Business Class ya Economy options dekhna chahenge?"
                    alt_card = normalize_clarification(
                        {
                            "field": "cabin",
                            "question": "First Class available nahi hai. Kya aap Business Class ya Economy dekhna chahenge?",
                            "options": [
                                {"label": "Business Class", "value": "business"},
                                {"label": "Economy", "value": "economy"},
                                {"label": "Change Date", "value": "change_date"},
                            ],
                            "allowCustom": True,
                        },
                        turn_id,
                    )
                    if session.sessionId and alt_card:
                        clarification_store.open(session.sessionId, alt_card)
                    if on_clarification and alt_card:
                        on_clarification(alt_card)
                elif session.cabinClass == "business":
                    zero_results_reason = (
                        f"Maine {orig_name} se {dest_name} ke liye Business Class check kiya dost, "
                        f"par {d_str} ke liye seats available nahi hain."
                    )
                    zero_results_suggestion = "Kya hum Economy options check karein ya travel date badlein?"
                    alt_card = normalize_clarification(
                        {
                            "field": "cabin",
                            "question": "Business Class available nahi hai. Kya aap Economy dekhna chahenge?",
                            "options": [
                                {"label": "Economy", "value": "economy"},
                                {"label": "Change Date", "value": "change_date"},
                            ],
                            "allowCustom": True,
                        },
                        turn_id,
                    )
                    if session.sessionId and alt_card:
                        clarification_store.open(session.sessionId, alt_card)
                    if on_clarification and alt_card:
                        on_clarification(alt_card)
                elif dates_failed and not dates_zero_offers:
                    zero_results_reason = (
                        f"Maine {orig_name} se {dest_name} ke liye flights check karne ki koshish ki yaar, "
                        f"par supplier timeout ya network error ki wajah se {', '.join(dates_failed)} ka schedule check nahi ho paya."
                    )
                    zero_results_suggestion = "Aap retry button daba kar dobara check kar sakte hain."
                elif dates_failed and dates_zero_offers:
                    zero_results_reason = (
                        f"Maine {orig_name} se {dest_name} ke liye flights check ki. "
                        f"{', '.join(dates_zero_offers)} ke liye koi flight available nahi mili, aur {', '.join(dates_failed)} ka schedule network issue ki wajah se check nahi ho paya."
                    )
                    zero_results_suggestion = "Aap failed date ke liye retry kar sakte hain."
                else:
                    zero_results_reason = f"Maine {orig_name} se {dest_name} ki flights check ki yaar, par {d_str} ke liye saari flights fully booked dikha raha hai (seats available nahi hain)."
                    zero_results_suggestion = "Kya hum kisi dusri date pe try karein ya nearby airport check karun?"
            elif attempted_train and not results.trains:
                orig_name = session.origin.name if session.origin else "origin"
                dest_name = session.destination.name if session.destination else "destination"
                d_str = session.dates.departureDate or "in dates"
                zero_results_reason = f"Maine {orig_name} se {dest_name} ki trains check ki dost, par {d_str} ke liye koi available seats nahi mil rahi hain (sab waiting list ya full dikha raha hai)."
                zero_results_suggestion = "Kya aap koi doosri date ya alternative train try karna chahenge?"

        # 5. Build Facts
        facts = build_facts(session, results)
        if preflight.inferred_date and preflight.status_message:
            facts["date.inferredDefault"] = preflight.status_message
        if attempted_flight:
            facts["search.status"] = "has_results" if results.flights else ("failed" if dates_failed and not dates_with_offers else "zero_results")
            facts["search.datesChecked"] = ", ".join(dep_dates)
            if dates_with_offers:
                facts["search.datesWithFlights"] = ", ".join(dates_with_offers)
            if dates_zero_offers:
                facts["search.datesUnavailable"] = ", ".join(dates_zero_offers)
            if dates_failed:
                facts["search.datesFailed"] = ", ".join(dates_failed)
        if zero_results_reason:
            facts["search.status"] = "zero_results" if not dates_failed else "failed"
            facts["search.noResultsReason"] = zero_results_reason
            facts["search.suggestion"] = zero_results_suggestion

        # 6. Specific Instructions
        specific_instruction = ""
        if parse_result.validationError:
            specific_instruction = (
                f"CRITICAL INSTRUCTION - INVALID INPUT / PAST DATE: "
                f"The user selected or asked for a date that has already passed: '{parse_result.validationError}'. "
                f"In 1-2 friendly sentences, clearly explain that this date has already passed, and ask them for today's or a future travel date (e.g. tomorrow or a specific upcoming date). "
                f"DO NOT invent or describe any flight or train search results."
            )
        elif zero_results_reason:
            specific_instruction = (
                f"CRITICAL DIRECTIVE - SEARCH COMPLETED WITH NO RESULTS: "
                f"The system already searched for availability, but found no options. "
                f"In 1-2 friendly, natural sentences: "
                f"1. Explain to the user using this natural reason: '{zero_results_reason}' "
                f"2. Suggest the alternative: '{zero_results_suggestion}' "
                f"STRICT RULES: "
                f"- NEVER mention technical errors, APIs, status codes, 403, or databases. "
                f"- NEVER say 'dhoondh rahi hoon', 'search kar rahi hoon', 'ek minute ruko', or tell the user to wait. The search has ALREADY concluded. "
                f"- Speak warmly and directly like a helpful human friend."
            )
        elif is_out_of_scope:
            specific_instruction = (
                "CRITICAL DIRECTIVE - NATURAL CASUAL CHAT (OFF-TOPIC / NON-TRAVEL): "
                "The user's message is not a travel search (it might be general conversation, a random question, cooking, news, gossip, joke, or playful banter). "
                "Respond warmly, naturally, and casually in 1 single short sentence like a real human friend talking to another friend. "
                "STRICT RULES: "
                "1. NEVER use repetitive, robotic customer-agent disclaimers like 'Main ek travel agent / travel buddy hoon' or 'ye sab nahi bata sakti/sakta'. NEVER repeat canned refusal lines. "
                "2. React casually or playfully to what they said in natural Hinglish, and then smoothly ask where they want to travel or if they'd like to plan a trip/flight/train/hotel. "
                "3. Every reply must be unique, conversational, and fresh."
            )
        elif not is_out_of_scope and is_transport_turn and not preflight.ready:
            dest_str = session.destination.name if session.destination else "destination"
            mode_str = "trains" if session.mode == "train" else "flights"
            pending_clar = getattr(session, "pendingClarification", "") or ""

            if preflight.missing_field == "origin":
                if pending_clar.startswith("unrecognized_origin:"):
                    bad_c = pending_clar.split(":", 1)[1]
                    specific_instruction = (
                        f"CRITICAL INSTRUCTION - UNRECOGNIZED ORIGIN: The user mentioned '{bad_c}', which is not a recognized departure city. "
                        f"In 1 friendly sentence, explain that '{bad_c.capitalize()}' is not a recognized city or station, and ask where they are traveling from. "
                        f"DO NOT accept '{bad_c}' as a valid origin and DO NOT invent any search results."
                    )
                else:
                    specific_instruction = (
                        f"CRITICAL INSTRUCTION - MISSING ORIGIN: The user wants {mode_str} to {dest_str}, "
                        "but has not specified where they are traveling from. "
                        "In 1 friendly, natural sentence, ask: 'Aap kahan se travel karna chahte ho?' "
                        "DO NOT claim you are searching live options or ask which preference option they prefer."
                    )
            elif preflight.missing_field == "destination":
                if pending_clar.startswith("unrecognized_destination:"):
                    bad_c = pending_clar.split(":", 1)[1]
                    specific_instruction = (
                        f"CRITICAL INSTRUCTION - UNRECOGNIZED DESTINATION: The user mentioned '{bad_c}', which is not a recognized destination. "
                        f"In 1 friendly sentence, explain that '{bad_c.capitalize()}' is not a recognized city, and ask where they want to travel. "
                        f"DO NOT accept '{bad_c}' as a valid destination and DO NOT invent any search results."
                    )
                else:
                    specific_instruction = (
                        "CRITICAL INSTRUCTION - MISSING DESTINATION: The user has not specified their destination. "
                        "In 1 friendly, natural sentence, ask where they want to travel. "
                        "DO NOT claim you are searching live options or ask which preference option they prefer."
                    )
            elif preflight.missing_field == "passengers":
                specific_instruction = (
                    "CRITICAL INSTRUCTION - PASSENGER COUNT: "
                    "In 1 friendly, natural sentence, ask: 'Kitne log travel karenge?' "
                    "DO NOT claim results exist or describe options."
                )
            elif preflight.missing_field == "date":
                specific_instruction = (
                    "CRITICAL INSTRUCTION - AMBIGUOUS OR MISSING DATE: The travel date was ambiguous or unclear. "
                    "In 1 friendly, natural sentence, ask the user which date they would like to travel on. "
                    "DO NOT claim results exist or describe flight options."
                )
        elif scope_check.isMixed:
            specific_instruction = (
                "CRITICAL INSTRUCTION - MIXED REQUEST: "
                "The user mentioned something non-travel alongside travel. Casually acknowledge it in half a sentence, but focus entirely on helping them with their travel/stay request. Do not give stiff disclaimers."
            )
        elif is_mode_comparison and (results.trains or results.flights):
            diff = facts.get("comparison.fareDifference", "₹0")
            cheaper = facts.get("comparison.cheaperMode", "train")
            train_name = facts.get("comparison.cheapestTrain.name", "")
            train_fare = facts.get("comparison.cheapestTrain.fare", "")
            flight_airline = facts.get("comparison.cheapestFlight.airline", "")
            flight_fare = facts.get("comparison.cheapestFlight.fare", "")

            specific_instruction = f"""CRITICAL INSTRUCTION - LIVE MODE COMPARISON (TRAIN VS FLIGHT):
The user asked: "{last_user}".
1. In the VERY FIRST SENTENCE, directly state which option is cheaper and by how much based on live facts: {cheaper.upper()} is cheaper by {diff}! (Train {train_fare} vs Flight {flight_fare}).
2. State the cheapest train: {train_name} ({facts.get('comparison.cheapestTrain.number', '')}) with fare {train_fare}.
3. State the cheapest flight: {flight_airline} with fare {flight_fare}.
4. Give a clear 1-sentence trade-off comparison between saving money on train vs saving time on flight.
5. Quote ONLY the real numbers from facts, no placeholders."""
        elif (session.compareAllCabins or session.cabinClass == "all" or any(k.startswith("cabin.") for k in facts)) and results.flights:
            specific_instruction = (
                "CRITICAL INSTRUCTION - ALL CABIN CLASSES COMPARISON: "
                "The user requested to compare all flight cabin classes. "
                "1. Compare EVERY available cabin class (Economy, Premium Economy, Business, First Class) based on the live facts: "
                "For EACH class in the facts, state the cheapest fare in ₹, airline name, and departure time. "
                "2. Clearly highlight the price differences between Economy, Premium Economy, and Business so the user can easily decide. "
                "3. State your recommendation in 1 sentence based on value vs comfort. "
                "4. Quote only exact numbers from the facts, no placeholders, and do NOT ask unnecessary closing questions."
            )
        elif results.hotels:
            specific_instruction = (
                "CRITICAL INSTRUCTION - HOTEL SEARCH RESULTS: "
                "Directly answer the user's hotel request using the verified hotel results in facts. "
                "Search is ALREADY DONE: NEVER say 'checking', 'check kar rahi hoon', 'dhundh rahi hoon' or ask the user to wait. "
                "Summarise the top hotel options now: for each top hotel, state the name, stars/rating, area/address, and room fare in ₹. "
                "Give a friendly 1-sentence recommendation on the best option. Do NOT ask unnecessary closing questions."
            )
        elif results.trains or results.flights:
            nearest_note = ""
            if session.destination and not session.destination.hasAirport and results.flights:
                nearest_note = f" IMPORTANT ROUTE NOTE: The user's destination '{session.destination.name}' has no commercial airport. We searched flights to nearest airport {session.destination.nearestAirportName} ({session.destination.nearestAirportCode}). In your very first sentence, inform the user warmly and naturally (e.g. '{session.destination.name} ke liye direct flight nahi hai yaar, toh nearest airport {session.destination.nearestAirportName} ki flights nikaal li hain!') and summarize the flight options."
            multi_date_note = ""
            if len(dep_dates) > 1 and results.flights:
                multi_date_note = (
                    f" MULTI-DATE SEARCH SUMMARY: Dates searched: {', '.join(dep_dates)}. "
                    f"Dates WITH available flights: {', '.join(dates_with_offers)}. "
                    f"{'Dates with NO availability: ' + ', '.join(dates_zero_offers) + '. ' if dates_zero_offers else ''}"
                    f"{'Dates we could not check (supplier timeout / error): ' + ', '.join(dates_failed) + '. Do NOT claim these dates are fully booked or have no flights! Inform the user we could not check them and they can retry. ' if dates_failed else ''}"
                    "Summarise the available dates with airline and fares. DO NOT claim that all flights or all dates are fully booked when inventory exists in facts! "
                )
            specific_instruction = (
                f"Directly answer the user's travel question using the verified search results below.{nearest_note}{multi_date_note} "
                "Search is ALREADY DONE: NEVER say 'checking', 'check kar rahi hoon', 'dhundh rahi hoon' or ask the user to wait. "
                "Summarise the actual options now: for EACH travel date with availability in the facts, give the cheapest option with airline/train, exact departure-arrival times and price in ₹. "
                "Do NOT ask unnecessary closing questions."
            )

        # 7. Construct Prompt
        facts_lines = [f"{k}: {v}" for k, v in facts.items()]
        if outcome:
            facts_lines.insert(0, outcome)
        facts_summary = ("Facts:\n" + "\n".join(facts_lines)) if facts_lines else (outcome or "No supplier data yet.")
        prompt = build_system_prompt_with_context(
            session=session,
            time_zone=self.timezone,
            facts=facts_summary,
            specific_instruction=specific_instruction,
            is_interactive_mode=interactive_mode,
            character=character,
        )

        # 8. Call AI Service
        if on_status:
            if session.compareAllCabins or session.cabinClass == "all":
                on_status("Comparing all cabin classes…")
            elif results.flights and results.trains:
                on_status("Summarising flights and trains…")
            elif results.flights:
                on_status("Summarising top flights…")
            elif results.trains:
                on_status("Summarising top trains…")
            elif results.hotels:
                on_status("Summarising top hotels…")
            else:
                on_status("Thinking…")

        has_results = bool(results.flights or results.trains or results.hotels)
        stall = re.compile(
            r"\b(?:"
            r"ek\s+min(?:ute|at)?|"
            r"ek\s+sec(?:ond)?|"
            r"ruko|"
            r"wait(?:\s+karo)?|"
            r"hold\s+on|"
            r"thoda\s+(?:time|waqt|ruko)|"
            r"time\s+lagega|"
            r"waqt\s+lagega|"
            r"dekh\s+rah[ei]|"
            r"dekh\s+ke|"
            r"dhundh|dhoondh|"
            r"dhoond\s+ke|"
            r"dhund\s+ke|"
            r"search\s+kar|"
            r"check\s+kar|"
            r"chek\s+kar|"
            r"check\s+karke|"
            r"let\s+me\s+(?:check|search|find)"
            r")\b",
            re.IGNORECASE,
        )
        raw_text = ""
        reply_content = ""
        is_streaming_enabled = bool(getattr(settings, "LLM_TOKEN_STREAMING_ENABLED", False) and on_sentence is not None)
        try:
            if is_streaming_enabled:
                sentence_buffer = SentenceBuffer()
                gate = GroundingGate.from_results_and_facts(results, facts)
                approved_sentences = []
                llm_start_mono = time.monotonic()

                async for token in self.ai.chat_stream(
                    system_prompt=prompt,
                    messages=history[-30:],
                    temperature=0.6,
                    max_tokens=400 if interactive_mode else 1200,
                    response_format=REPLY_SCHEMA,
                    model=model,
                ):
                    if llm_ttft_ms is None:
                        llm_ttft_ms = int((time.monotonic() - llm_start_mono) * 1000)
                    raw_text += token
                    for s in sentence_buffer.feed(token):
                        ok, reason = gate.is_sentence_grounded(s)
                        if ok:
                            approved_sentences.append(s)
                            if on_sentence:
                                on_sentence(s)
                        else:
                            logger.warning(f"Grounding rejected streaming sentence: '{s}' (reason: {reason})")

                for s in sentence_buffer.finalize():
                    ok, reason = gate.is_sentence_grounded(s)
                    if ok:
                        approved_sentences.append(s)
                        if on_sentence:
                            on_sentence(s)
                    else:
                        logger.warning(f"Grounding rejected final sentence: '{s}' (reason: {reason})")

                if approved_sentences:
                    reply_content = " ".join(approved_sentences)
                else:
                    if is_transport_turn and not preflight.ready:
                        if preflight.missing_field == "origin":
                            if getattr(preflight, "unrecognized_candidate", None):
                                fallback = f"'{preflight.unrecognized_candidate}' koi jaani-pehchaani city ya airport nahi lag rahi dost. Aap kahan se travel karna chahte ho?"
                            else:
                                fallback = "Aap kahan se travel karna chahte ho?"
                        elif preflight.missing_field == "destination":
                            if getattr(preflight, "unrecognized_candidate", None):
                                fallback = f"'{preflight.unrecognized_candidate}' koi jaani-pehchaani city ya airport nahi lag rahi dost. Kahan chalne ka plan hai?"
                            else:
                                fallback = "Kahan chalne ka plan hai yaar? Destination batao toh options dekhein!"
                        elif preflight.missing_field == "passengers":
                            fallback = "Kitne log travel karenge dost?"
                        else:
                            fallback = "Aap kis date ko travel karna chahte ho?"
                    elif results.flights or results.trains or results.hotels:
                        fallback = "Maine live travel options nikaal li hain, aap neeche diye gaye cards mein verified fares aur details dekh sakte hain! ✈️"
                    else:
                        fallback = "Aap kahan se travel karna chahte ho?"
                    reply_content = fallback
                    if on_sentence:
                        on_sentence(fallback)
            else:
                repair = ""
                for attempt in range(2):
                    ai_resp = await self.ai.chat(
                        system_prompt=prompt + repair,
                        messages=history[-30:],
                        temperature=0.6,
                        max_tokens=400 if interactive_mode else 1200,
                        response_format=REPLY_SCHEMA,
                        model=model,
                    )
                    raw_text = ai_resp.message.content or ""
                    reply_content = publish_reply(raw_text, facts, last_user, session)
                    if has_results and stall.search(reply_content):
                        repair = "\nCORRECTION: Results are already in the facts. Do NOT say you are checking or searching or that it will take time ('thoda time lagega'). Present the actual flights/trains with times and prices now."
                        continue
                    if not has_results and stall.search(reply_content):
                        repair = "\nCORRECTION: Do NOT claim you are checking, searching, or tell the user to wait ('ruko', 'ek minute ruko', 'dekh rahi hoon', 'thoda time lagega'). If any travel detail (date, destination, or origin) is needed, ask the user for it directly."
                        continue
                    break

            booked_pattern = re.compile(
                r"\b(saari\s+flights\s+fully\s+booked|all\s+flights\s+are\s+fully\s+booked|fully\s+booked|koi\s+flight\s+nahi\s+mil|seats\s+available\s+nahi|no\s+flights?\s+available)\b",
                re.IGNORECASE,
            )
            is_stalled = bool(stall.search(reply_content))
            is_contradictory = bool(has_results and results.flights and booked_pattern.search(reply_content))

            if has_results and (is_stalled or is_contradictory):
                # Guaranteed safety: Never let a stalling phrase or contradictory 'fully booked' claim be published when live results exist!
                if results.flights:
                    f0 = results.flights[0]
                    orig_n = session.origin.name if session.origin else "origin"
                    dest_n = session.destination.name if session.destination else "destination"
                    dep_t = f0.outbound.departureLocal[11:16] if f0.outbound.departureLocal and len(f0.outbound.departureLocal) >= 16 else ""
                    arr_t = f0.outbound.arrivalLocal[11:16] if f0.outbound.arrivalLocal and len(f0.outbound.arrivalLocal) >= 16 else ""
                    fare_val = f"₹{int(f0.price.amount)}" if f0.price and f0.price.amount else ""
                    if len(dep_dates) > 1:
                        dates_str = ", ".join(dates_with_offers) if dates_with_offers else (f"{session.dates.dateRangeStart} to {session.dates.dateRangeEnd}" if session.dates.dateRangeStart else "in dates")
                        failed_note = f" (Note: We couldn't check {', '.join(dates_failed)} due to a supplier timeout - you can retry those dates)." if dates_failed else ""
                        reply_content = (
                            f"{orig_n} se {dest_n} ke liye {dates_str} ki {len(results.flights)} flights available hain! ✈️ "
                            f"Sabse sasta option **{f0.outbound.airline}** ka hai **{fare_val}** mein ({dep_t} - {arr_t}). "
                            f"{failed_note}Kaunsa option book karna chahenge?"
                        )
                    else:
                        d_val = session.dates.departureDate or ""
                        reply_content = (
                            f"{orig_n} se {dest_n} ke liye {d_val} ki {len(results.flights)} flights available hain! ✈️ "
                            f"Sabse sasta option **{f0.outbound.airline}** ka hai **{fare_val}** mein ({dep_t} - {arr_t}). "
                            f"Kaunsa option book karna chahenge?"
                        )
                elif results.trains:
                    t0 = results.trains[0]
                    orig_n = session.origin.name if session.origin else "origin"
                    dest_n = session.destination.name if session.destination else "destination"
                    d_val = session.dates.departureDate or ""
                    fare_str = f"₹{t0.classFares[0].fare}" if t0.classFares and t0.classFares[0].fare else ""
                    reply_content = (
                        f"{orig_n} se {dest_n} ke liye {d_val} ki {len(results.trains)} trains available hain! 🚆 "
                        f"Sabse fast **{t0.trainName} ({t0.trainNumber})** hai, departure {t0.departureTime} ({fare_str})."
                    )
                elif results.hotels:
                    h0 = results.hotels[0]
                    dest_n = session.destination.name if session.destination else (session.hotel.destination or "")
                    fare_str = f"₹{int(h0.price.amount)}" if h0.price and h0.price.amount else ""
                    reply_content = (
                        f"{dest_n} ke liye {len(results.hotels)} hotels available hain! 🏨 "
                        f"Top option **{h0.name}** hai ({fare_str} per night)."
                    )

            elif not has_results:
                is_stalled = bool(stall.search(reply_content))
                has_pref_hallucination = bool(re.search(r"\b(kaunsi flight|which flight|kaunsa option|konsi flight|prefer|pasand|choose)\b", reply_content, re.IGNORECASE))
                if is_transport_turn and not preflight.ready and (is_stalled or has_pref_hallucination or not reply_content.strip()):
                    if preflight.missing_field == "origin":
                        if getattr(preflight, "unrecognized_candidate", None):
                            reply_content = f"'{preflight.unrecognized_candidate}' koi jaani-pehchaani city ya airport nahi lag rahi dost. Aap kahan se travel karna chahte ho?"
                        else:
                            reply_content = "Aap kahan se travel karna chahte ho?"
                    elif preflight.missing_field == "destination":
                        if getattr(preflight, "unrecognized_candidate", None):
                            reply_content = f"'{preflight.unrecognized_candidate}' koi jaani-pehchaani city ya airport nahi lag rahi dost. Kahan chalne ka plan hai?"
                        else:
                            reply_content = "Kahan chalne ka plan hai yaar? Destination batao toh options dekhein!"
                    elif preflight.missing_field == "passengers":
                        reply_content = "Kitne log travel karenge dost?"
                    elif preflight.missing_field == "date":
                        reply_content = "Aap kis date ko travel karna chahte ho?"
                elif is_stalled:
                    if zero_results_reason:
                        reply_content = f"{zero_results_reason} {zero_results_suggestion}"
                    elif session.currentTask == "hotel" and not (session.hotel.checkIn and session.hotel.checkOut):
                        dest = session.destination.name if session.destination else (session.hotel.destination or "")
                        reply_content = f"{dest} ke liye check-in aur check-out dates kab ki hain dost? Dates batao toh best hotels nikaalun!" if dest else "Kahan ke liye hotel chahiye dost? City aur dates batao toh best hotels dekhein!"
                    elif session.origin and session.destination and not session.dates.departureDate:
                        reply_content = f"{session.origin.name} se {session.destination.name} ke liye kis date ko nikalne ka plan hai dost?"
                    elif not session.origin and session.destination:
                        reply_content = "Aap kahan se travel karna chahte ho?"
                    elif not session.destination:
                        reply_content = "Kahan chalne ka plan hai yaar? Destination batao toh options dekhein!"
                    else:
                        reply_content = re.sub(
                            r"[^.!?]*\b(?:"
                            r"ek\s+min(?:ute|at)?|"
                            r"ek\s+sec(?:ond)?|"
                            r"ruko|"
                            r"wait(?:\s+karo)?|"
                            r"hold\s+on|"
                            r"thoda\s+(?:time|waqt|ruko)|"
                            r"time\s+lagega|"
                            r"waqt\s+lagega|"
                            r"dekh\s+rah[ei]|"
                            r"dekh\s+ke|"
                            r"dhundh|dhoondh|"
                            r"dhoond\s+ke|"
                            r"dhund\s+ke|"
                            r"search\s+kar|"
                            r"check\s+kar|"
                            r"chek\s+kar|"
                            r"check\s+karke|"
                            r"let\s+me\s+(?:check|search|find)"
                            r")[^.!?]*[.!?]?",
                            "",
                            reply_content,
                            flags=re.IGNORECASE,
                        ).strip()
                        if not reply_content:
                            reply_content = "Aap kahan se travel karna chahte ho?" if not session.origin else "Kab nikalna hai dost? Date batao toh best options nikalun!"

            narration = build_narration(raw_text, facts, results, last_user, session)
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            if raw_text:
                reply_content = publish_reply(raw_text, facts, last_user, session)
            else:
                reply_content = (
                    "Abhi thoda network issue tha. Kya aap ek baar dobara pooch sakte hain?"
                    if character == "kabir"
                    else "Abhi thoda network issue tha. Kya aap ek baar dobara pooch sakti hain?"
                )

        # 9. Clean up and store
        self.store.save(session)
        logger.info(f"Turn finished in {int((time.time() - start_time) * 1000)}ms")

        return AgentResult(
            content=reply_content,
            toolCalls=tool_calls,
            results=results,
            model=ai_resp.model if ("ai_resp" in locals() and ai_resp) else (model or self.ai.model_name),
            sessionId=session.sessionId,
            narration=narration,
            telemetry={
                "ttft_ms": llm_ttft_ms,
                "suppliers_ms": supplier_durations,
            },
        )

    def _filter_and_rank(self, session: TravelSession, results: ChatResults, last_user: str = ""):
        # Sort flights by price
        if results.flights:
            if session.cabinClass == "all" or session.compareAllCabins:
                # Top flights per cabin class first, followed by all remaining distinct offers
                cabins = ["economy", "premium_economy", "business", "first"]
                kept = []
                for c in cabins:
                    c_flights = sorted(
                        [f for f in results.flights if (f.cabinClass or "").lower() == c],
                        key=lambda x: x.price.amount if x.price else float("inf")
                    )
                    kept.extend(c_flights[:3])
                kept_keys = {offer_key(o) for o in kept}
                remaining = [o for o in _dedupe(results.flights) if offer_key(o) not in kept_keys]
                results.flights = (kept if kept else _dedupe(results.flights)) + remaining
            else:
                if session.cabinClass and session.cabinClass != "all":
                    req_c = session.cabinClass.lower()
                    results.flights = [f for f in results.flights if (f.cabinClass or "economy").lower() == req_c]
                # Per date: cheapest offer per canonical airline brand first, followed by all remaining fetched distinct offers
                ledger = get_ledger(session.sessionId, getattr(session, "activeSearchContextId", None))
                ledger.remember(results.flights)
                # Group offers by date, and sort within each date group
                by_date: Dict[str, List[FlightOffer]] = {}
                for o in results.flights:
                    d_key = (o.outbound.departureLocal or "")[:10]
                    by_date.setdefault(d_key, []).append(o)
                grouped_ordered: List[FlightOffer] = []
                for d in sorted(by_date.keys()):
                    d_offers = by_date[d]
                    diverse_picks, _shortfall = select_diverse(d_offers)
                    diverse_keys = {offer_key(o) for o in diverse_picks}
                    remaining = [o for o in _dedupe(d_offers) if offer_key(o) not in diverse_keys]
                    grouped_ordered.extend(diverse_picks + remaining)
                results.flights = grouped_ordered if grouped_ordered else results.flights
                all_diverse, _shortfall = select_diverse(results.flights)
                ledger.mark_shown(all_diverse)

        # Sort trains by cheapest fare
        if results.trains:
            def min_fare(t: TrainOption) -> float:
                fares = [cf.fare for cf in (t.classFares or []) if cf.fare and cf.fare > 0]
                return min(fares) if fares else float("inf")

            results.trains.sort(key=min_fare)
            # Keep top 6
            results.trains = results.trains[:6]

        # Sort hotels by price or rating
        if results.hotels:
            if re.search(r"\b(best|top|rating|luxury)\b", last_user, re.IGNORECASE):
                results.hotels.sort(key=lambda h: h.rating.score if h.rating else 0, reverse=True)
            else:
                results.hotels.sort(key=lambda h: h.price.amount if h.price else float("inf"))
            results.hotels = results.hotels[:4]

    async def retry_date(
        self,
        session_id: str,
        departure_date: str,
        search_context_id: Optional[str] = None,
        cabin: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retry searching flights for a single date using the existing session's parameters."""
        from .search_preflight import today_ist, check_future_iso
        today = today_ist(tz=self.timezone)
        date_check = check_future_iso(departure_date, today)
        if date_check:
            return {
                "success": False,
                "error": f"Invalid travel date {departure_date}: {date_check}",
                "date": departure_date,
                "status": "failed",
                "offers": [],
            }

        session = self.store.get(session_id) if session_id else None
        # Attempt session recovery from search_context_id if session is missing or incomplete
        if (not session or not session.origin or not session.destination) and search_context_id:
            try:
                parts = search_context_id.split(":")
                if len(parts) >= 5:
                    orig_code, dest_code, _, ctx_cabin, ctx_adults = parts[:5]
                    orig_loc = LocationResolver.resolve_city(orig_code)
                    dest_loc = LocationResolver.resolve_city(dest_code)
                    if orig_loc and dest_loc:
                        sid = session_id or str(uuid.uuid4())
                        session = create_initial_session(sid)
                        session.origin = orig_loc
                        session.destination = dest_loc
                        session.cabinClass = cabin or ctx_cabin or "economy"
                        try:
                            session.passengers.adults = max(1, min(9, int(ctx_adults)))
                        except Exception:
                            session.passengers.adults = 1
                        session.activeSearchContextId = search_context_id
                        self.store.save(session)
            except Exception as e:
                logger.warning(f"Failed to recover session from searchContextId '{search_context_id}': {e}")

        if not session or not session.origin or not session.destination:
            return {
                "success": False,
                "error": "Search context expired. Please repeat your search.",
                "expired": True,
                "date": departure_date,
                "status": "failed",
                "offers": [],
            }

        flight_orig = LocationResolver.get_flight_code(session.origin.code or session.origin.name)
        flight_dest = LocationResolver.get_flight_code(session.destination.code or session.destination.name)
        search_ctx_id = search_context_id or getattr(session, "activeSearchContextId", None)

        retry_results = ChatResults()
        effective_cabin = cabin or session.cabinClass or "economy"
        try:
            t = await self.executor.execute(
                "search_flights",
                {
                    "origin": flight_orig,
                    "destination": flight_dest,
                    "departureDate": departure_date,
                    "adults": session.passengers.adults if session.passengers else 1,
                    "cabin": effective_cabin,
                },
                retry_results,
            )
            if not t.success:
                return {
                    "success": False,
                    "error": t.error or "Supplier error",
                    "date": departure_date,
                    "status": "failed",
                    "offers": [],
                }
        except Exception as e:
            logger.error(f"Error retrying flight search for {departure_date}: {e}")
            return {
                "success": False,
                "error": str(e),
                "date": departure_date,
                "status": "failed",
                "offers": [],
            }

        self._filter_and_rank(session, retry_results)
        matching_offers = [
            f for f in _dedupe(retry_results.flights)
            if (f.outbound.departureLocal or "")[:10] == departure_date
        ] if retry_results.flights else []

        for f in matching_offers:
            if search_ctx_id and not getattr(f, "searchContextId", None):
                f.searchContextId = search_ctx_id

        if matching_offers and search_ctx_id:
            ledger = get_ledger(session.sessionId, search_ctx_id)
            ledger.remember(matching_offers)

        status = "success" if matching_offers else "empty"
        return {
            "success": True,
            "date": departure_date,
            "status": status,
            "offers": [o.model_dump() for o in matching_offers],
        }

