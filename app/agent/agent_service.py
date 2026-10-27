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

        all_sigs = [f"{m.role}:{m.content}" for m in history]
        if stored and stored.processedMessages:
            prev_len = len(stored.processedMessages)
            if len(all_sigs) >= prev_len and all_sigs[:prev_len] == stored.processedMessages:
                to_process = history[prev_len:]
                if not to_process:
                    to_process = history[-1:]
                session.processedMessages = all_sigs
            else:
                to_process = history
                session.processedMessages = list(stored.processedMessages) + all_sigs
        else:
            to_process = history
            session.processedMessages = all_sigs

        for m in to_process:
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

        # Context preservation across turns: extract origin/destination if still missing (only from user messages, never distractions/abandoned)
        if (
            (not session.origin or not session.destination)
            and not session.abandoned
            and session.lastUserIntent != "distraction"
            and not session.pendingClarification
        ):
            for m in to_process:
                if m.role != "user":
                    continue
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
            has_train = any(re.search(r"\b(train|trains|rail|railway|irctc|gaadi|gadi)\b|ट्रेन|रेल", m.content or "", re.IGNORECASE) for m in history if m.role == "user")
            has_flight = any(re.search(r"\b(flight|flights|plane|planes|aeroplane|airline)\b|फ्लाइट|शोध", m.content or "", re.IGNORECASE) for m in history if m.role == "user")
            if has_train and not has_flight:
                session.mode = "train"
            elif has_flight and not has_train:
                session.mode = "flight"

        if parse_result and (parse_result.validationError or session.lastUserIntent in ("distraction", "abandon", "abandon_followup")):
            return session, parse_result

        parse_result = TripIntentParser.evaluate_next_action(session, self.timezone)
        return session, parse_result

    def _resolve_selected_offer(
        self,
        last_user: str,
        session: TravelSession,
        explicit_offer_id: Optional[str] = None,
    ) -> tuple[Optional[FlightOffer], Optional[str], Optional[str]]:
        offers: List[FlightOffer] = list(session.lastOffers or [])

        # 1. Explicit fareId / offerId mentioned in text
        explicit_fid_m = re.search(
            r"(?:\b(?:fare\s*id|offer\s*id|price\s*id)\s*[:=]?\s*([A-Za-z0-9_~\-\.]+)|\b((?:fare_|offer_|bankg_|5-|FARE_|OFFER_|FAKE_)[A-Za-z0-9_~\-\.]+)\b)",
            last_user,
            re.IGNORECASE,
        )
        if explicit_fid_m:
            req_id = explicit_fid_m.group(1) or explicit_fid_m.group(2)
            clean_req = req_id[len("bankg_"):] if req_id.startswith("bankg_") else req_id
            if not offers:
                return None, None, "no_active_search_results"
            for o in offers:
                obs_fid = (o.observedDetails or {}).get("fareId")
                if o.offerId == req_id or o.offerId == f"bankg_{clean_req}" or obs_fid == req_id or obs_fid == clean_req:
                    return o, str(obs_fid or o.offerId), None
            return None, None, "mismatched_fare_id"

        if explicit_offer_id:
            if not offers:
                return None, None, "no_active_search_results"
            clean_req = explicit_offer_id[len("bankg_"):] if explicit_offer_id.startswith("bankg_") else explicit_offer_id
            for o in offers:
                obs_fid = (o.observedDetails or {}).get("fareId")
                if o.offerId == explicit_offer_id or o.offerId == f"bankg_{clean_req}" or obs_fid == explicit_offer_id or obs_fid == clean_req:
                    return o, str(obs_fid or o.offerId), None
            return None, None, "mismatched_fare_id"

        if not offers:
            return None, None, "no_active_search_results"

        # 2. Match by flight number (e.g. "6E 6045", "SG 252", "AI 805", "UK 955")
        fn_m = re.search(r"\b([A-Z0-9]{2})\s*[- ]?\s*(\d{3,4})\b", last_user, re.IGNORECASE)
        if fn_m and any(c.isalpha() for c in fn_m.group(1)) and not fn_m.group(1).upper().startswith("20"):
            target_fn = f"{fn_m.group(1).upper()}{fn_m.group(2)}"
            matched_fn_offers = []
            for o in offers:
                seg_fns = [
                    (s.flightNumber or "").replace(" ", "").replace("-", "").upper()
                    for s in (o.outbound.segments or [])
                ]
                out_fn = (getattr(o.outbound, "flightNumber", "") or "").replace(" ", "").replace("-", "").upper()
                if target_fn in seg_fns or target_fn == out_fn:
                    matched_fn_offers.append(o)
            if not matched_fn_offers:
                return None, None, "flight_not_in_results"
            chosen = min(matched_fn_offers, key=lambda x: x.price.amount if x.price and x.price.amount else float("inf"))
            fid = (chosen.observedDetails or {}).get("fareId") or chosen.offerId
            return chosen, str(fid), None

        # 3. Match by ordinal ("first"/"1st", "second"/"2nd", "third"/"3rd", "option 1")
        lower_u = last_user.lower()
        if re.search(r"\b(first|1st|option\s*1|flight\s*1|pehla|pehli)\b", lower_u):
            chosen = offers[0]
            return chosen, str((chosen.observedDetails or {}).get("fareId") or chosen.offerId), None
        if re.search(r"\b(second|2nd|option\s*2|flight\s*2|doosra|dusra|dusri)\b", lower_u):
            if len(offers) >= 2:
                chosen = offers[1]
                return chosen, str((chosen.observedDetails or {}).get("fareId") or chosen.offerId), None
            return None, None, "flight_not_in_results"
        if re.search(r"\b(third|3rd|option\s*3|flight\s*3|teesra|tisra)\b", lower_u):
            if len(offers) >= 3:
                chosen = offers[2]
                return chosen, str((chosen.observedDetails or {}).get("fareId") or chosen.offerId), None
            return None, None, "flight_not_in_results"

        # 4. Match by superlative ("cheapest", "fastest")
        if re.search(r"\b(cheapest|lowest|sabse\s+sasta|sasti)\b", lower_u):
            chosen = min(offers, key=lambda x: x.price.amount if x.price and x.price.amount else float("inf"))
            return chosen, str((chosen.observedDetails or {}).get("fareId") or chosen.offerId), None
        if re.search(r"\b(fastest|shortest|quickest|sabse\s+fast)\b", lower_u):
            chosen = min(offers, key=lambda x: x.outbound.durationMinutes or 999999)
            return chosen, str((chosen.observedDetails or {}).get("fareId") or chosen.offerId), None

        # 5. Match by airline name ("IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa")
        for brand_kw, brand_match in (
            ("indigo", "indigo"),
            ("air india", "air india"),
            ("spicejet", "spicejet"),
            ("vistara", "vistara"),
            ("akasa", "akasa"),
        ):
            if brand_kw in lower_u:
                brand_offers = [o for o in offers if brand_match in (o.outbound.airline or "").lower()]
                if not brand_offers:
                    return None, None, "flight_not_in_results"
                distinct_fns = {
                    (o.outbound.segments[0].flightNumber if o.outbound.segments else o.offerId)
                    for o in brand_offers
                }
                if len(distinct_fns) > 1:
                    return None, None, "ambiguous_selection"
                chosen = brand_offers[0]
                return chosen, str((chosen.observedDetails or {}).get("fareId") or chosen.offerId), None

        # 6. Match by time of day reference ("that morning flight", "that evening flight")
        for tod, (h_start, h_end) in (("morning", (5, 11)), ("afternoon", (12, 16)), ("evening", (17, 20)), ("night", (21, 23))):
            if tod in lower_u:
                tod_offers = []
                for o in offers:
                    dep_s = o.outbound.departureLocal or ""
                    if len(dep_s) >= 13 and dep_s[11:13].isdigit():
                        hr = int(dep_s[11:13])
                        if h_start <= hr <= h_end:
                            tod_offers.append(o)
                if len(tod_offers) == 1:
                    chosen = tod_offers[0]
                    return chosen, str((chosen.observedDetails or {}).get("fareId") or chosen.offerId), None
                elif len(tod_offers) > 1:
                    return None, None, "ambiguous_selection"

        # 7. Fallback to already-selected offer or single offer or top offer
        if session.selectedFareId:
            for o in offers:
                obs_fid = (o.observedDetails or {}).get("fareId")
                if o.offerId == session.selectedFareId or obs_fid == session.selectedFareId:
                    return o, session.selectedFareId, None
        chosen = offers[0]
        return chosen, str((chosen.observedDetails or {}).get("fareId") or chosen.offerId), None

    def _reconcile_fare_review(
        self,
        selected_offer: Optional[FlightOffer],
        requested_fare_id: str,
        rev: Dict[str, Any],
    ) -> Dict[str, Any]:
        orig_price = (
            float(selected_offer.price.amount)
            if (selected_offer and selected_offer.price and selected_offer.price.amount is not None)
            else (
                float(rev["searchPrice"])
                if rev.get("searchPrice") is not None
                else (float(rev["originalSearchFare"]) if rev.get("originalSearchFare") is not None else None)
            )
        )
        orig_curr = (
            (selected_offer.price.currency if (selected_offer and selected_offer.price and selected_offer.price.currency) else None)
            or rev.get("searchCurrency")
            or rev.get("originalCurrency")
            or "INR"
        ).strip().upper()
        orig_basis = (
            rev.get("originalPriceBasis")
            or rev.get("searchPriceBasis")
            or ((selected_offer.observedDetails or {}).get("priceBasis") if selected_offer else None)
            or (selected_offer.priceBasis if (selected_offer and selected_offer.priceBasis) else None)
            or "per_person"
        )

        raw_rev_price = (
            rev.get("reviewedPrice")
            if rev.get("reviewedPrice") is not None
            else (rev.get("reviewedFare") if rev.get("reviewedFare") is not None else rev.get("totalFare"))
        )
        rev_price: Optional[float] = None
        if raw_rev_price is not None:
            try:
                val = float(raw_rev_price)
                if val > 0:
                    rev_price = val
            except (TypeError, ValueError):
                rev_price = None

        rev_curr = str(rev.get("reviewedCurrency") or rev.get("currency") or orig_curr or "INR").strip().upper()
        rev_basis = str(rev.get("priceBasis") or orig_basis)
        missing_tax = rev.get("missingTaxFields") is True

        returned_fid = rev.get("fareId") or rev.get("returnedFareId")
        clean_req = requested_fare_id[len("bankg_"):] if requested_fare_id.startswith("bankg_") else requested_fare_id
        clean_ret = str(returned_fid)[len("bankg_"):] if returned_fid and str(returned_fid).startswith("bankg_") else (str(returned_fid) if returned_fid else None)
        explicit_fid_mismatch = bool(
            rev.get("status") == "FARE_ID_MISMATCH"
            or rev.get("fareIdMatched") is False
            or (clean_ret and clean_ret != clean_req)
        )

        rev["searchPrice"] = orig_price
        rev["originalSearchFare"] = orig_price
        rev["searchCurrency"] = orig_curr
        rev["originalCurrency"] = orig_curr
        rev["reviewedPrice"] = rev_price
        rev["reviewedFare"] = rev_price
        rev["reviewedCurrency"] = rev_curr
        rev["currency"] = rev_curr
        rev["originalPriceBasis"] = orig_basis
        rev["priceBasis"] = rev_basis

        if explicit_fid_mismatch:
            rev["status"] = "FARE_ID_MISMATCH"
            rev["fareChangeDirection"] = "fare_id_mismatch"
            rev["fareIdMismatch"] = True
            rev["fareIdMatched"] = False
            rev["priceChanged"] = True
            rev["fareChanged"] = True
            rev["originalPriceConfirmed"] = False
            rev["priceDiscrepancyNotice"] = (
                f"Supplier returned fare identifier '{returned_fid or rev.get('priceId')}' which does not match requested fareId '{requested_fare_id}'."
            )
        elif rev_price is None or missing_tax:
            rev["status"] = "PRICE_MISSING"
            rev["fareChangeDirection"] = "unverified"
            rev["missingPriceFields"] = True
            rev["priceVerified"] = False
            rev["priceChanged"] = False
            rev["fareChanged"] = False
            rev["originalPriceConfirmed"] = False
            rev["priceDiscrepancyNotice"] = (
                "Reviewed fare could not be verified because price or tax fields were missing in the supplier review response."
            )
        elif orig_curr != rev_curr:
            rev["status"] = "CURRENCY_MISMATCH"
            rev["fareChangeDirection"] = "currency_mismatch"
            rev["currencyMismatch"] = True
            rev["priceChanged"] = True
            rev["fareChanged"] = True
            rev["priceDelta"] = None
            rev["fareDifference"] = None
            rev["originalPriceConfirmed"] = False
            rev["priceDiscrepancyNotice"] = (
                f"Currency mismatch between search ({orig_curr} {orig_price}) and fare review ({rev_curr} {rev_price})."
            )
        elif orig_basis != rev_basis:
            rev["status"] = "BASIS_MISMATCH"
            rev["fareChangeDirection"] = "basis_mismatch"
            rev["priceBasisMismatch"] = True
            rev["priceChanged"] = True
            rev["fareChanged"] = True
            rev["priceDelta"] = None
            rev["fareDifference"] = None
            rev["originalPriceConfirmed"] = False
            rev["priceDiscrepancyNotice"] = (
                f"Price basis mismatch between search ({orig_basis}: ₹{orig_price}) and fare review ({rev_basis}: ₹{rev_price})."
            )
        elif orig_price is not None:
            delta = round(rev_price - orig_price, 2)
            rev["priceDelta"] = delta
            rev["fareDifference"] = delta
            rev["priceVerified"] = True
            if abs(delta) >= 0.01:
                direction = "increased" if delta > 0 else "decreased"
                sign = "+" if delta > 0 else "-"
                rev["status"] = "PRICE_CHANGED"
                rev["fareChangeDirection"] = direction
                rev["priceChanged"] = True
                rev["fareChanged"] = True
                rev["originalPriceConfirmed"] = False
                rev["priceDiscrepancyNotice"] = (
                    f"Fare {direction} during supplier review from ₹{orig_price:.2f} (airSearch) "
                    f"to ₹{rev_price:.2f} (flightReview) ({sign}₹{abs(delta):.2f})."
                )
            else:
                rev["status"] = "LOCKED"
                rev["fareChangeDirection"] = "unchanged"
                rev["priceChanged"] = False
                rev["fareChanged"] = False
                rev["originalPriceConfirmed"] = True
                rev["priceDiscrepancyNotice"] = None
        else:
            rev["status"] = "LOCKED"
            rev["fareChangeDirection"] = "unchanged"
            rev["priceChanged"] = False
            rev["fareChanged"] = False
            rev["originalPriceConfirmed"] = True

        return rev

    def _format_prebooking_reply(
        self,
        session: TravelSession,
        results: ChatResults,
        tool_calls: List[ToolCallSummary],
        prebooking_notes: List[str],
    ) -> str:
        parts: List[str] = list(prebooking_notes)
        if isinstance(results.fareReview, dict) and results.fareReview:
            rev = results.fareReview
            st = rev.get("status")
            sp = rev.get("searchPrice")
            rp = rev.get("reviewedPrice")
            bid = rev.get("bookingId")
            if st == "FARE_ID_MISMATCH":
                parts.append(
                    f"Fare review returned a mismatched fare identifier ({rev.get('priceDiscrepancyNotice')}), so the selected flight's fare is not confirmed."
                )
            elif st == "PRICE_MISSING":
                parts.append(
                    f"The original search price was ₹{sp}, but the supplier review response omitted complete price or tax fields, so the reviewed fare could not be verified."
                )
            elif st == "CURRENCY_MISMATCH":
                parts.append(
                    f"Currency mismatch detected: search returned {rev.get('searchCurrency')} {sp} while fare review returned {rev.get('reviewedCurrency')} {rp}. The fare cannot be confirmed without currency alignment."
                )
            elif st == "BASIS_MISMATCH":
                parts.append(
                    f"Price basis mismatch detected between search ({rev.get('originalPriceBasis')}: ₹{sp}) and fare review ({rev.get('priceBasis')}: ₹{rp})."
                )
            elif st == "PRICE_CHANGED" or rev.get("priceChanged"):
                direction = rev.get("fareChangeDirection") or ("increased" if (rev.get("priceDelta") or 0) > 0 else "decreased")
                delta_abs = abs(float(rev.get("priceDelta") or 0))
                parts.append(
                    f"During fare review, the supplier fare {direction} from the search price of ₹{sp} to the reviewed price of ₹{rp} (a change of ₹{delta_abs:g}). Booking session ID: {bid}. Note: actual ticket booking is disabled in this version."
                )
            else:
                parts.append(
                    f"Fare review succeeded! The fare is confirmed at ₹{rp} (unchanged from search price ₹{sp}), with booking session ID {bid}. We can now check the seat map or fare rules (actual ticket booking is disabled in this version)."
                )

        if isinstance(results.seatMap, dict) and results.seatMap:
            smap = results.seatMap
            seat_maps_list = smap.get("seatMaps") or smap.get("seats") or []
            avail_seats = []
            for sm in seat_maps_list:
                if isinstance(sm, dict):
                    for s_item in sm.get("seats") or sm.get("seatLayout") or []:
                        if isinstance(s_item, dict) and s_item.get("isAvailable", True):
                            sc = s_item.get("seatNumber") or s_item.get("seatCode") or s_item.get("code")
                            sp = s_item.get("price") if s_item.get("price") is not None else s_item.get("amount")
                            if sc:
                                avail_seats.append(f"{sc} (₹{sp})" if sp is not None else str(sc))
                    if not avail_seats and sm.get("availableSeats"):
                        for sc in sm.get("availableSeats", []):
                            avail_seats.append(str(sc))
            if avail_seats:
                parts.append(f"Seat map retrieved for booking {smap.get('bookingId')}: available seats include {', '.join(avail_seats[:6])}.")
            else:
                parts.append("The seat map response did not contain any available selectable seats for this flight segment.")

        if isinstance(results.fareRules, dict) and results.fareRules:
            frules = results.fareRules
            rules_list = frules.get("fareRules") or frules.get("rules") or []
            rule_strs = []
            for rl in rules_list:
                if isinstance(rl, dict):
                    cat = rl.get("category") or rl.get("type") or rl.get("ruleType") or "Rule"
                    fee = rl.get("amount") if rl.get("amount") is not None else rl.get("fee")
                    desc = rl.get("description") or rl.get("policy") or rl.get("text") or ""
                    if fee is not None and desc:
                        rule_strs.append(f"{cat}: ₹{fee} ({desc})")
                    elif fee is not None:
                        rule_strs.append(f"{cat}: ₹{fee}")
                    elif desc:
                        rule_strs.append(f"{cat}: {desc}")
                elif isinstance(rl, str):
                    rule_strs.append(rl)
            if rule_strs:
                parts.append(f"Supplier fare rules: {'; '.join(rule_strs)}. Actual ticket booking and cancellation submission remain disabled in this version.")
            else:
                parts.append("The supplier returned no structured fare rules for this fare, so exact cancellation or date-change fees are not provided.")

        return " ".join(parts).strip()

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

        # 2. Scope & Adversarial Safety Guards
        raw_user_text = (last_msg.content or "").strip() if last_msg else last_user
        user_lang = session.userLanguage
        has_active_loc = bool(session.destination or session.origin)
        scope_check = TravelDomainGuard.evaluate(
            last_user,
            user_language=user_lang,
            character=character,
            has_active_session_with_location=has_active_loc,
        )
        is_out_of_scope = scope_check.isOutOfScope

        # Guard A: Prompt injection, system prompt / credential extraction, or fake price/PNR override
        if re.search(
            r"\b(ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions?|disregard\s+(?:previous|system)\s+instructions?|"
            r"reveal\s+(?:your\s+)?(?:system\s+prompt|merchant\s*key|secret|credentials|authorization|headers?)|"
            r"system\s+prompt|merchant\s*key|client\s*secret|authorization\s+header|xchecksum\s+key|xapikey|encryption\s+key|"
            r"pretend\s+(?:the\s+)?(?:fare|price|review|flightreview)|override\s+(?:safety|policy|rules)|"
            r"confirm\s+my\s+ticket\s+with\s+pnr|generate\s+a?\s*fake\s+pnr)\b",
            raw_user_text,
            re.IGNORECASE,
        ):
            self.store.save(session)
            return AgentResult(
                content=(
                    "I cannot override safety rules, reveal internal prompts or credentials, or fabricate prices and booking confirmations. "
                    "I can help you search flights, review real supplier fares, check seat maps, or view fare rules."
                ),
                toolCalls=[],
                results=empty_results(),
                model=self.ai.model_name,
                sessionId=session.sessionId,
                narration=[],
            )

        # Guard B: Sensitive booking / amendment record lookups without server-side authentication
        if re.search(
            r"\b(bookingDetails|booking_details|get_booking_details|amendmentDetails|amendment_details|get_amendment_details|"
            r"passenger\s+(?:phone|manifest|contact|passport)|(?:show|return|get|give|check|find|lookup|retrieve)\s+(?:me\s+)?(?:the\s+)?pnr|"
            r"pnr\s+(?:for|of)\s+booking|for\s+pnr\s+[A-Z0-9]+|someone\s+else'?s\s+booking|"
            r"lookup\s+booking|retrieve\s+pnr)\b",
            raw_user_text,
            re.IGNORECASE,
        ):
            self.store.save(session)
            return AgentResult(
                content=(
                    "Unauthorized: Access to sensitive booking and amendment records (such as PNR, ticket numbers, and passenger contact details) "
                    "requires verified server-side caller authentication and booking ownership."
                ),
                toolCalls=[],
                results=empty_results(),
                model=self.ai.model_name,
                sessionId=session.sessionId,
                narration=[],
            )

        # Guard C: Flight booking / instant ticketing is permanently disabled in this release
        if (
            re.search(
                r"\b(flightBook|flight_book|book_flight|instant_ticketing|call\s+flightbook|"
                r"book\s+(?:this|the|that|my|first|second|1st|2nd|cheapest)?\s*(?:flight|ticket|seat|option)|"
                r"ticket\s+book\s+karo|ticket\s+book\s+kar\s+do|seat\s+confirm\s+karo|proceed\s+to\s+book|"
                r"confirm\s+(?:my\s+)?(?:booking|ticket)|book\s+it|issue\s+my\s+ticket)\b",
                raw_user_text,
                re.IGNORECASE,
            )
            and not re.search(r"\b(policy|rules|charges|fee|review|price|fare)\b", raw_user_text, re.IGNORECASE)
        ):
            session.workflowState = "TRAVEL_PREPARATION_ONLY"
            self.store.save(session)
            return AgentResult(
                content="I've found the option and can help you review the fare, seats, baggage, and cancellation policy. Actual ticket booking is disabled in this version, so I haven't booked anything. Shall I check the remaining details?",
                toolCalls=[],
                results=empty_results(),
                model=self.ai.model_name,
                sessionId=session.sessionId,
                narration=[],
            )

        # Guard D: Cancellation / amendment submission is permanently disabled in this release
        if (
            re.search(
                r"\b(submitAmendment|submit_amendment|cancel_flight|cancel_booking|call\s+submitamendment|"
                r"cancel\s+(?:my|the|this)?\s*(?:flight|ticket|booking)|ticket\s+cancel\s+karo|ticket\s+cancel\s+kar\s+do)\b",
                raw_user_text,
                re.IGNORECASE,
            )
            and not re.search(r"\b(policy|charges|fee|rules|refund\s+amount|refund\s+policy)\b", raw_user_text, re.IGNORECASE)
        ):
            self.store.save(session)
            return AgentResult(
                content="Ticket cancellation and amendment submission are disabled in this release. I can check the fare cancellation rules or existing amendment status for you though. Would you like me to check that?",
                toolCalls=[],
                results=empty_results(),
                model=self.ai.model_name,
                sessionId=session.sessionId,
                narration=[],
            )

        # Guard E: Abandonment / task switching
        if session.lastUserIntent in ("abandon", "abandon_followup") or (
            has_active_loc
            and re.search(
                r"\b(stop\s+(?:the\s+)?search|stop\s+search|leave\s+(?:the\s+)?flight|leave\s+it|forget\s+travel|forget\s+it|"
                r"not\s+interested\s+anymore|nahi\s+chahiye|mat\s+dhoondo|rehne\s+do|cancel\s+the\s+search|let'?s\s+do\s+something\s+else)\b",
                raw_user_text,
                re.IGNORECASE,
            )
        ):
            session.workflowState = "FLOW_ABANDONED"
            session.abandoned = True
            session.lastAskedSlot = "none"
            session.pendingClarification = None
            self.store.save(session)
            return AgentResult(
                content="No worries! We have paused the travel search. Let me know whenever you would like to plan a trip again!",
                toolCalls=[],
                results=empty_results(),
                model=self.ai.model_name,
                sessionId=session.sessionId,
                narration=[],
            )

        # Guard F: Pure distraction during an active or new session — never run tools or mutate route
        if session.lastUserIntent == "distraction":
            is_out_of_scope = True
            self.store.save(session)
            if has_active_loc:
                orig_n = session.origin.name if session.origin else "your origin"
                dest_n = session.destination.name if session.destination else "your destination"
                if session.lastOffers:
                    msg = (
                        f"I don't have live updates for that off-topic question, dost! "
                        f"Your {orig_n} to {dest_n} flight search results are still ready whenever you'd like to continue."
                    )
                else:
                    missing_slot = (
                        "destination"
                        if not session.destination
                        else ("origin" if not session.origin else "travel date")
                    )
                    msg = (
                        f"Haha, I can only help with travel planning, dost! "
                        f"Want to continue with your {orig_n} to {dest_n} {session.mode if session.mode != 'unknown' else 'flight'} search? "
                        f"I still need your {missing_slot}."
                    )
            else:
                msg = "Haha, main sirf travel planning mein help kar sakti hoon dost! Kahan chalne ka plan hai?"
            return AgentResult(
                content=msg,
                toolCalls=[],
                results=empty_results(),
                model=self.ai.model_name,
                sessionId=session.sessionId,
                narration=[],
            )

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

        # 3b. Detect Post-Search Pre-Booking Intents (review_fare, get_seat_map, get_fare_rules, or filter/sort existing results)
        wants_review = bool(
            (
                getattr(last_msg, "selected_offer_id", None)
                or re.search(
                    r"\b(review\s+(?:the\s+|this\s+|that\s+)?(?:fare|flight|option|price|first|1st|second|2nd|third|3rd|option\s*\d+|flight\s*\d+|cheapest|fastest|indigo|air\s+india|spicejet|vistara|akasa|[A-Z0-9]{2}\s*[- ]?\d{3,4}|morning\s+flight|evening\s+flight|fareid|fare_id|bankg_[A-Za-z0-9_~\-\.]+|fare_[A-Za-z0-9_~\-\.]+)|"
                    r"fare\s+review|flightreview|review_fare|check\s+(?:the\s+)?reviewed\s+(?:fare|price)|lock\s+(?:the\s+)?(?:fare|price)|revalidate\s+(?:the\s+)?(?:fare|price)|"
                    r"(?:select|choose|pick|take)\s+(?:the\s+)?(?:first|1st|second|2nd|third|3rd|option\s*\d+|flight\s*\d+|cheapest|fastest|indigo|air\s+india|spicejet|vistara|akasa|[A-Z0-9]{2}\s*[- ]?\d{3,4}|morning\s+flight|evening\s+flight|fareid|fare_id|bankg_[A-Za-z0-9_~\-\.]+|fare_[A-Za-z0-9_~\-\.]+))\b",
                    raw_user_text,
                    re.IGNORECASE,
                )
            )
            and not re.search(r"\b(?:without\s+(?:running\s+|calling\s+)?|skip\s+)(?:flightreview|review_fare|fare\s+review)\b", raw_user_text, re.IGNORECASE)
        )
        wants_seat_map = bool(
            re.search(
                r"\b(seat\s*map|seatmap|flightseatmap|get_seat_map|seat\s+layout|available\s+seats?|check\s+seats?|"
                r"show\s+(?:me\s+)?(?:the\s+)?seats?|window\s+seat|aisle\s+seat|seat\s+dikhao|seats?\s+available)\b",
                last_user,
                re.IGNORECASE,
            )
        )
        wants_fare_rules = bool(
            re.search(
                r"\b(fare\s*rules?|farerule|get_fare_rules|cancellation\s+(?:rules?|charges?|fees?|policy)|"
                r"date[\s-]*change\s+(?:rules?|charges?|fees?|policy)|reschedule\s+(?:charges?|fees?|rules?)|"
                r"refund\s+(?:rules?|policy|charges?)|is\s+(?:it|this\s+flight|this\s+fare)\s+refundable|no[\s-]*show\s+(?:rules?|charges?|fee))\b",
                last_user,
                re.IGNORECASE,
            )
        )
        wants_filter_or_sort = bool(
            session.lastOffers
            and not wants_review
            and not wants_seat_map
            and not wants_fare_rules
            and re.search(
                r"\b(show\s+(?:me\s+)?(?:the\s+)?(?:cheapest|fastest|shortest|earliest|latest|morning|evening|afternoon|night|nonstop|non-stop|direct)|"
                r"which\s+(?:one|flight)\s+is\s+(?:the\s+)?(?:cheapest|fastest|shortest|nonstop|direct|best)|"
                r"filter\s+(?:by|to)|sort\s+by|only\s+(?:nonstop|non-stop|direct|morning|evening)|"
                r"allow\s+connecting|compare\s+(?:the\s+)?(?:top|these|two|flights)|baggage\s+allowance)\b",
                last_user,
                re.IGNORECASE,
            )
        )
        is_post_search_action = bool(wants_review or wants_seat_map or wants_fare_rules or wants_filter_or_sort)
        prebooking_notes: List[str] = []

        if not is_out_of_scope and is_post_search_action and not parse_result.validationError:
            if wants_filter_or_sort and session.lastOffers:
                ledger = get_ledger(session.sessionId, getattr(session, "activeSearchContextId", None))
                base_offers = list(ledger.pool.values()) if ledger.pool else list(session.lastOffers)
                results.flights = base_offers
                results.searchContextId = session.activeSearchContextId
                self._filter_and_rank(session, results, last_user)
                if results.flights:
                    session.lastOffers = list(results.flights)
                session.workflowState = "RESULTS_READY"
                if not results.flights:
                    prebooking_notes.append("No flights in the current search results matched that filter. You can relax the filter or try another date.")
                else:
                    top_f = results.flights[0]
                    fn_str = top_f.outbound.segments[0].flightNumber if top_f.outbound.segments else top_f.offerId
                    prebooking_notes.append(
                        f"From the current {session.origin.name if session.origin else ''} to {session.destination.name if session.destination else ''} results, "
                        f"the top matching option is {top_f.outbound.airline} ({fn_str}) at ₹{top_f.price.amount:g} departing at {(top_f.outbound.departureLocal or '')[11:16]}."
                    )

            if wants_review:
                results.flights = list(session.lastOffers or [])
                results.searchContextId = session.activeSearchContextId
                sel_offer, sel_fid, sel_err = self._resolve_selected_offer(
                    raw_user_text,
                    session,
                    getattr(last_msg, "selected_offer_id", None),
                )
                if sel_err == "no_active_search_results":
                    prebooking_notes.append(
                        "There are no active flight search results to review yet. Please search for flights on your route and date first."
                    )
                elif sel_err in ("mismatched_fare_id", "flight_not_in_results"):
                    prebooking_notes.append(
                        "That flight or fare identifier does not match any option in the current active search results. Please select a valid flight from the current search."
                    )
                elif sel_err == "ambiguous_selection":
                    cand_strs = [
                        f"{o.outbound.airline} ({o.outbound.segments[0].flightNumber if o.outbound.segments else o.offerId} at {(o.outbound.departureLocal or '')[11:16]}, ₹{o.price.amount:g})"
                        for o in (session.lastOffers or [])[:4]
                    ]
                    prebooking_notes.append(
                        f"Multiple flights match that description: {', '.join(cand_strs)}. Which specific flight number or departure time would you like me to review?"
                    )
                elif sel_offer and sel_fid:
                    session.workflowState = "FLIGHT_SELECTED"
                    t_rev = await self.executor.execute(
                        "review_fare",
                        {
                            "fareId": sel_fid,
                            "searchPrice": sel_offer.price.amount if sel_offer.price else None,
                            "searchCurrency": sel_offer.price.currency if sel_offer.price else "INR",
                        },
                        results,
                    )
                    tool_calls.append(t_rev)
                    if t_rev.success and isinstance(results.fareReview, dict):
                        self._reconcile_fare_review(sel_offer, sel_fid, results.fareReview)
                        if results.fareReview.get("status") != "FARE_ID_MISMATCH":
                            session.selectedOfferIds = [sel_offer.offerId]
                            session.selectedFareId = sel_fid
                            session.bookingSessionId = results.fareReview.get("bookingId") or None
                            session.reviewedFareData = results.fareReview
                            session.workflowState = "FARE_REVIEWED"
                    else:
                        prebooking_notes.append(
                            f"Supplier fare review could not be completed ({t_rev.error or 'provider error'}). Your search results are still saved if you'd like to retry or choose another flight."
                        )

            if wants_seat_map:
                # Check for injected/unverified bookingId in raw user text
                inj_bid_m = re.search(r"\bbooking\s*id\s*[:=]?\s*([A-Za-z0-9_\-]+)\b", raw_user_text, re.IGNORECASE)
                injected_bid = inj_bid_m.group(1) if inj_bid_m else None
                if injected_bid and injected_bid != session.bookingSessionId:
                    prebooking_notes.append(
                        f"Cannot retrieve a seat map using unverified or stale bookingId '{injected_bid}'. Please review a flight from the active search first."
                    )
                elif re.search(r"\bwithout\s+running\s+flightreview\b", raw_user_text, re.IGNORECASE):
                    prebooking_notes.append(
                        "Cannot call flightSeatMap without first completing flightReview to obtain a verified bookingId."
                    )
                elif session.bookingSessionId:
                    t_smap = await self.executor.execute(
                        "get_seat_map",
                        {"bookingId": session.bookingSessionId},
                        results,
                    )
                    tool_calls.append(t_smap)
                    if t_smap.success and isinstance(results.seatMap, dict):
                        session.lastSeatMap = results.seatMap
                    else:
                        prebooking_notes.append(
                            "The seat map is currently unavailable from the supplier for this flight, though your reviewed fare remains valid."
                        )
                else:
                    prebooking_notes.append(
                        "To view the seat map, we first need to select a flight from the active search results and review its fare (flightReview) to generate a booking session ID."
                    )

            if wants_fare_rules:
                inj_pid_m = re.search(r"\b(?:price\s*id|fare\s*id)\s*[:=]?\s*([A-Za-z0-9_~\-\.]+)\b", raw_user_text, re.IGNORECASE)
                injected_pid = inj_pid_m.group(1) if inj_pid_m else None
                valid_ids = set()
                if session.selectedFareId:
                    valid_ids.add(session.selectedFareId)
                if session.reviewedFareData and session.reviewedFareData.get("priceId"):
                    valid_ids.add(str(session.reviewedFareData.get("priceId")))
                for o in session.lastOffers or []:
                    valid_ids.add(o.offerId)
                    if (o.observedDetails or {}).get("fareId"):
                        valid_ids.add(str(o.observedDetails["fareId"]))

                if injected_pid and injected_pid not in valid_ids:
                    prebooking_notes.append(
                        f"Cannot retrieve fare rules for unverified or stale identifier '{injected_pid}'. Please select a flight from the active search first."
                    )
                else:
                    target_pid = (
                        (session.reviewedFareData or {}).get("priceId")
                        or session.selectedFareId
                    )
                    if not target_pid and session.lastOffers:
                        # Only resolve from lastOffers if 1 offer exists or user explicitly named the flight/ordinal
                        if len(session.lastOffers) == 1 or re.search(
                            r"\b(first|1st|second|2nd|third|3rd|cheapest|fastest|indigo|air\s+india|spicejet|vistara|akasa|[A-Z0-9]{2}\s*[- ]?\d{3,4})\b",
                            raw_user_text,
                            re.IGNORECASE,
                        ):
                            ro, rfid, rerr = self._resolve_selected_offer(raw_user_text, session)
                            if ro and rfid and not rerr:
                                target_pid = rfid
                    if target_pid:
                        t_rules = await self.executor.execute(
                            "get_fare_rules",
                            {"priceId": str(target_pid)},
                            results,
                        )
                        tool_calls.append(t_rules)
                        if t_rules.success and isinstance(results.fareRules, dict):
                            session.lastFareRules = results.fareRules
                        else:
                            prebooking_notes.append(
                                "Structured fare rules could not be retrieved from the supplier at this moment."
                            )
                    else:
                        prebooking_notes.append(
                            "Please select and review a flight from the active search results first so I can check the exact cancellation and date-change fare rules for that fare."
                        )

        # 4. Search Execution Preflight
        turn_id = str(uuid.uuid4())
        preflight = validate_flight_preflight(session, today_val, last_user, turn_id)
        is_pure_greeting_or_capability = bool(
            not session.origin
            and not session.destination
            and not getattr(session, "pendingClarification", None)
            and re.search(
                r"^\s*(?:hi|hello|hey|namaste|namaskar|hola|good\s+(?:morning|afternoon|evening)|"
                r"what\s+can\s+you\s+(?:do|help)|how\s+can\s+you\s+help|who\s+are\s+you|capabilities|help\s+me\s+with)\b",
                raw_user_text,
                re.IGNORECASE,
            )
            and not re.search(r"\b(flight|flights|train|trains|hotel|hotels|fly|ticket|tickets)\b", raw_user_text, re.IGNORECASE)
        )
        is_transport_turn = (
            not is_pure_greeting_or_capability
            and (session.mode in ["flight", "train", "any"] or session.currentTask in ["transport", "full_trip"])
        )

        if is_transport_turn and not is_out_of_scope and not is_post_search_action and not preflight.ready:
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
        dep_dates: List[str] = []
        dates_with_offers: List[str] = []
        dates_zero_offers: List[str] = []
        dates_failed: List[str] = []

        if not is_out_of_scope and not is_post_search_action and (actionable.readyForSearch or is_multi_day) and not parse_result.validationError:
            if on_status:
                on_status("Searching live flights and trains…")

            search_tasks = []
            req_dates = _requested_dates(last_user, self.timezone)
            multi_dates = (
                preflight.resolved_dates
                or resolve_multi_date_range(last_user, session.lastAskedQuestion, session.dates, today_val)
            )
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
                            f_args: Dict[str, Any] = {
                                "origin": flight_orig,
                                "destination": flight_dest,
                                "departureDate": d,
                                "adults": session.passengers.adults,
                                "cabin": c,
                            }
                            if session.passengers.children:
                                f_args["child"] = session.passengers.children
                            if session.passengers.infantsOnLap:
                                f_args["infant"] = session.passengers.infantsOnLap
                            t = await self.executor.execute(
                                "search_flights",
                                f_args,
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
                            f_args: Dict[str, Any] = {
                                "origin": flight_orig,
                                "destination": flight_dest,
                                "departureDate": d,
                                "adults": session.passengers.adults,
                                "cabin": session.cabinClass,
                            }
                            if session.passengers.children:
                                f_args["child"] = session.passengers.children
                            if session.passengers.infantsOnLap:
                                f_args["infant"] = session.passengers.infantsOnLap
                            t = await self.executor.execute(
                                "search_flights",
                                f_args,
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
                session.lastOffers = list(results.flights)
                if results.flights:
                    session.workflowState = "RESULTS_READY"
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

            hotel_tc_failed = any(tc.name == "search_hotels" and not tc.success for tc in tool_calls)
            train_tc_failed = any(tc.name == "search_trains" and not tc.success for tc in tool_calls)

            if attempted_hotel and not results.hotels:
                dest_name = session.destination.name if session.destination else (session.hotel.destination or "is destination")
                if hotel_tc_failed:
                    zero_results_reason = f"Maine {dest_name} ke hotels check karne ki koshish ki yaar, par hotel provider error / timeout ki wajah se live rates check nahi ho paye."
                    zero_results_suggestion = "Aap retry button daba kar dobara check kar sakte hain."
                else:
                    zero_results_reason = f"Maine {dest_name} ke hotels check kiye yaar, par in dates ke liye abhi koi rooms/properties available nahi mile."
                    zero_results_suggestion = "Kya aap dates thodi aage-peeche karna chahenge ya nearby kisi area me dekhein?"
            elif attempted_flight and not results.flights and not zero_results_reason:
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
                if train_tc_failed:
                    zero_results_reason = f"Maine {orig_name} se {dest_name} ki trains check karne ki koshish ki dost, par railway provider timeout ya network issue ki wajah se live schedule check nahi ho paya."
                    zero_results_suggestion = "Aap thodi der baad retry kar sakte hain."
                else:
                    zero_results_reason = f"Maine {orig_name} se {dest_name} ki trains check ki dost, par {d_str} ke liye koi available seats nahi mil rahi hain (sab waiting list ya full dikha raha hai)."
                    zero_results_suggestion = "Kya aap koi doosri date ya alternative train try karna chahenge?"

        # 5. Build Facts
        facts = build_facts(session, results)
        if not is_out_of_scope and is_post_search_action and not parse_result.validationError:
            if on_results:
                on_results(results)
            reply_content = self._format_prebooking_reply(session, results, tool_calls, prebooking_notes)
            narration = build_narration(reply_content, facts, results, last_user, session)
            self.store.save(session)
            return AgentResult(
                content=reply_content,
                toolCalls=tool_calls,
                results=results,
                model=model or self.ai.model_name,
                sessionId=session.sessionId,
                narration=narration,
                telemetry={"ttft_ms": llm_ttft_ms, "suppliers_ms": supplier_durations},
            )
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
            if has_active_loc:
                missing_str = preflight.missing_field if not preflight.ready else "travel dates"
                specific_instruction = (
                    "CRITICAL DIRECTIVE - HARMLESS DISTRACTION DURING TRAVEL PLANNING: "
                    f"The user has an active travel planning session (route: {session.origin.name if session.origin else ''} -> {session.destination.name if session.destination else ''}), "
                    "but temporarily asked a joke or made a casual/off-topic comment. "
                    "Respond warmly to their comment/joke in 1 sentence, and then briefly invite them to continue: "
                    f"'Want to continue with the {session.mode} search? I still need your {missing_str}.' "
                    "STRICT RULES: "
                    "- Do NOT discard their existing route or travel state. "
                    "- Do NOT use repetitive robotic disclaimers. Speak like a friendly buddy."
                )
            else:
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
                    if is_out_of_scope and not has_active_loc and scope_check.refusalMessage:
                        fallback = scope_check.refusalMessage
                    elif not is_out_of_scope and is_transport_turn and not preflight.ready:
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
                            f"{failed_note}Kaunsa option review karna chahenge?"
                        )
                    else:
                        d_val = session.dates.departureDate or ""
                        reply_content = (
                            f"{orig_n} se {dest_n} ke liye {d_val} ki {len(results.flights)} flights available hain! ✈️ "
                            f"Sabse sasta option **{f0.outbound.airline}** ka hai **{fare_val}** mein ({dep_t} - {arr_t}). "
                            f"Kaunsa option review karna chahenge?"
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
                if is_out_of_scope and not has_active_loc:
                    if is_stalled or not reply_content.strip() or "travel" not in reply_content.lower():
                        reply_content = scope_check.refusalMessage or "Main aapki travel buddy hoon! Flights, trains ya hotels plan karne mein help kar sakti hoon. Kahan chalne ka plan hai?"
                elif not is_out_of_scope and is_transport_turn and not preflight.ready and (is_stalled or has_pref_hallucination or not reply_content.strip()):
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
            elif is_out_of_scope:
                reply_content = "Main aapka travel assistant hoon aur flights, trains ya hotels dhoondhne mein help kar sakta hoon. Kahan chalna hai?"
            elif getattr(session, "cabinClass", None) == "first" and not (results.flights if hasattr(results, "flights") else []):
                reply_content = "Is route par First Class flights available nahi hain. Kya aap Business ya Economy dekhna chahenge?"
            elif session.pendingClarification or not session.origin or not session.destination:
                if not session.origin:
                    reply_content = "Aap kahan se travel karna chahte ho?"
                elif not session.destination:
                    reply_content = "Aap kahan travel karna chahte hain?"
                else:
                    reply_content = "Kab nikalna hai dost? Date batao toh best options nikalun!"
            elif hasattr(results, "flights") and not results.flights and is_transport_turn:
                reply_content = "Is route aur date par koi flights available nahi hain. Kya aap koi doosri date try karna chahenge?"
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

                if getattr(session, "nonstopOnly", False):
                    ns_flights = [f for f in results.flights if getattr(f.outbound, "stops", 0) == 0]
                    if ns_flights or re.search(r"\b(only|filter|nonstop|non-stop|direct)\b", last_user, re.IGNORECASE):
                        results.flights = ns_flights

                tod = getattr(session.constraints, "timeOfDay", "any") if getattr(session, "constraints", None) else "any"
                if tod and tod != "any":
                    def _matches_tod(f: FlightOffer) -> bool:
                        dep = (f.outbound.departureLocal or "") if f.outbound else ""
                        if len(dep) >= 13 and dep[11:13].isdigit():
                            hr = int(dep[11:13])
                            if tod == "morning":
                                return 5 <= hr < 12
                            if tod == "afternoon":
                                return 12 <= hr < 17
                            if tod == "evening":
                                return 17 <= hr < 21
                            if tod == "night":
                                return hr >= 21 or hr < 5
                        return True
                    tod_flights = [f for f in results.flights if _matches_tod(f)]
                    if tod_flights or re.search(r"\b(only|filter)\b", last_user, re.IGNORECASE):
                        results.flights = tod_flights

                if getattr(session, "rankingPreference", "smart") == "fastest":
                    results.flights = sorted(
                        _dedupe(results.flights),
                        key=lambda f: (
                            f.outbound.durationMinutes if (f.outbound and f.outbound.durationMinutes) else 999999,
                            f.price.amount if f.price else float("inf"),
                        ),
                    )
                else:
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

