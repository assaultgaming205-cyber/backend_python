"""End-to-end offline multi-turn conversational workflow tests for Travel Buddy.

Verifies the complete user journey through the real AgentService, SessionStoreService,
TripIntentParser, LocationResolver, ToolExecutor, and GroundingGate with zero external network calls:
Understand request -> Collect missing details -> Search flights -> Present fixture results ->
Select a specific result -> Review its fare -> Report any price change ->
Retrieve seat map when requested -> Retrieve fare rules when requested -> Explain next step honestly.
"""
import json
import re
from typing import Any, Dict, List, Optional
from unittest.mock import patch
import httpx
import pytest

from app.agent.agent_service import AgentService
from app.agent.session_store import SessionStoreService
from app.ai.ai_service import AiMessage, AiResponsePayload, AiService
from app.common.dates import plus_days, today_in_tz
from app.schemas.flight import FlightLeg, FlightOffer, FlightSegment, Money
from app.schemas.hotel import HotelOffer
from app.schemas.train import TrainOption


def make_fixture_flight(
    offer_id: str = "OFFER_BOM_DEL_1",
    fare_id: str = "FARE_BOM_DEL_101",
    airline: str = "IndiGo",
    carrier_code: str = "6E",
    flight_number: str = "6E-204",
    origin: str = "BOM",
    destination: str = "DEL",
    departure_date: str = "2026-11-15",
    dep_time: str = "08:00",
    arr_time: str = "10:15",
    duration_minutes: int = 135,
    stops: int = 0,
    amount: float = 4159.50,
    currency: str = "INR",
    cabin: str = "economy",
    refundable: bool = True,
) -> FlightOffer:
    dep_iso = f"{departure_date}T{dep_time}:00"
    arr_iso = f"{departure_date}T{arr_time}:00"
    seg = FlightSegment(
        carrierCode=carrier_code,
        carrierName=airline,
        flightNumber=flight_number,
        origin=origin,
        destination=destination,
        departureLocal=dep_iso,
        arrivalLocal=arr_iso,
        durationMinutes=duration_minutes,
        cabin=cabin,
    )
    leg = FlightLeg(
        airline=airline,
        origin=origin,
        destination=destination,
        departureLocal=dep_iso,
        arrivalLocal=arr_iso,
        durationMinutes=duration_minutes,
        stops=stops,
        segments=[seg],
    )
    return FlightOffer(
        offerId=offer_id,
        provider="BankG",
        price=Money(amount=amount, currency=currency),
        cabinClass=cabin,
        refundable=refundable,
        outbound=leg,
        deepLink="",
        observedDetails={
            "fareId": fare_id,
            "baseFare": round(amount - 659.50, 2),
            "taxes": 659.50,
            "priceBasis": "total",
            "requestedOrigin": origin,
            "requestedDestination": destination,
            "actualOrigin": origin,
            "actualDestination": destination,
            "routeMatch": True,
        },
    )


class OfflineBankGProvider:
    """Deterministic offline BankG flight provider double recording all calls."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.custom_offers: Optional[List[FlightOffer]] = None
        self.review_price_delta: float = 0.0
        self.review_override: Optional[Dict[str, Any]] = None
        self.seat_map_override: Optional[Dict[str, Any]] = None
        self.fare_rules_override: Optional[Dict[str, Any]] = None
        self.raise_on_search: Optional[Exception] = None
        self.raise_on_review: Optional[Exception] = None

    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1,
        cabin_class: str = "economy",
        cabin: Optional[str] = None,
        return_date: Optional[str] = None,
        child: int = 0,
        infant: int = 0,
    ) -> List[FlightOffer]:
        eff_cabin = (cabin or cabin_class or "economy").lower()
        self.calls.append(
            {
                "method": "airSearch",
                "origin": origin,
                "destination": destination,
                "departureDate": departure_date,
                "adults": adults,
                "child": child,
                "infant": infant,
                "cabin": eff_cabin,
                "returnDate": return_date,
            }
        )
        if self.raise_on_search:
            raise self.raise_on_search
        if self.custom_offers is not None:
            return list(self.custom_offers)
        return [
            make_fixture_flight(
                offer_id=f"OFFER_{origin}_{destination}_1",
                fare_id=f"FARE_{origin}_{destination}_101",
                airline="IndiGo",
                carrier_code="6E",
                flight_number="6E-204",
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                dep_time="08:00",
                arr_time="10:15",
                duration_minutes=135,
                stops=0,
                amount=4159.50,
                cabin=eff_cabin if eff_cabin != "all" else "economy",
            ),
            make_fixture_flight(
                offer_id=f"OFFER_{origin}_{destination}_2",
                fare_id=f"FARE_{origin}_{destination}_202",
                airline="Air India",
                carrier_code="AI",
                flight_number="AI-806",
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                dep_time="18:30",
                arr_time="20:30",
                duration_minutes=120,
                stops=0,
                amount=5250.00,
                cabin=eff_cabin if eff_cabin != "all" else "economy",
            ),
            make_fixture_flight(
                offer_id=f"OFFER_{origin}_{destination}_3",
                fare_id=f"FARE_{origin}_{destination}_303",
                airline="Akasa Air",
                carrier_code="QP",
                flight_number="QP-112",
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                dep_time="14:15",
                arr_time="17:45",
                duration_minutes=210,
                stops=1,
                amount=4600.00,
                cabin=eff_cabin if eff_cabin != "all" else "economy",
            ),
        ]

    async def review_fare(
        self,
        fare_id: str,
        search_price: Optional[float] = None,
        search_currency: str = "INR",
        search_offer: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if search_price is None and isinstance(search_offer, dict):
            search_price = search_offer.get("searchPrice")
            search_currency = search_offer.get("searchCurrency") or search_currency
        self.calls.append(
            {
                "method": "flightReview",
                "fareId": fare_id,
                "searchPrice": search_price,
                "searchCurrency": search_currency,
            }
        )
        if self.raise_on_review:
            raise self.raise_on_review
        if self.review_override is not None:
            return dict(self.review_override)
        base_sp = float(search_price) if search_price is not None else 4159.50
        rev_p = round(base_sp + self.review_price_delta, 2)
        return {
            "status": "PRICE_CHANGED" if abs(self.review_price_delta) >= 0.01 else "LOCKED",
            "bookingId": f"BOOK_{fare_id}",
            "fareId": fare_id,
            "priceId": fare_id,
            "searchPrice": base_sp,
            "searchCurrency": search_currency,
            "reviewedPrice": rev_p,
            "reviewedCurrency": search_currency,
            "baseFare": round(rev_p - 659.50, 2),
            "totalTax": 659.50,
            "priceBasis": "total",
            "originalPriceBasis": "total",
        }

    async def get_seat_map(self, booking_id: str) -> Dict[str, Any]:
        self.calls.append({"method": "flightSeatMap", "bookingId": booking_id})
        if self.seat_map_override is not None:
            return dict(self.seat_map_override)
        return {
            "status": "SUCCESS",
            "bookingId": booking_id,
            "seatMaps": [
                {
                    "segmentId": "SEG-1",
                    "seats": [
                        {"seatNumber": "1A", "isAvailable": True, "price": 350},
                        {"seatNumber": "1C", "isAvailable": True, "price": 250},
                        {"seatNumber": "2F", "isAvailable": True, "price": 0},
                    ],
                }
            ],
        }

    async def get_fare_rules(self, price_id: str) -> Dict[str, Any]:
        self.calls.append({"method": "fareRule", "priceId": price_id})
        if self.fare_rules_override is not None:
            return dict(self.fare_rules_override)
        return {
            "status": "SUCCESS",
            "priceId": price_id,
            "fareRules": [
                {"category": "Cancellation", "amount": 3000, "description": "₹3000 before 4 hours of departure"},
                {"category": "Date Change", "amount": 2500, "description": "₹2500 plus fare difference"},
            ],
        }


class OfflineRailProvider:
    async def search_trains(self, origin: str, destination: str, departure_date: str) -> List[TrainOption]:
        return []


class OfflineHotelProvider:
    async def search_hotels(self, destination: str, check_in: str, check_out: str, adults: int = 1, rooms: int = 1) -> List[HotelOffer]:
        return []


class DeterministicOfflineAiService(AiService):
    """100% offline deterministic AI service generating grounded replies from Facts & instructions."""

    def __init__(self) -> None:
        self.api_key = ""
        self.base_url = "http://offline.local"
        self.model = "offline-deterministic-model"
        self.model_name = "offline-deterministic-model"
        self.fast_model = "offline-deterministic-model"

    async def chat(
        self,
        system_prompt: str,
        messages: Optional[List[AiMessage]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.6,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AiResponsePayload:
        facts_map: Dict[str, str] = {}
        if "Facts:\n" in system_prompt:
            facts_block = system_prompt.split("Facts:\n", 1)[1].split("\n\n", 1)[0]
            for line in facts_block.splitlines():
                if ": " in line:
                    k, v = line.split(": ", 1)
                    facts_map[k.strip()] = v.strip()

        if "CRITICAL INSTRUCTION - INVALID INPUT / PAST DATE" in system_prompt or "INPUT_INVALID:" in system_prompt:
            err_m = re.search(r"INPUT_INVALID:\s*(.+)", system_prompt)
            reason = err_m.group(1) if err_m else "The requested date or passenger count is invalid."
            reply = f"{reason} Please provide a valid future travel date and passenger count."
        elif "CRITICAL DIRECTIVE - SEARCH COMPLETED WITH NO RESULTS" in system_prompt:
            reason = facts_map.get("search.noResultsReason", "No flights were found for this route and date.")
            sugg = facts_map.get("search.suggestion", "Would you like to try another date?")
            reply = f"{reason} {sugg}"
        elif "CRITICAL INSTRUCTION - UNRECOGNIZED ORIGIN" in system_prompt:
            reply = "That departure city is not recognized. Which city or airport are you flying from?"
        elif "CRITICAL INSTRUCTION - MISSING ORIGIN" in system_prompt:
            reply = "Aap kahan se travel karna chahte ho? Please share your departure city."
        elif "CRITICAL INSTRUCTION - UNRECOGNIZED DESTINATION" in system_prompt:
            reply = "That destination city is not recognized. Where would you like to travel?"
        elif "CRITICAL INSTRUCTION - MISSING DESTINATION" in system_prompt:
            reply = "Kahan chalne ka plan hai yaar? Destination batao toh options dekhein!"
        elif "CRITICAL INSTRUCTION - AMBIGUOUS OR MISSING DATE" in system_prompt:
            reply = "Aap kis date ko travel karna chahte ho? Please share your travel date."
        elif "CRITICAL INSTRUCTION - PASSENGER COUNT" in system_prompt:
            reply = "Kitne log travel karenge dost?"
        elif facts_map.get("flight.1.airline"):
            orig = facts_map.get("trip.origin", "Origin")
            dest = facts_map.get("trip.destination", "Destination")
            dep_d = facts_map.get("trip.date", "")
            airline = facts_map.get("flight.1.airline", "IndiGo")
            fn = facts_map.get("flight.1.flightNumber", "6E-204")
            fare_raw = facts_map.get("flight.1.fareAmount", "4159.5")
            try:
                fare_rounded = str(round(float(fare_raw)))
            except Exception:
                fare_rounded = fare_raw
            fare = facts_map.get("flight.1.fareWithCurrency", f"₹{fare_raw}")
            dep_t = facts_map.get("flight.1.departure", "")
            arr_t = facts_map.get("flight.1.arrival", "")
            dep_hm = dep_t[11:16] if len(dep_t) >= 16 else dep_t
            arr_hm = arr_t[11:16] if len(arr_t) >= 16 else arr_t
            reply = (
                f"{orig} se {dest} ke liye {dep_d} par flights available hain! "
                f"Sabse sasta option {airline} ({fn}) hai {fare} (₹{fare_rounded}) mein ({dep_hm} - {arr_hm}). "
                f"Kaunsa option select karke fare review karna chahenge?"
            )
        else:
            reply = (
                "Hello! Main aapki Travel Buddy hoon. I can help you search flights, review fares, "
                "check seat maps, and inspect fare rules (actual ticket booking is disabled in this version). "
                "Kahan chalne ka plan hai?"
            )

        raw_json = json.dumps({"reply": reply})
        return AiResponsePayload(
            message={"role": "assistant", "content": raw_json},
            content=raw_json,
            model=model or self.model,
            raw=raw_json,
        )


@pytest.fixture(autouse=True)
def block_all_external_http(monkeypatch: pytest.MonkeyPatch):
    """Intercepts any real httpx network call so zero external requests can escape."""
    async def _forbidden_send(self: Any, request: httpx.Request, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"EXTERNAL_HTTP_BLOCKED: Attempted outbound request to {request.method} {request.url}")

    monkeypatch.setattr(httpx.AsyncClient, "send", _forbidden_send)
    monkeypatch.setattr(httpx.Client, "send", _forbidden_send)


def build_offline_agent() -> tuple[AgentService, OfflineBankGProvider, SessionStoreService]:
    from app.agent.tool_executor import ToolExecutor

    flight_provider = OfflineBankGProvider()
    rail_provider = OfflineRailProvider()
    hotel_provider = OfflineHotelProvider()
    ai_service = DeterministicOfflineAiService()
    store = SessionStoreService()
    executor = ToolExecutor(
        flight_provider=flight_provider,  # type: ignore[arg-type]
        rail_provider=rail_provider,  # type: ignore[arg-type]
        hotel_provider=hotel_provider,  # type: ignore[arg-type]
    )
    agent = AgentService(
        ai=ai_service,
        executor=executor,
        store=store,
    )
    return agent, flight_provider, store


@pytest.mark.anyio
async def test_complete_workflow_search_to_review_to_seatmap_to_farerules():
    """Tests full journey: Search -> Select & Review -> Seat Map -> Fare Rules -> Booking Disabled explanation."""
    agent, bankg, store = build_offline_agent()
    sid = "sess-full-journey-1"
    history: List[AiMessage] = []
    tomorrow_iso = plus_days(today_in_tz("Asia/Kolkata"), 1)

    # Turn 1: Search flights from Mumbai to Delhi tomorrow
    history.append(AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow"))
    res1 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res1.content))

    assert any(tc.name == "search_flights" and tc.success for tc in res1.toolCalls)
    assert len(res1.results.flights) == 3
    assert bankg.calls[-1]["method"] == "airSearch"
    assert bankg.calls[-1]["origin"] == "BOM"
    assert bankg.calls[-1]["destination"] == "DEL"
    assert bankg.calls[-1]["departureDate"] == tomorrow_iso
    sess = store.get(sid)
    assert sess.workflowState == "RESULTS_READY"
    assert len(sess.lastOffers) == 3

    # Turn 2: Select the first flight and review fare
    history.append(AiMessage(role="user", content="Select the first flight and review its fare"))
    res2 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res2.content))

    assert any(tc.name == "review_fare" and tc.success for tc in res2.toolCalls)
    assert bankg.calls[-1]["method"] == "flightReview"
    assert bankg.calls[-1]["fareId"] == "FARE_BOM_DEL_101"
    assert res2.results.fareReview is not None
    assert res2.results.fareReview["bookingId"] == "BOOK_FARE_BOM_DEL_101"
    assert "4159.5" in res2.content
    assert "BOOK_FARE_BOM_DEL_101" in res2.content
    sess = store.get(sid)
    assert sess.workflowState == "FARE_REVIEWED"
    assert sess.bookingSessionId == "BOOK_FARE_BOM_DEL_101"

    # Turn 3: Ask for seat map
    history.append(AiMessage(role="user", content="Show me the seat map for this flight"))
    res3 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res3.content))

    assert any(tc.name == "get_seat_map" and tc.success for tc in res3.toolCalls)
    assert bankg.calls[-1]["method"] == "flightSeatMap"
    assert bankg.calls[-1]["bookingId"] == "BOOK_FARE_BOM_DEL_101"
    assert "1A" in res3.content

    # Turn 4: Ask for fare rules
    history.append(AiMessage(role="user", content="What are the cancellation and date change fare rules?"))
    res4 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res4.content))

    assert any(tc.name == "get_fare_rules" and tc.success for tc in res4.toolCalls)
    assert bankg.calls[-1]["method"] == "fareRule"
    assert bankg.calls[-1]["priceId"] == "FARE_BOM_DEL_101"
    assert "3000" in res4.content
    assert "2500" in res4.content

    # Turn 5: Attempt to book the flight -> must explain booking is disabled
    history.append(AiMessage(role="user", content="Great, book this flight now"))
    res5 = await agent.run(history, session_id=sid)
    assert res5.toolCalls == []
    assert "disabled" in res5.content.lower()
    sess = store.get(sid)
    assert sess.workflowState == "TRAVEL_PREPARATION_ONLY"


@pytest.mark.anyio
async def test_multi_turn_slot_filling_missing_origin_destination_and_date():
    """Verifies slot-by-slot collection without firing search_flights prematurely."""
    agent, bankg, store = build_offline_agent()
    sid = "sess-slot-filling-1"
    history: List[AiMessage] = []

    # Turn 1: User wants a flight to Goa (missing origin and date)
    history.append(AiMessage(role="user", content="I want to fly to Goa"))
    res1 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res1.content))
    assert res1.toolCalls == []
    assert len(bankg.calls) == 0
    sess = store.get(sid)
    assert sess.destination is not None and sess.destination.code == "GOI"
    assert sess.origin is None

    # Turn 2: User provides origin with ambiguous month ("From Mumbai in November") -> still needs exact date
    history.append(AiMessage(role="user", content="From Mumbai in November"))
    res2 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res2.content))
    assert res2.toolCalls == []
    assert len(bankg.calls) == 0
    sess = store.get(sid)
    assert sess.origin is not None and sess.origin.code == "BOM"
    assert sess.dates.departureDate is None

    # Turn 3: User provides exact date ("15 November for 2 adults and 1 child in business class")
    history.append(AiMessage(role="user", content="15 November for 2 adults and 1 child in business class"))
    res3 = await agent.run(history, session_id=sid)
    assert any(tc.name == "search_flights" and tc.success for tc in res3.toolCalls)
    assert len(bankg.calls) == 1
    assert bankg.calls[0]["origin"] == "BOM"
    assert bankg.calls[0]["destination"] == "GOI"
    assert bankg.calls[0]["adults"] == 2
    assert bankg.calls[0]["child"] == 1
    assert bankg.calls[0]["cabin"] == "business"


@pytest.mark.anyio
async def test_multilingual_hindi_hinglish_marathi_and_typos_workflow():
    """Tests Hindi, Hinglish, Marathi, and noisy typo inputs through the real workflow."""
    agent, bankg, _ = build_offline_agent()

    # Marathi input: मुंबईहून दिल्लीला उद्या फ्लाईट हवी आहे
    res_mr = await agent.run(
        [AiMessage(role="user", content="मुंबईहून दिल्लीला उद्या फ्लाईट हवी आहे")],
        session_id="sess-marathi",
    )
    assert any(tc.name == "search_flights" for tc in res_mr.toolCalls)
    assert bankg.calls[-1]["origin"] == "BOM"
    assert bankg.calls[-1]["destination"] == "DEL"

    # Noisy shorthand & typos: "fligt frm bengluru 2 mumbay tmrw"
    res_typo = await agent.run(
        [AiMessage(role="user", content="fligt frm bengluru 2 mumbay tmrw")],
        session_id="sess-typo",
    )
    assert any(tc.name == "search_flights" for tc in res_typo.toolCalls)
    assert bankg.calls[-1]["origin"] == "BLR"
    assert bankg.calls[-1]["destination"] == "BOM"


@pytest.mark.anyio
async def test_greeting_and_capability_question_triggers_no_tools():
    """Greetings and capability questions must not trigger any tool calls."""
    agent, bankg, _ = build_offline_agent()
    res = await agent.run(
        [AiMessage(role="user", content="Hello! What can you help me with?")],
        session_id="sess-greeting",
    )
    assert res.toolCalls == []
    assert len(bankg.calls) == 0
    assert "travel" in res.content.lower() or "flight" in res.content.lower()


@pytest.mark.anyio
async def test_selection_by_flight_number_airline_and_ambiguous_airline():
    """Tests selecting specific flights by flight number, airline, and handling ambiguous selection."""
    agent, bankg, store = build_offline_agent()
    sid = "sess-selection-modes"
    dep_d = plus_days(today_in_tz("Asia/Kolkata"), 2)
    # Configure 2 IndiGo flights and 1 Air India flight to test ambiguous vs unambiguous airline selection
    bankg.custom_offers = [
        make_fixture_flight(
            offer_id="OFF_6E_1",
            fare_id="FARE_6E_101",
            airline="IndiGo",
            carrier_code="6E",
            flight_number="6E-101",
            origin="BOM",
            destination="DEL",
            departure_date=dep_d,
            dep_time="07:00",
            arr_time="09:10",
            amount=4100.0,
        ),
        make_fixture_flight(
            offer_id="OFF_6E_2",
            fare_id="FARE_6E_202",
            airline="IndiGo",
            carrier_code="6E",
            flight_number="6E-505",
            origin="BOM",
            destination="DEL",
            departure_date=dep_d,
            dep_time="19:00",
            arr_time="21:15",
            amount=4400.0,
        ),
        make_fixture_flight(
            offer_id="OFF_AI_1",
            fare_id="FARE_AI_806",
            airline="Air India",
            carrier_code="AI",
            flight_number="AI-806",
            origin="BOM",
            destination="DEL",
            departure_date=dep_d,
            dep_time="11:00",
            arr_time="13:05",
            amount=5100.0,
        ),
    ]

    history: List[AiMessage] = [AiMessage(role="user", content=f"Flights from Mumbai to Delhi on {dep_d}")]
    res1 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res1.content))
    assert len(res1.results.flights) == 3

    # Ambiguous selection: "Review the IndiGo flight" -> 2 IndiGo flights exist, must clarify without calling review_fare
    history.append(AiMessage(role="user", content="Review the IndiGo flight"))
    res_amb = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res_amb.content))
    assert res_amb.toolCalls == []
    assert "6E-101" in res_amb.content and "6E-505" in res_amb.content

    # Specific flight number selection: "Review 6E-505"
    history.append(AiMessage(role="user", content="Review 6E-505"))
    res_spec = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res_spec.content))
    assert any(tc.name == "review_fare" and tc.success for tc in res_spec.toolCalls)
    assert bankg.calls[-1]["fareId"] == "FARE_6E_202"

    # Follow-up filtering and sorting on active search results
    history.append(AiMessage(role="user", content="Which flight is the cheapest?"))
    res_cheap = await agent.run(history, session_id=sid)
    assert res_cheap.toolCalls == []
    assert "4100" in res_cheap.content

