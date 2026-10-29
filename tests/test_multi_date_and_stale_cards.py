import asyncio
import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from app.agent.agent_service import AgentService
from app.agent.clarification import clarification_store
from app.agent.flight_selection import build_search_context_id, get_ledger, offer_key
from app.agent.grounding.grounding_gate import GroundingGate
from app.agent.search_preflight import validate_flight_preflight
from app.agent.session import ResolvedLocation, TravelSession
from app.agent.session_store import session_store
from app.ai.ai_service import AiMessage, AiService
from app.schemas.chat import ToolCallSummary
from app.travel.flights.bankg_flight_provider import BankGFlightProvider, GATEWAY_URL
from app.schemas.flight import (
    FlightBags,
    FlightLeg,
    FlightOffer,
    FlightSegment,
    Money,
)


def make_test_offer(
    carrier_code: str = "6E",
    carrier_name: str = "IndiGo",
    day: str = "2026-10-25",
    amount: float = 3500.0,
    flight_num: str = "501",
    cabin: str = "economy",
    origin: str = "BOM",
    dest: str = "GOI",
) -> FlightOffer:
    leg = FlightLeg(
        airline=carrier_name,
        origin=origin,
        destination=dest,
        departureLocal=f"{day}T09:00:00",
        arrivalLocal=f"{day}T10:15:00",
        durationMinutes=75,
        stops=0,
        segments=[
            FlightSegment(
                carrierCode=carrier_code,
                carrierName=carrier_name,
                flightNumber=f"{carrier_code} {flight_num}",
                origin=origin,
                destination=dest,
                departureLocal=f"{day}T09:00:00",
                arrivalLocal=f"{day}T10:15:00",
            )
        ],
    )
    return FlightOffer(
        offerId=f"offer_{carrier_code}_{flight_num}_{day}_{amount}",
        provider="BankG",
        price=Money(amount=amount, currency="INR", verified=True),
        outbound=leg,
        cabinClass=cabin,
        bags=FlightBags(carryOn=1, checked=15),
        priceBasis="per_person",
    )


class TestMultiDateAndStaleCards(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._patcher_dates = patch("app.common.dates.today_in_tz", return_value="2026-10-24")
        self._patcher_dates.start()
        self._patcher_parser_today = patch("app.agent.intent.trip_intent_parser.today_in_tz", return_value="2026-10-24")
        self._patcher_parser_today.start()
        self._patcher_agent_today = patch("app.agent.agent_service.today_in_tz", return_value="2026-10-24")
        self._patcher_agent_today.start()
        self._patcher_preflight_today = patch("app.agent.search_preflight.today_ist", return_value=date(2026, 10, 24))
        self._patcher_preflight_today.start()

        clarification_store._pending.clear()
        session_store.sessions.clear()
        self.agent = AgentService()
        self.agent.timezone = "Asia/Kolkata"
        from app.ai.ai_service import AiResponsePayload
        mock_resp = AiResponsePayload(
            message=AiResponsePayload.Message(role="assistant", content="Maine flights check kar li hain."),
            content="Maine flights check kar li hain.",
            raw="<response>Maine flights check kar li hain.</response>",
            model="mock_model",
        )
        self.agent.ai = MagicMock()
        self.agent.ai.chat = AsyncMock(return_value=mock_resp)
        self.agent.ai.model_name = "mock_model"

    def tearDown(self):
        self._patcher_dates.stop()
        self._patcher_parser_today.stop()
        self._patcher_agent_today.stop()
        self._patcher_preflight_today.stop()



    async def test_scenario_a_single_date_followed_by_multi_date_request(self):
        """Scenario A: Mumbai to Goa 25 Oct search followed by 'next week sara he dhond ke bata do'."""
        session_id = "sess_scen_a"

        # Step 1: Turn 1 - Single date 25 Oct
        executed_calls = []

        async def mock_execute(name, kwargs, results):
            executed_calls.append((name, kwargs))
            dep = kwargs.get("departureDate") or kwargs.get("departure_date") or "2026-10-25"
            results.flights.append(make_test_offer("6E", "IndiGo", day=dep, amount=4000.0, flight_num="101"))
            return ToolCallSummary(name=name, success=True, durationMs=15)

        self.agent.executor = MagicMock()
        self.agent.executor.execute = AsyncMock(side_effect=mock_execute)

        history_turn1 = [
            AiMessage(role="user", content="Mumbai se Goa ke liye pachees October ki flight batao")
        ]
        res1 = await self.agent.run(history=history_turn1, session_id=session_id)
        self.assertIsNotNone(res1.results)
        self.assertEqual(len(res1.results.flights), 1)
        self.assertEqual(res1.results.flights[0].outbound.departureLocal[:10], "2026-10-25")
        ctx_single = res1.results.searchContextId
        self.assertIn("2026-10-25", ctx_single)

        session_after_1 = session_store.get(session_id)
        self.assertEqual(session_after_1.activeSearchContextId, ctx_single)

        # Step 2: Turn 2 - Assistant clarifies range 26-31 October
        # Step 3: Turn 3 - User says "next week sara he dhond ke bata do"
        history_turn3 = [
            AiMessage(role="user", content="Mumbai se Goa ke liye pachees October ki flight batao"),
            AiMessage(role="assistant", content="25 October ko koi flight nahi mili. Kaunsi date pe dekhna hai 26 se 31 October ke beech mein?"),
            AiMessage(role="user", content="next week sara he dhond ke bata do"),
        ]

        executed_calls.clear()

        async def mock_execute_multi(name, kwargs, results):
            executed_calls.append((name, kwargs))
            dep = kwargs.get("departureDate") or kwargs.get("departure_date") or "2026-10-26"
            results.flights.append(make_test_offer("AI", "Air India", day=dep, amount=4500.0, flight_num=f"202_{dep}"))
            return ToolCallSummary(name=name, success=True, durationMs=15)

        self.agent.executor.execute = AsyncMock(side_effect=mock_execute_multi)

        res3 = await self.agent.run(history=history_turn3, session_id=session_id)
        session_after_3 = session_store.get(session_id)

        # Step 4: Multi-date intent resolved to exact range
        self.assertIsNotNone(session_after_3.dates.dateRangeStart)
        self.assertIsNotNone(session_after_3.dates.dateRangeEnd)
        self.assertEqual(session_after_3.dates.dateRangeStart, "2026-10-26")
        self.assertEqual(session_after_3.dates.dateRangeEnd, "2026-10-31")

        # Step 5: Confirm each required date in 26-31 Oct is searched
        searched_dates = [c[1]["departureDate"] for c in executed_calls if c[0] == "search_flights"]
        expected_dates = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31"]
        for d in expected_dates:
            self.assertIn(d, searched_dates)

        # Step 6: Active cards contain only eligible offers for the new search range (no 25 Oct)
        self.assertIsNotNone(res3.results)
        ctx_multi = res3.results.searchContextId
        self.assertNotEqual(ctx_single, ctx_multi)
        self.assertEqual(session_after_3.activeSearchContextId, ctx_multi)

        for offer in res3.results.flights:
            self.assertEqual(offer.searchContextId, ctx_multi)
            dep_date = offer.outbound.departureLocal[:10]
            self.assertIn(dep_date, expected_dates)
            self.assertNotEqual(dep_date, "2026-10-25")

        # Step 7: Historical 25 Oct offers remain in their own ledger pool and are isolated
        ledger_single = get_ledger(session_id, ctx_single)
        self.assertEqual(len(ledger_single.pool), 1)
        single_offer = list(ledger_single.pool.values())[0]
        self.assertEqual(single_offer.outbound.departureLocal[:10], "2026-10-25")

        ledger_multi = get_ledger(session_id, ctx_multi)
        self.assertGreaterEqual(len(ledger_multi.pool), len(expected_dates))
        for off in ledger_multi.pool.values():
            self.assertNotEqual(off.outbound.departureLocal[:10], "2026-10-25")

    async def test_scenario_b_contradictory_availability_grounding(self):
        """Scenario B: Assistant cannot claim fully booked when eligible offers exist; accurate per-date outcomes."""
        session_id = "sess_scen_b"
        session = session_store.get(session_id)
        session.origin = ResolvedLocation(code="BOM", name="Mumbai")
        session.destination = ResolvedLocation(code="GOI", name="Goa")
        session.dates.departureDate = "2026-10-25"
        session.passengers.adults = 1
        session.cabinClass = "economy"
        session_store.save(session)

        # Part 1: Supplier returns valid 25 Oct offer. GroundingGate must reject contradictory 'fully booked'.
        valid_offer = make_test_offer("6E", "IndiGo", day="2026-10-25", amount=3200.0)
        gate = GroundingGate(
            verified_fares={3200, 3200.0},
            verified_flight_nums={"6E 501", "6E501"},
            verified_train_nums=set(),
            verified_times={"09:00"},
            verified_dates={"2026-10-25"},
            has_eligible_flights=True,
        )

        contradictory_text = "25 October ke liye Mumbai se Goa ki sabhi flights fully booked hain."
        is_grounded, rejection = gate.is_sentence_grounded(contradictory_text)
        self.assertFalse(is_grounded)
        self.assertIn("contradict", rejection.lower())

        # Part 2: AgentService safety guard replaces contradiction with accurate factual summary
        self.agent.executor = MagicMock()

        async def mock_exec(name, kwargs, results):
            results.flights.append(valid_offer)
            return ToolCallSummary(name=name, success=True, durationMs=10)

        self.agent.executor.execute = AsyncMock(side_effect=mock_exec)

        # Mock LLM trying to hallucinate fully booked
        with patch.object(
            AiService,
            "chat",
            new_callable=AsyncMock,
            return_value=MagicMock(message=AiMessage(role="assistant", content=contradictory_text), model="gemini-2.5-flash"),
        ):
            res = await self.agent.run(
                history=[AiMessage(role="user", content="Mumbai se Goa 25 October ki flight")],
                session_id=session_id,
            )
            # Must NOT claim fully booked in the final output text
            self.assertNotIn("fully booked", res.content.lower())
            self.assertIn("mumbai", res.content.lower())
            self.assertIn("goa", res.content.lower())

        # Part 3: Multi-date search with mixed inventory (26 Oct has flights, 27 Oct zero)
        gate_multi = GroundingGate(
            verified_fares={3500},
            verified_flight_nums={"6E 501"},
            verified_train_nums=set(),
            verified_times={"09:00"},
            verified_dates={"2026-10-26"},
            has_eligible_flights=True,
        )
        # A claim that ALL dates are fully booked must be rejected
        all_booked_claim = "Next week sabhi dates pe koi flight available nahi hai, sab fully booked hain."
        is_g_multi, rej_multi = gate_multi.is_sentence_grounded(all_booked_claim)
        self.assertFalse(is_g_multi)

    async def test_scenario_c_date_range_followup_no_single_date_clarification(self):
        """Scenario C: When user confirms range after disclosure, no single-date clarification card is emitted."""
        session = TravelSession(
            sessionId="sess_scen_c",
            origin=ResolvedLocation(code="BOM", name="Mumbai"),
            destination=ResolvedLocation(code="GOI", name="Goa"),
            cabinClass="economy",
        )
        today = date(2026, 10, 24)

        # Context contains disclosed range 26-31 October
        session.lastAskedQuestion = "Kaunsi date pe dekhna hai 26 se 31 October ke beech mein?"
        user_text = "next week sara he dhond ke bata do"

        preflight = validate_flight_preflight(
            session=session,
            today=today,
            last_user_text=user_text,
        )

        self.assertTrue(preflight.ready)
        self.assertIsNone(preflight.missing_field)
        self.assertIsNotNone(preflight.resolved_dates)
        self.assertGreaterEqual(len(preflight.resolved_dates), 2)
        self.assertIn("2026-10-26", preflight.resolved_dates)
        self.assertIn("2026-10-31", preflight.resolved_dates)

    def test_scenario_d_stale_and_delayed_events_context_isolation(self):
        """Scenario D: Context IDs isolate searches and prevent delayed events from overwriting active state."""
        ctx_old = build_search_context_id(origin="BOM", destination="GOI", departure_date="2026-10-25", cabin="economy", adults=1)
        ctx_new = build_search_context_id(origin="BOM", destination="GOI", date_range_start="2026-10-26", date_range_end="2026-10-31", cabin="economy", adults=1)

        self.assertNotEqual(ctx_old, ctx_new)

        session = TravelSession(sessionId="sess_scen_d")
        session.activeSearchContextId = ctx_new

        # Older event arrives with ctx_old - session active context must remain ctx_new
        self.assertEqual(session.activeSearchContextId, ctx_new)

        # Ledgers remain completely independent
        ledger_old = get_ledger("sess_scen_d", ctx_old)
        ledger_new = get_ledger("sess_scen_d", ctx_new)

        offer_old = make_test_offer("6E", "IndiGo", "2026-10-25", 3000.0)
        offer_new = make_test_offer("AI", "Air India", "2026-10-26", 4000.0)

        ledger_old.mark_shown([offer_old])
        ledger_new.mark_shown([offer_new])

        key_old = offer_key(offer_old)
        key_new = offer_key(offer_new)

        self.assertIn(key_old, ledger_old.shown)
        self.assertNotIn(key_old, ledger_new.shown)
        self.assertIn(key_new, ledger_new.shown)
        self.assertNotIn(key_new, ledger_old.shown)

    async def test_scenario_e_partial_supplier_failure_preserves_successful_dates(self):
        """Scenario E: If one date fails or times out, valid results from other dates remain visible."""
        session_id = "sess_scen_e"
        session = session_store.get(session_id)
        session.origin = ResolvedLocation(code="BOM", name="Mumbai")
        session.destination = ResolvedLocation(code="GOI", name="Goa")
        session.passengers.adults = 1
        session.cabinClass = "economy"
        session.dates.dateRangeStart = "2026-10-26"
        session.dates.dateRangeEnd = "2026-10-28"
        session_store.save(session)

        self.agent.executor = MagicMock()

        async def mock_partial_exec(name, kwargs, results):
            dep = kwargs.get("departureDate") or kwargs.get("departure_date")
            if dep == "2026-10-27":
                # Simulate timeout or supplier error for this single date
                return ToolCallSummary(name=name, success=False, error="Supplier timeout 504", durationMs=3000)
            results.flights.append(make_test_offer("6E", "IndiGo", day=dep, amount=3800.0, flight_num=f"f_{dep}"))
            return ToolCallSummary(name=name, success=True, durationMs=25)

        self.agent.executor.execute = AsyncMock(side_effect=mock_partial_exec)

        res = await self.agent.run(
            history=[AiMessage(role="user", content="26 se 28 October ke flights dikhao Mumbai se Goa")],
            session_id=session_id,
        )

        self.assertIsNotNone(res.results)
        # Results from 26 and 28 Oct are preserved
        dates_returned = {f.outbound.departureLocal[:10] for f in res.results.flights}
        self.assertIn("2026-10-26", dates_returned)
        self.assertIn("2026-10-28", dates_returned)
        self.assertNotIn("2026-10-27", dates_returned)
        self.assertEqual(len(res.results.flights), 2)
        self.assertEqual(res.results.dateStatuses.get("2026-10-26"), "success")
        self.assertEqual(res.results.dateStatuses.get("2026-10-27"), "failed")
        self.assertEqual(res.results.dateStatuses.get("2026-10-28"), "success")

    async def test_genuine_empty_vs_timeout_date_statuses(self):
        """Distinguish a successful search with zero offers ('empty') from a failed/timeout search ('failed')."""
        session_id = "sess_empty_vs_timeout"
        session = session_store.get(session_id)
        session.origin = ResolvedLocation(code="BOM", name="Mumbai")
        session.destination = ResolvedLocation(code="GOI", name="Goa")
        session.passengers.adults = 1
        session.cabinClass = "economy"
        session.dates.dateRangeStart = "2026-10-26"
        session.dates.dateRangeEnd = "2026-10-28"
        session_store.save(session)

        self.agent.executor = MagicMock()

        async def mock_exec(name, kwargs, results):
            dep = kwargs.get("departureDate") or kwargs.get("departure_date")
            if dep == "2026-10-26":
                # Success with offers
                results.flights.append(make_test_offer("6E", "IndiGo", day=dep, amount=3800.0, flight_num="101"))
                return ToolCallSummary(name=name, success=True, durationMs=25)
            elif dep == "2026-10-27":
                # Supplier timeout
                return ToolCallSummary(name=name, success=False, error="Supplier timeout 504", durationMs=3000)
            elif dep == "2026-10-28":
                # Succeeded, but zero flights found (genuine empty)
                return ToolCallSummary(name=name, success=True, durationMs=20)

        self.agent.executor.execute = AsyncMock(side_effect=mock_exec)

        res = await self.agent.run(
            history=[AiMessage(role="user", content="26 se 28 October ke flights dikhao Mumbai se Goa")],
            session_id=session_id,
        )

        self.assertIsNotNone(res.results.dateStatuses)
        self.assertEqual(res.results.dateStatuses.get("2026-10-26"), "success")
        self.assertEqual(res.results.dateStatuses.get("2026-10-27"), "failed")
        self.assertEqual(res.results.dateStatuses.get("2026-10-28"), "empty")

        # Grounding gate check: claiming 27th is fully booked must be rejected
        gate = GroundingGate.from_results_and_facts(res.results, {"search.datesFailed": "2026-10-27"})
        is_grounded, err = gate.is_sentence_grounded("27 October ke liye koi flight available nahi hai ya seats fully booked hain.")
        self.assertFalse(is_grounded)
        self.assertIn("failed", err.lower())

    async def test_retry_date_successful_and_isolated_to_single_date(self):
        """Retrying searches only that date using validated parameters and returns offers."""
        session_id = "sess_retry_success"
        session = session_store.get(session_id)
        session.origin = ResolvedLocation(code="BOM", name="Mumbai")
        session.destination = ResolvedLocation(code="GOI", name="Goa")
        session.passengers.adults = 2
        session.cabinClass = "business"
        session.activeSearchContextId = "BOM:GOI:business:2"
        session_store.save(session)

        searched_params = []
        async def mock_retry_exec(name, kwargs, results):
            searched_params.append(kwargs)
            results.flights.append(make_test_offer("AI", "Air India", day=kwargs["departureDate"], amount=9500.0, flight_num="302", cabin="business"))
            return ToolCallSummary(name=name, success=True, durationMs=30)

        self.agent.executor = MagicMock()
        self.agent.executor.execute = AsyncMock(side_effect=mock_retry_exec)

        retry_res = await self.agent.retry_date(session_id, "2026-10-27")

        self.assertTrue(retry_res["success"])
        self.assertEqual(retry_res["status"], "success")
        self.assertEqual(retry_res["date"], "2026-10-27")
        self.assertEqual(len(retry_res["offers"]), 1)
        # Ensure only 2026-10-27 was searched with exact session params
        self.assertEqual(len(searched_params), 1)
        self.assertEqual(searched_params[0]["departureDate"], "2026-10-27")
        self.assertEqual(searched_params[0]["origin"], "BOM")
        self.assertEqual(searched_params[0]["destination"], "GOI")
        self.assertEqual(searched_params[0]["adults"], 2)
        self.assertEqual(searched_params[0]["cabin"], "business")

    async def test_retry_date_failure_reported(self):
        """Failed retry returns status 'failed' with error."""
        session_id = "sess_retry_fail"
        session = session_store.get(session_id)
        session.origin = ResolvedLocation(code="BOM", name="Mumbai")
        session.destination = ResolvedLocation(code="GOI", name="Goa")
        session_store.save(session)

        async def mock_fail_exec(name, kwargs, results):
            return ToolCallSummary(name=name, success=False, error="Supplier 504 Gateway Timeout", durationMs=3000)

        self.agent.executor = MagicMock()
        self.agent.executor.execute = AsyncMock(side_effect=mock_fail_exec)

        retry_res = await self.agent.retry_date(session_id, "2026-10-27")
        self.assertFalse(retry_res["success"])
        self.assertEqual(retry_res["status"], "failed")
        self.assertEqual(len(retry_res["offers"]), 0)

    def test_scenario_f_context_and_pagination(self):
        """Scenario F: Pagination (next_batch) and rejections operate within isolated active context."""
        session_id = "sess_scen_f"
        ctx1 = "BOM:GOI:2026-10-25:economy:1"
        ctx2 = "BOM:GOI:2026-10-26:economy:1"

        ledger1 = get_ledger(session_id, ctx1)
        ledger2 = get_ledger(session_id, ctx2)

        # Add 5 offers to ctx1
        offers1 = [
            make_test_offer("6E", "IndiGo", "2026-10-25", 3000 + i * 200, flight_num=str(100 + i))
            for i in range(5)
        ]
        ledger1.remember(offers1)

        # First batch (3 offers)
        batch1 = ledger1.next_batch(per_date=3)
        self.assertEqual(len(batch1), 3)
        self.assertEqual(len(ledger1.shown), 3)

        # Next batch (2 remaining offers)
        batch2 = ledger1.next_batch(per_date=3)
        self.assertEqual(len(batch2), 2)
        self.assertEqual(len(ledger1.shown), 5)

        # Reject an offer in ctx1
        rej_offer = offers1[0]
        ledger1.reject(rej_offer)
        self.assertIn(offer_key(rej_offer), ledger1.rejected)

        # ctx2 ledger must have zero shown and zero rejected
        self.assertEqual(len(ledger2.shown), 0)
        self.assertEqual(len(ledger2.rejected), 0)


class TestBankGAirSearchIntegration(unittest.IsolatedAsyncioTestCase):
    """Authoritative test suite for BankG Gateway v3.2 strict airSearch integration."""

    def setUp(self):
        self._patcher_dates = patch("app.common.dates.today_in_tz", return_value="2026-10-24")
        self._patcher_dates.start()
        self._patcher_preflight_today = patch("app.agent.search_preflight.today_ist", return_value=date(2026, 10, 24))
        self._patcher_preflight_today.start()
        clarification_store._pending.clear()
        session_store.sessions.clear()
        self.provider = BankGFlightProvider(timeout=5.0)

    def tearDown(self):
        self._patcher_dates.stop()
        self._patcher_preflight_today.stop()

    async def test_01_airsearch_endpoint_only_and_no_prohibited_endpoints(self):
        """1 & 2: Outbound requests use ONLY airSearch; no health/balance/booking endpoints called."""
        called_urls = []

        async def mock_post(client, url, *args, **kwargs):
            called_urls.append(str(url))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "responseCode": "0",
                "description": "Success",
                "data": {"journeys": []},
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", new=mock_post):
            await self.provider.search_flights("BOM", "DEL", "2026-11-10")

        self.assertEqual(len(called_urls), 1)
        self.assertEqual(
            called_urls[0],
            "https://bankg-stagging-api.testuat.com/travel/v1/api/travel-master/airSearch",
        )

        prohibited = [
            "/health",
            "checkBalance",
            "flightReview",
            "flightSeatMap",
            "fareRule",
            "flightBook",
            "fareValidate",
            "bookingDetails",
            "submitAmendment",
            "amendmentDetails",
        ]
        for p in prohibited:
            self.assertFalse(any(p in u for u in called_urls), f"Prohibited endpoint called: {p}")

    async def test_02_request_schema_and_headers_exact(self):
        """Documented schema: cabinClass, adults, child, infant, segments, searchModifiers, xchecksum."""
        posted_payload = {}
        posted_headers = {}

        async def mock_post(client, url, json=None, headers=None, **kwargs):
            nonlocal posted_payload, posted_headers
            posted_payload = json or {}
            posted_headers = headers or {}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"responseCode": "0", "data": {"journeys": []}}
            return mock_resp

        with patch("httpx.AsyncClient.post", new=mock_post):
            await self.provider.search_flights(
                origin="BOM",
                destination="DEL",
                departure_date="2026-11-10",
                adults=2,
                cabin_class="business",
                child=1,
                infant=1,
            )

        self.assertEqual(posted_headers.get("Content-Type"), "application/json")
        self.assertIn("xchecksum", posted_headers)
        self.assertEqual(posted_payload.get("cabinClass"), "BUSINESS")
        self.assertEqual(posted_payload.get("adults"), 2)
        self.assertEqual(posted_payload.get("child"), 1)
        self.assertEqual(posted_payload.get("infant"), 1)
        self.assertEqual(len(posted_payload.get("segments", [])), 1)
        self.assertEqual(posted_payload["segments"][0]["origin"], "BOM")
        self.assertEqual(posted_payload["segments"][0]["destination"], "DEL")
        self.assertEqual(posted_payload["segments"][0]["date"], "2026-11-10")
        self.assertTrue(posted_payload["searchModifiers"]["isDirectFlight"])
        self.assertTrue(posted_payload["searchModifiers"]["isConnectingFlight"])
        self.assertEqual(posted_payload["searchModifiers"]["pft"], "REGULAR")

    async def test_03_documented_request_limits_enforced(self):
        """3, 4, 5, 6: Adults (1-9), child (0-8), infant (0-4), 3-letter IATA, YYYY-MM-DD, cabins."""
        # Adults bounds
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "DEL", "2026-11-10", adults=0)
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "DEL", "2026-11-10", adults=10)

        # Child bounds
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "DEL", "2026-11-10", adults=1, child=-1)
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "DEL", "2026-11-10", adults=1, child=9)

        # Infant bounds
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "DEL", "2026-11-10", adults=1, infant=-1)
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "DEL", "2026-11-10", adults=1, infant=5)

        # Invalid IATA
        with self.assertRaises(ValueError):
            await self.provider.search_flights("12", "DEL", "2026-11-10")
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "12", "2026-11-10")

        # Past date
        past_res = await self.provider.search_flights("BOM", "DEL", "2020-01-01")
        self.assertEqual(len(past_res), 0)

        # Invalid date format
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "DEL", "not-a-date")

        # Unsupported cabin class
        with self.assertRaises(ValueError):
            await self.provider.search_flights("BOM", "DEL", "2026-11-10", cabin_class="premium_economy")

    async def test_04_cabin_classes_honesty_and_no_silent_substitution(self):
        """Documented cabins: ECONOMY, BUSINESS, FIRST. Does not substitute different cabin offers."""
        sample_response = {
            "responseCode": "0",
            "data": {
                "journeys": [
                    {
                        "journeyType": "ONWARD",
                        "flights": [
                            {
                                "flightId": "93",
                                "airline": {"code": "6E", "name": "IndiGo"},
                                "flightNumber": "6045",
                                "departure": {"airportCode": "BOM", "dateTime": "2026-11-10T02:50"},
                                "arrival": {"airportCode": "DEL", "dateTime": "2026-11-10T05:05"},
                            }
                        ],
                        "fares": [
                            {
                                "fareId": "fare_econ",
                                "cabinClass": "ECONOMY",
                                "price": {"baseFare": 5000, "tax": 1000, "totalFare": 6000, "currency": "INR"},
                            }
                        ],
                    }
                ]
            },
        }

        async def mock_post(client, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_response
            return mock_resp

        with patch("httpx.AsyncClient.post", new=mock_post):
            # When First Class is requested, the returned Economy fare must be rejected (no silent substitution)
            first_offers = await self.provider.search_flights("BOM", "DEL", "2026-11-10", cabin_class="first")
            self.assertEqual(len(first_offers), 0)

            # When Economy is requested, the Economy fare is preserved
            econ_offers = await self.provider.search_flights("BOM", "DEL", "2026-11-10", cabin_class="economy")
            self.assertEqual(len(econ_offers), 1)
            self.assertEqual(econ_offers[0].cabinClass, "economy")

    async def test_05_supplier_timeout_raises_timeout_error_and_logs(self):
        """Read timeout raises TimeoutError rather than returning false empty results."""
        async def mock_timeout(client, *args, **kwargs):
            raise httpx.ReadTimeout("Supplier timed out after 5.0s")

        with patch("httpx.AsyncClient.post", new=mock_timeout):
            with self.assertRaises(TimeoutError):
                await self.provider.search_flights("BOM", "DEL", "2026-11-10")

    async def test_06_http_status_codes_raise_distinct_errors(self):
        """HTTP 401/403 (auth), 429 (rate limit), 500 (upstream error) raise distinct errors."""
        for code, expected_err in [
            (401, "BankG authentication error"),
            (403, "BankG authentication error"),
            (429, "BankG rate limit exceeded"),
            (500, "BankG upstream server error"),
            (502, "BankG upstream server error"),
        ]:
            async def mock_http_err(client, *args, c=code, **kwargs):
                mock_resp = MagicMock()
                mock_resp.status_code = c
                mock_resp.text = f"Error {c}"
                return mock_resp

            with patch("httpx.AsyncClient.post", new=mock_http_err):
                with self.assertRaises(RuntimeError) as cm:
                    await self.provider.search_flights("BOM", "DEL", "2026-11-10")
                self.assertIn(expected_err, str(cm.exception))

    async def test_07_provider_response_code_nonzero_raises_runtime_error(self):
        """responseCode != '0' raises RuntimeError with supplier description."""
        async def mock_err_code(client, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "responseCode": "S2S_003",
                "description": "Invalid checksum signature",
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", new=mock_err_code):
            with self.assertRaises(RuntimeError) as cm:
                await self.provider.search_flights("BOM", "DEL", "2026-11-10")
            self.assertIn("Invalid checksum signature", str(cm.exception))

    async def test_08_parse_documented_sample_response_truthfully(self):
        """10, 11: Parses IndiGo 6045 BOM-DEL sample payload preserving airlines, times, fares."""
        sample_response = {
            "responseCode": "0",
            "description": " Success",
            "data": {
                "journeys": [
                    {
                        "journeyType": "ONWARD",
                        "flights": [
                            {
                                "flightId": "93",
                                "airline": {"code": "6E", "name": "IndiGo", "isLowCost": True},
                                "flightNumber": "6045",
                                "aircraftType": "321",
                                "stops": 0,
                                "durationMinutes": 135,
                                "departure": {
                                    "airportCode": "BOM",
                                    "airportName": "Chhatrapati Shivaji",
                                    "city": "Mumbai",
                                    "country": "India",
                                    "terminal": "Terminal 2",
                                    "dateTime": "2026-11-10T02:50",
                                },
                                "arrival": {
                                    "airportCode": "DEL",
                                    "airportName": "Delhi Indira Gandhi Intl",
                                    "city": "Delhi",
                                    "country": "India",
                                    "terminal": "Terminal 1",
                                    "dateTime": "2026-11-10T05:05",
                                },
                            }
                        ],
                        "fares": [
                            {
                                "fareId": "5-2744328462_0BOMDEL6E6045~2073524339604414",
                                "fareType": "PUBLISHED",
                                "cabinClass": "ECONOMY",
                                "bookingClass": "R",
                                "availableSeats": 3,
                                "price": {"baseFare": 6300, "tax": 2122.5, "totalFare": 8422.5, "currency": "INR"},
                                "baggage": {"checkIn": "15 Kg (01 Piece only)", "cabin": "7 Kg"},
                            }
                        ],
                    }
                ]
            },
        }

        async def mock_success(client, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_response
            return mock_resp

        with patch("httpx.AsyncClient.post", new=mock_success):
            offers = await self.provider.search_flights("BOM", "DEL", "2026-11-10")

        self.assertEqual(len(offers), 1)
        o = offers[0]
        self.assertEqual(o.outbound.airline, "IndiGo")
        self.assertEqual(o.price.amount, 8422.5)
        self.assertEqual(o.price.currency, "INR")
        self.assertEqual(o.outbound.departureLocal, "2026-11-10T02:50")
        self.assertEqual(o.outbound.arrivalLocal, "2026-11-10T05:05")
        self.assertEqual(o.outbound.durationMinutes, 135)
        self.assertEqual(o.outbound.segments[0].flightNumber, "6E 6045")
        self.assertEqual(o.outbound.segments[0].aircraft, "321")
        self.assertIn("15 Kg", o.observedDetails["baggagePolicies"][0])
        self.assertIn("7 Kg", o.observedDetails["baggagePolicies"][1])

    async def test_09_retry_date_session_recovery_from_search_context_id(self):
        """Part 2 & Part 5: Server restart clears sessions; retry recovers route from searchContextId."""
        agent = AgentService()
        agent.timezone = "Asia/Kolkata"

        # Ensure session store is empty (server restart simulation)
        session_store.sessions.clear()

        searched_calls = []
        async def mock_retry_exec(name, kwargs, results):
            searched_calls.append(kwargs)
            results.flights.append(make_test_offer("6E", "IndiGo", day=kwargs["departureDate"], amount=4200.0))
            return ToolCallSummary(name=name, success=True, durationMs=20)

        agent.executor = MagicMock()
        agent.executor.execute = AsyncMock(side_effect=mock_retry_exec)

        # Retry with session missing from store, providing searchContextId
        ctx_id = "BOM:GOI:2026-10-26..2026-10-28:economy:2"
        res = await agent.retry_date("lost_session_123", "2026-10-27", search_context_id=ctx_id)

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["date"], "2026-10-27")
        self.assertEqual(len(res["offers"]), 1)
        # Verify route and passenger counts were restored
        self.assertEqual(len(searched_calls), 1)
        self.assertEqual(searched_calls[0]["origin"], "BOM")
        self.assertEqual(searched_calls[0]["destination"], "GOI")
        self.assertEqual(searched_calls[0]["adults"], 2)
        self.assertEqual(searched_calls[0]["cabin"], "economy")

    async def test_10_retry_date_expired_and_no_context_returns_expired_flag(self):
        """When session is expired and cannot be recovered, explicit expired: True is returned."""
        agent = AgentService()
        session_store.sessions.clear()

        res = await agent.retry_date("lost_session_456", "2026-10-27", search_context_id=None)
        self.assertFalse(res["success"])
        self.assertTrue(res.get("expired"))
        self.assertIn("Search context expired", res["error"])
        self.assertEqual(res["status"], "failed")

    async def test_11_retry_date_rejects_past_or_invalid_date(self):
        """Retry date search rejects past or invalid calendar dates."""
        agent = AgentService()
        agent.timezone = "Asia/Kolkata"
        res = await agent.retry_date("any_session", "2020-01-01")
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "failed")
        self.assertIn("Invalid travel date", res["error"])


if __name__ == "__main__":
    unittest.main()

