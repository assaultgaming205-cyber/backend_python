import asyncio
import json
import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.agent_service import AgentService
from app.agent.clarification import clarification_store, date_options, normalize_clarification
from app.agent.search_preflight import validate_flight_preflight, resolve_flight_date
from app.ai.ai_service import AiMessage
from app.config import settings
from app.schemas.chat import ChatResults, ToolCallSummary, empty_results
from app.schemas.flight import FlightBags, FlightLeg, FlightOffer, FlightSegment, Money
from app.travel.airlines import canonical_airline_name


TODAY = date(2026, 10, 24)


def make_flight(carrier_code="IX", carrier_name="Air India Express", amount=4500):
    leg = FlightLeg(
        airline=carrier_name,
        origin="BOM",
        destination="JAI",
        departureLocal="2026-10-25T10:00:00",
        arrivalLocal="2026-10-25T11:45:00",
        durationMinutes=105,
        stops=0,
        segments=[
            FlightSegment(
                carrierCode=carrier_code,
                carrierName=carrier_name,
                flightNumber=f"{carrier_code} 101",
                origin="BOM",
                destination="JAI",
                departureLocal="2026-10-25T10:00:00",
                arrivalLocal="2026-10-25T11:45:00",
            )
        ],
    )
    return FlightOffer(
        offerId="test_offer_1",
        provider="BankG",
        price=Money(amount=amount, currency="INR", verified=True),
        outbound=leg,
        cabinClass="economy",
        bags=FlightBags(carryOn=1, checked=None),
    )


class TestStep9AEndToEndScenarios(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._patcher_dates = patch("app.common.dates.today_in_tz", return_value="2026-10-24")
        self._patcher_dates.start()
        self._patcher_parser_today = patch("app.agent.intent.trip_intent_parser.today_in_tz", return_value="2026-10-24")
        self._patcher_parser_today.start()
        self._patcher_agent_today = patch("app.agent.agent_service.today_in_tz", return_value="2026-10-24")
        self._patcher_agent_today.start()
        self._patcher_preflight_today = patch("app.agent.search_preflight.today_ist", return_value=TODAY)
        self._patcher_preflight_today.start()

        clarification_store._pending.clear()
        self.agent = AgentService()
        self.agent.timezone = "Asia/Kolkata"

    def tearDown(self):
        self._patcher_dates.stop()
        self._patcher_parser_today.stop()
        self._patcher_agent_today.stop()
        self._patcher_preflight_today.stop()

    async def test_scenario_1_destination_only_emits_clarification_no_supplier(self):
        """Scenario 1: User gives destination only; no supplier call, clarification emitted, origin asked."""
        executor_mock = MagicMock()
        executor_mock.execute = AsyncMock()
        self.agent.executor = executor_mock

        clarifications = []
        statuses = []
        history = [AiMessage(role="user", content="Mujhe Jaipur ke liye flight chahiye.")]

        res = await self.agent.run(
            history=history,
            session_id="sess_sc1",
            on_clarification=lambda c: clarifications.append(c),
            on_status=lambda s: statuses.append(s),
        )

        # 1. No supplier request made
        executor_mock.execute.assert_not_called()

        # 2. Clarification card emitted
        self.assertEqual(len(clarifications), 1)
        card = clarifications[0]
        self.assertEqual(card["type"], "clarification")
        self.assertEqual(card["field"], "origin")
        self.assertIn("kahan se", card["question"].lower())
        self.assertLessEqual(len(card["options"]), 3)
        self.assertTrue(card.get("allowCustom"))

        # 3. Response asks for origin without asking which flight they prefer
        self.assertIn("kahan se", res.content.lower())
        self.assertNotIn("kaunsi flight", res.content.lower())
        self.assertNotIn("which flight", res.content.lower())

    async def test_scenario_2_resolve_missing_origin_preserves_destination(self):
        """Scenario 2: User provides origin in turn 2; destination Jaipur is preserved, exactly 1 search executed."""
        fake_flight = make_flight()
        executor_mock = MagicMock()

        async def fake_search(name, kwargs, results):
            results.flights.append(fake_flight)
            return ToolCallSummary(name=name, success=True, durationMs=20, error=None)

        executor_mock.execute = AsyncMock(side_effect=fake_search)
        self.agent.executor = executor_mock

        history = [
            AiMessage(role="user", content="Mujhe Jaipur ke liye flight chahiye."),
            AiMessage(role="assistant", content="Aap kahan se travel karna chahte ho?"),
            AiMessage(role="user", content="Mumbai se."),
        ]

        cards_emitted = []
        res = await self.agent.run(
            history=history,
            session_id="sess_sc2",
            on_card=lambda ctype, cdata: cards_emitted.append((ctype, cdata)),
        )

        # Search executed exactly once
        self.assertEqual(executor_mock.execute.call_count, 1)
        call_args = executor_mock.execute.call_args[0]
        self.assertEqual(call_args[0], "search_flights")
        self.assertEqual(call_args[1]["origin"], "BOM")
        self.assertEqual(call_args[1]["destination"], "JAI")
        self.assertEqual(call_args[1]["departureDate"], "2026-10-25")

        # Flight cards delivered
        self.assertTrue(any(c[0] == "flight" for c in cards_emitted))

    async def test_scenario_3_date_already_supplied_origin_missing(self):
        """Scenario 3: Date 25 Oct supplied in turn 1; origin missing. Original date used once origin is given."""
        executor_mock = MagicMock()
        executor_mock.execute = AsyncMock()
        self.agent.executor = executor_mock

        clarifications = []
        history_t1 = [AiMessage(role="user", content="Jaipur ke flights 25 October ko chahiye.")]

        # Turn 1: Origin missing -> no supplier call
        res1 = await self.agent.run(
            history=history_t1,
            session_id="sess_sc3",
            on_clarification=lambda c: clarifications.append(c),
        )
        executor_mock.execute.assert_not_called()
        self.assertEqual(len(clarifications), 1)
        self.assertEqual(clarifications[0]["field"], "origin")

        # Turn 2: User provides origin "Mumbai"
        fake_flight = make_flight()

        async def fake_search(name, kwargs, results):
            results.flights.append(fake_flight)
            return ToolCallSummary(name=name, success=True, durationMs=20, error=None)

        executor_mock.execute = AsyncMock(side_effect=fake_search)
        history_t2 = history_t1 + [
            AiMessage(role="assistant", content="Aap kahan se travel karna chahte ho?"),
            AiMessage(role="user", content="Mumbai"),
        ]

        res2 = await self.agent.run(history=history_t2, session_id="sess_sc3")
        self.assertEqual(executor_mock.execute.call_count, 1)
        call_kwargs = executor_mock.execute.call_args[0][1]
        self.assertEqual(call_kwargs["departureDate"], "2026-10-25")
        self.assertEqual(call_kwargs["origin"], "BOM")
        self.assertEqual(call_kwargs["destination"], "JAI")

    async def test_scenario_4_origin_destination_and_date_known_searches_once_no_clarification(self):
        """Scenario 4: Complete route and date known upfront; runs once, no clarification card."""
        fake_flight = make_flight()
        executor_mock = MagicMock()

        async def fake_search(name, kwargs, results):
            results.flights.append(fake_flight)
            return ToolCallSummary(name=name, success=True, durationMs=20, error=None)

        executor_mock.execute = AsyncMock(side_effect=fake_search)
        self.agent.executor = executor_mock

        clarifications = []
        history = [AiMessage(role="user", content="Mumbai se Jaipur, 25 October.")]

        res = await self.agent.run(
            history=history,
            session_id="sess_sc4",
            on_clarification=lambda c: clarifications.append(c),
        )

        self.assertEqual(len(clarifications), 0)
        self.assertEqual(executor_mock.execute.call_count, 1)
        call_kwargs = executor_mock.execute.call_args[0][1]
        self.assertEqual(call_kwargs["origin"], "BOM")
        self.assertEqual(call_kwargs["destination"], "JAI")
        self.assertEqual(call_kwargs["departureDate"], "2026-10-25")

    async def test_scenario_5_missing_date_applies_disclosed_default(self):
        """Scenario 5: Route known, date missing; India-local tomorrow applied and disclosed."""
        executor_mock = MagicMock()

        async def fake_search(name, kwargs, results):
            results.flights.append(make_flight())
            return ToolCallSummary(name=name, success=True, durationMs=20, error=None)

        executor_mock.execute = AsyncMock(side_effect=fake_search)
        self.agent.executor = executor_mock

        statuses = []
        history = [AiMessage(role="user", content="Delhi se Mumbai flight chahiye.")]

        res = await self.agent.run(
            history=history,
            session_id="sess_sc5",
            on_status=lambda s: statuses.append(s),
        )

        self.assertEqual(executor_mock.execute.call_count, 1)
        # Default date disclosure announced
        self.assertTrue(any("kal" in s.lower() or "25" in s.lower() for s in statuses))
        call_kwargs = executor_mock.execute.call_args[0][1]
        self.assertEqual(call_kwargs["departureDate"], "2026-10-25")

    async def test_scenario_6_ambiguous_voice_transcript_purchase_october(self):
        """Scenario 6: 'Purchase October' does not invent an exact date; asks clarification; no supplier call."""
        executor_mock = MagicMock()
        executor_mock.execute = AsyncMock()
        self.agent.executor = executor_mock

        clarifications = []
        history = [
            AiMessage(role="user", content="Mumbai se Jaipur jana hai."),
            AiMessage(role="assistant", content="Kab nikalna hai? Date batayein."),
            AiMessage(role="user", content="Purchase October"),
        ]

        res = await self.agent.run(
            history=history,
            session_id="sess_sc6",
            on_clarification=lambda c: clarifications.append(c),
        )

        executor_mock.execute.assert_not_called()
        self.assertEqual(len(clarifications), 1)
        self.assertEqual(clarifications[0]["field"], "date")
        self.assertLessEqual(len(clarifications[0]["options"]), 3)

    async def test_scenario_7_empty_results_truthful_and_no_preference_question(self):
        """Scenario 7: When supplier returns 0 results, no flight preference question is asked."""
        executor_mock = MagicMock()

        async def fake_empty_search(name, kwargs, results):
            return ToolCallSummary(name=name, success=True, durationMs=20, error=None)

        executor_mock.execute = AsyncMock(side_effect=fake_empty_search)
        self.agent.executor = executor_mock

        history = [AiMessage(role="user", content="Mumbai se Jaipur, 25 October.")]
        res = await self.agent.run(history=history, session_id="sess_sc7")

        self.assertNotIn("kaunsi flight", res.content.lower())
        self.assertNotIn("which flight", res.content.lower())
        self.assertIn("flights", res.content.lower())

    async def test_scenario_8_stale_and_duplicate_cards_rejected(self):
        """Scenario 8: Duplicate or stale clarification answers are safely rejected."""
        card = normalize_clarification(
            {"field": "origin", "question": "Aap kahan se travel karna chahte ho?", "options": ["Delhi", "Mumbai"]},
            turn_id="turn_101",
        )
        clarification_store.open("sess_sc8", card)

        # 1. First answer succeeds
        ans1 = clarification_store.answer("sess_sc8", card["questionId"], "turn_101", "Delhi", TODAY)
        self.assertEqual(ans1["status"], "ok")
        self.assertEqual(ans1["value"], "Delhi")

        # 2. Immediate second tap on the same card is rejected as stale/already answered
        ans2 = clarification_store.answer("sess_sc8", card["questionId"], "turn_101", "Delhi", TODAY)
        self.assertEqual(ans2["status"], "stale")

        # 3. Wrong turnId is rejected as stale
        clarification_store.open("sess_sc8", card)
        ans3 = clarification_store.answer("sess_sc8", card["questionId"], "old_turn", "Delhi", TODAY)
        self.assertEqual(ans3["status"], "stale")

    async def test_scenario_9_stream_ordering_contract(self):
        """Scenario 9: SSE stream sequence emits turn_started -> clarification without premature cards."""
        from starlette.testclient import TestClient
        from app.main import app
        from app.ai.ai_service import AiResponsePayload, AiService

        mock_resp = AiResponsePayload(
            message=AiResponsePayload.Message(role="assistant", content="Aap kahan se travel karna chahte ho?"),
            content="Aap kahan se travel karna chahte ho?",
            raw="<response>Aap kahan se travel karna chahte ho?</response>",
            model="mock_model",
        )

        with patch.object(AiService, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_resp
            client = TestClient(app)
            resp = client.post(
                "/api/chat/stream",
                json={
                    "sessionId": "sess_sc9",
                    "character": "meera",
                    "messages": [{"role": "user", "content": "Mujhe Jaipur ke liye flights book karni hain."}],
                },
            )
        self.assertEqual(resp.status_code, 200)

        events = []
        for line in resp.text.split("\n\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        event_types = [e.get("type") for e in events]
        self.assertIn("turn_started", event_types)
        self.assertIn("clarification", event_types)
        self.assertNotIn("card", event_types)  # No supplier card before origin is collected!
        self.assertIn("done", event_types)

    def test_scenario_10_existing_functionality_invariants(self):
        """Scenario 10: Invariants on TTS model, silence windows, and canonical airline names."""
        # Bulbul v4 flash only
        self.assertEqual(getattr(settings, "SARVAM_TTS_MODEL", "bulbul:v4-flash"), "bulbul:v4-flash")
        # 2-second speech pause
        self.assertEqual(getattr(settings, "SARVAM_STT_SILENCE_DURATION_MS", 2000), 2000)
        # Canonical airline names
        self.assertEqual(canonical_airline_name("AI Express", "IX").display, "Air India Express")
        self.assertEqual(canonical_airline_name("Air India", "AI").display, "Air India")


if __name__ == "__main__":
    unittest.main()
