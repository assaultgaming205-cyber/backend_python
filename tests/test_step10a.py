import asyncio
import json
import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.agent_service import AgentService
from app.agent.clarification import clarification_store
from app.agent.intent.trip_intent_parser import TripIntentParser
from app.agent.resolvers.location_resolver import LocationResolver
from app.agent.session import TravelSession
from app.ai.ai_service import AiMessage
from app.config import settings
from app.schemas.chat import ToolCallSummary
from app.schemas.flight import FlightBags, FlightLeg, FlightOffer, FlightSegment, Money
from app.travel.flights.bankg_flight_provider import BankGFlightProvider

TODAY = date(2026, 10, 24)


def make_flight(carrier_code="AI", carrier_name="Air India", cabin="economy", amount=5000):
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
        offerId=f"test_offer_{carrier_code}_{cabin}",
        provider="BankG",
        price=Money(amount=amount, currency="INR", verified=True),
        outbound=leg,
        cabinClass=cabin,
        bags=FlightBags(carryOn=1, checked=None),
        priceBasis="per_person",
    )


class TestStep10ACabinClassAndPassengers(unittest.IsolatedAsyncioTestCase):
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

    # --- PART A: Cabin Class Tests ---

    def test_01_cabin_intent_parsing_and_normalization(self):
        """1 & 2: First Class and speech variants like 'pharst class' parse to 'first'."""
        s1 = TravelSession(sessionId="s1")
        TripIntentParser.parse_turn("First class flights chahiye Mumbai se Jaipur", s1)
        self.assertEqual(s1.cabinClass, "first")

        s2 = TravelSession(sessionId="s2")
        TripIntentParser.parse_turn("Pharst class mein search karo", s2)
        self.assertEqual(s2.cabinClass, "first")

        s3 = TravelSession(sessionId="s3")
        TripIntentParser.parse_turn("Business class chahiye", s3)
        self.assertEqual(s3.cabinClass, "business")

        s4 = TravelSession(sessionId="s4")
        TripIntentParser.parse_turn("Premium economy dikhao", s4)
        self.assertEqual(s4.cabinClass, "premium_economy")

        s5 = TravelSession(sessionId="s5")
        TripIntentParser.parse_turn("Economy mein sabse sasti flight chahiye", s5)
        self.assertEqual(s5.cabinClass, "economy")

    def test_02_all_classes_comparison_intent(self):
        """10: Compare all classes sets compareAllCabins flag."""
        s = TravelSession(sessionId="s_all")
        TripIntentParser.parse_turn("Sabhi classes compare karo Mumbai se Delhi", s)
        self.assertTrue(s.compareAllCabins)
        self.assertEqual(s.cabinClass, "all")

    def test_03_cabin_classes_distinct(self):
        """3, 4, 5: Business is not First, Premium Economy is not Economy."""
        cabins = ["economy", "premium_economy", "business", "first"]
        self.assertEqual(len(set(cabins)), 4)
        for c in cabins:
            s = TravelSession(sessionId=f"s_{c}")
            TripIntentParser.parse_turn(f"{c.replace('_', ' ')} flights", s)
            self.assertEqual(s.cabinClass, c)

    async def test_04_provider_filtering_never_substitutes_economy_for_first(self):
        """6, 8, 11: Provider filters strictly and does not return Economy offers for First Class."""
        provider = BankGFlightProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "options": {
                "responseCode": "0",
                "data": {
                    "journeys": [
                        {
                            "flights": [
                                {
                                    "airline": {"name": "IndiGo", "code": "6E"},
                                    "flightNumber": "202",
                                    "departure": {"airportCode": "BOM", "dateTime": "2026-10-25T08:00:00"},
                                    "arrival": {"airportCode": "DEL", "dateTime": "2026-10-25T10:15:00"},
                                    "durationMinutes": 135,
                                }
                            ],
                            "fares": [
                                {
                                    "fareId": "fare_1",
                                    "price": {"totalFare": 4500, "baseFare": 3800, "tax": 700, "currency": "INR"},
                                    "cabinClass": "ECONOMY",
                                }
                            ],
                        }
                    ]
                },
            }
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            # Searching for First class should return 0 offers because raw data only has ECONOMY
            offers = await provider.search_flights(
                origin="BOM", destination="DEL", departure_date="2026-10-25", adults=1, cabin_class="first"
            )
            self.assertEqual(len(offers), 0)

            # Searching for Economy should return the offer
            offers_eco = await provider.search_flights(
                origin="BOM", destination="DEL", departure_date="2026-10-25", adults=1, cabin_class="economy"
            )
            self.assertEqual(len(offers_eco), 1)
            self.assertEqual(offers_eco[0].cabinClass, "economy")

    async def test_05_first_class_no_inventory_emits_alternative_card_not_economy_cards(self):
        """24 & 25: First Class request with zero inventory does not show Economy cards; emits alternative card."""
        executor_mock = MagicMock()

        # Executor returns empty flights for First Class
        async def fake_search(name, kwargs, results):
            return ToolCallSummary(name=name, success=True, durationMs=20, error=None)

        executor_mock.execute = AsyncMock(side_effect=fake_search)
        self.agent.executor = executor_mock

        clarifications = []
        cards_emitted = []
        history = [
            AiMessage(role="user", content="Mumbai se Jaipur 25 October ko First Class flight chahiye 1 passenger ke liye.")
        ]

        res = await self.agent.run(
            history=history,
            session_id="sess_fc_empty",
            on_clarification=lambda c: clarifications.append(c),
            on_card=lambda ctype, cdata: cards_emitted.append((ctype, cdata)),
        )

        # 1. No flight cards emitted as matching First Class
        flight_cards = [c for c in cards_emitted if c[0] == "flight"]
        self.assertEqual(len(flight_cards), 0)

        # 2. Alternative clarification card emitted
        self.assertTrue(len(clarifications) >= 1)
        card = clarifications[-1]
        self.assertEqual(card["field"], "cabin")
        self.assertLessEqual(len(card["options"]), 3)
        option_values = [o["value"] for o in card["options"]]
        self.assertIn("business", option_values)
        self.assertIn("economy", option_values)

        # 3. Response truthfully explains First Class is unavailable on this route
        self.assertIn("first class", res.content.lower())
        self.assertIn("available nahi", res.content.lower())

    async def test_06_search_all_cabins_runs_bounded_searches(self):
        """10 & 26: Sabhi classes compare karo searches each supported class and ranks without inventing."""
        executor_mock = MagicMock()
        searched_cabins = []

        async def fake_search(name, kwargs, results):
            if name == "search_flights":
                c = kwargs.get("cabin", "economy")
                searched_cabins.append(c)
                results.flights.append(make_flight("AI", "Air India", cabin=c, amount=5000 if c == "economy" else 15000))
            return ToolCallSummary(name=name, success=True, durationMs=20, error=None)

        executor_mock.execute = AsyncMock(side_effect=fake_search)
        self.agent.executor = executor_mock

        history = [
            AiMessage(role="user", content="Mumbai se Jaipur 25 October sabhi classes compare karo 1 adult")
        ]

        res = await self.agent.run(
            history=history,
            session_id="sess_compare_all",
        )

        # All supported Gateway v3.2 cabins were searched
        self.assertEqual(set(searched_cabins), {"economy", "business", "first"})
        self.assertEqual(len(searched_cabins), 3)

    # --- PART B: Passenger Count Tests ---

    def test_07_passenger_intent_parsing(self):
        """16: Passenger counts parsed correctly from Hindi/English text."""
        cases = [
            ("2 adults ke liye flight", 2),
            ("do log travel karenge", 2),
            ("teen passengers", 3),
            ("4 persons", 4),
            ("char log", 4),
            ("ek passenger", 1),
        ]
        for text, expected in cases:
            s = TravelSession(sessionId="s_pax")
            TripIntentParser.parse_turn(text, s)
            self.assertEqual(s.passengers.adults, expected)
            self.assertTrue(s.passengers.isExplicit)

    async def test_08_missing_passenger_count_triggers_clarification_card(self):
        """15, 13: Missing passenger count triggers a clarification card with <= 3 options and custom text."""
        executor_mock = MagicMock()
        executor_mock.execute = AsyncMock(return_value=ToolCallSummary(name="test", success=True, durationMs=10))
        self.agent.executor = executor_mock

        clarifications = []
        history = [
            AiMessage(role="user", content="Mumbai se Jaipur 25 October ko flight tickets chahiye")
        ]

        res = await self.agent.run(
            history=history,
            session_id="sess_missing_pass",
            on_clarification=lambda c: clarifications.append(c),
        )

        # Preflight stops search and emits passenger clarification card
        executor_mock.execute.assert_not_called()
        self.assertEqual(len(clarifications), 1)
        card = clarifications[0]
        self.assertEqual(card["field"], "passengers")
        self.assertIn("kitne log", card["question"].lower())
        self.assertLessEqual(len(card["options"]), 3)
        self.assertTrue(card.get("allowCustom"))

    async def test_09_passenger_count_preserves_across_turns_and_passes_to_supplier(self):
        """18, 20, 23: Answering passenger count resumes search with adults=2 and preserves origin/destination."""
        executor_mock = MagicMock()

        async def fake_search(name, kwargs, results):
            results.flights.append(make_flight("AI", "Air India", cabin="economy", amount=9000))
            return ToolCallSummary(name=name, success=True, durationMs=20, error=None)

        executor_mock.execute = AsyncMock(side_effect=fake_search)
        self.agent.executor = executor_mock

        history = [
            AiMessage(role="user", content="Mumbai se Jaipur 25 October ko flight tickets chahiye"),
            AiMessage(role="assistant", content="Kitne log travel karenge?"),
            AiMessage(role="user", content="2"),
        ]

        res = await self.agent.run(
            history=history,
            session_id="sess_pass_flow",
            clarification_answer={"field": "passengers", "value": "2"},
        )

        self.assertEqual(executor_mock.execute.call_count, 1)
        call_args = executor_mock.execute.call_args[0]
        self.assertEqual(call_args[0], "search_flights")
        self.assertEqual(call_args[1]["adults"], 2)
        self.assertEqual(call_args[1]["origin"], "BOM")
        self.assertEqual(call_args[1]["destination"], "JAI")

    async def test_10_price_basis_per_person_vs_party_total(self):
        """19: Provider labels price basis as per_person for 1 adult, party_total for >1 adults."""
        provider = BankGFlightProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "options": {
                "responseCode": "0",
                "data": {
                    "journeys": [
                        {
                            "flights": [
                                {
                                    "airline": {"name": "Air India", "code": "AI"},
                                    "flightNumber": "101",
                                    "departure": {"airportCode": "BOM", "dateTime": "2026-10-25T08:00:00"},
                                    "arrival": {"airportCode": "DEL", "dateTime": "2026-10-25T10:15:00"},
                                    "durationMinutes": 135,
                                }
                            ],
                            "fares": [
                                {
                                    "fareId": "fare_1",
                                    "price": {"totalFare": 10000, "baseFare": 8500, "tax": 1500, "currency": "INR"},
                                    "cabinClass": "ECONOMY",
                                }
                            ],
                        }
                    ]
                },
            }
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            offers_1 = await provider.search_flights(
                origin="BOM", destination="DEL", departure_date="2026-10-25", adults=1, cabin_class="economy"
            )
            self.assertEqual(offers_1[0].priceBasis, "per_person")

            offers_2 = await provider.search_flights(
                origin="BOM", destination="DEL", departure_date="2026-10-25", adults=2, cabin_class="economy"
            )
            self.assertEqual(offers_2[0].priceBasis, "party_total")

    # --- PART C: Failure 1 - Location Validation ("Lise") ---

    def test_11_unrecognized_origin_lise_rejected(self):
        """Failure 1: 'Lise' is not recognized as a supported city/airport, sets pendingClarification."""
        self.assertFalse(LocationResolver.is_supported_location("Lise"))
        self.assertFalse(LocationResolver.is_supported_location("lise"))

        session = TravelSession(sessionId="s_lise")
        session.mode = "train"
        TripIntentParser.parse_turn("Lise se Mumbai ke liye trains", session)

        # Origin must NOT be accepted as Lise
        self.assertIsNone(session.origin)
        # Must flag pendingClarification for unrecognized origin
        self.assertEqual(session.pendingClarification, "unrecognized_origin:lise")

    async def test_12_unrecognized_origin_blocks_supplier_and_emits_clarification(self):
        """Failure 1 E2E: 'Lise se Mumbai ke liye trains' does not call search_trains; emits clarification card."""
        executor_mock = MagicMock()
        executor_mock.execute = AsyncMock(return_value=ToolCallSummary(name="test", success=True, durationMs=10))
        self.agent.executor = executor_mock
        ai_mock = MagicMock()
        ai_mock.message = MagicMock(content="Lise ek recognized city nahi hai, aap kahan se travel karna chahte ho?")
        ai_mock.model = "mock-model"
        self.agent.ai.chat = AsyncMock(return_value=ai_mock)

        clarifications = []
        history = [
            AiMessage(role="user", content="Lise se Mumbai ke liye trains")
        ]

        res = await self.agent.run(
            history=history,
            session_id="sess_lise_test",
            on_clarification=lambda c: clarifications.append(c),
        )

        # 1. No train search executed
        executor_mock.execute.assert_not_called()

        # 2. Clarification card emitted for origin with <= 3 options
        self.assertEqual(len(clarifications), 1)
        card = clarifications[0]
        self.assertEqual(card["field"], "origin")
        self.assertIn("lise", card["question"].lower())
        self.assertLessEqual(len(card["options"]), 3)
        self.assertTrue(card.get("allowCustom"))

        # 3. Response clarifies politely that Lise is unrecognized
        self.assertIn("lise", res.content.lower())
        self.assertTrue("kaunse" in res.content.lower() or "kahan se" in res.content.lower())

    # --- PART D: TTS Model Integrity ---

    def test_13_bulbul_v4_flash_exclusive(self):
        """Hard requirement 10: bulbul:v4-flash remains the only permitted TTS model."""
        self.assertEqual(settings.SARVAM_TTS_MODEL, "bulbul:v4-flash")


if __name__ == "__main__":
    unittest.main()
