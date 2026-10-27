"""
RECORDED-RESPONSE REGRESSION TESTS (OFFLINE — NOT LIVE NETWORK TESTS)

These tests replay sanitized, redacted responses captured from the live BankG
UAT/Staging server (https://bankg-stagging-api.testuat.com/travel/v1/api) stored in
`tests/fixtures/bankg_live_recorded_sanitized.json`.

They prevent regressions in:
1. HTTP 201 response handling across all BankG POST endpoints.
2. Unwrapping of the live `{"options": {"responseCode": ..., "description": ..., "data": ...}}` envelope.
3. Extraction of nested `flights[].pricing`, `flights[].cancellationRules`, `flights[].dateChangeRules`,
   `flights[].noShowRules`, and `flights[].segments[].mealOptions` / `baggageOptions` in `flightReview`.
4. Preservation of genuine supplier `fareId` and `priceId` across the dependency chain:
   `airSearch -> fareId -> flightReview -> bookingId/priceId -> flightSeatMap & fareRule`.
5. Distinction between valid zero-flight responses ("No flight search results found...") and provider errors.
6. Local safety-gate rejection of prohibited operations (`flightBook`, `submitAmendment`, `fareValidate`, `checkBalance`)
   and unauthenticated `bookingDetails` / `amendmentDetails` with zero outbound HTTP calls.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from app.main import app
from app.agent.tool_executor import ToolExecutor
from app.agent.session import create_initial_session
from app.agent.grounding.grounded_reply import build_facts
from app.agent.grounding.grounding_gate import GroundingGate
from app.agent.resolvers.location_resolver import LocationResolver
from app.schemas.chat import ChatResults
from app.travel.flights.bankg_flight_provider import (
    BankGFlightProvider,
    OperationDisabledError,
    _resolve_iata,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bankg_live_recorded_sanitized.json"


def _make_httpx_response(status_code: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


class TestBankGRecordedRegression(unittest.IsolatedAsyncioTestCase):
    """Offline regression suite using recorded responses from the live BankG staging gateway."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(FIXTURE_PATH.read_text())

    async def test_01_recorded_health_404_is_reported_truthfully(self):
        """Recorded GET /health returns HTTP 404 on staging; check_health reports healthy=False without crashing."""
        provider = BankGFlightProvider()
        rec = self.fixtures["health_404"]
        mock_resp = _make_httpx_response(rec["http_status"], rec["body"])

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            res = await provider.check_health()

        self.assertEqual(res["httpStatus"], 404)
        self.assertFalse(res["healthy"])
        self.assertTrue(res["gatewayReachable"])
        self.assertFalse(res["healthRouteMounted"])
        self.assertEqual(res["status"], "HEALTH_ROUTE_UNMOUNTED")
        self.assertEqual(res["responseCode"], "1")
        self.assertIn("Cannot GET /travel/v1/api/health", res["description"])

    async def test_02_recorded_dependency_chain_airSearch_to_review_seatMap_fareRule(self):
        """
        Replays the recorded live dependency chain:
        airSearch (HTTP 201) -> fareId -> flightReview (HTTP 201) -> bookingId & priceId -> flightSeatMap & fareRule.
        """
        provider = BankGFlightProvider()
        executor = ToolExecutor(flight_provider=provider)
        results = ChatResults()

        # Step A: airSearch
        rec_search = self.fixtures["airSearch_201"]
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(rec_search["http_status"], rec_search["body"]),
        ) as mock_post:
            summary_search = await executor.execute(
                "search_flights",
                {"origin": "BOM", "destination": "DEL", "departureDate": "2099-11-10", "adults": 1, "cabin": "economy"},
                results,
            )
            self.assertTrue(summary_search.success)
            sent_headers = mock_post.call_args.kwargs["headers"]
            self.assertIn("xchecksum", sent_headers)

        self.assertGreaterEqual(len(results.flights), 3)
        first_offer = results.flights[0]
        raw_fare_id = first_offer.observedDetails.get("fareId")
        self.assertEqual(raw_fare_id, "4-9204425964_0BOMDELSG252~5951139211020472")
        self.assertEqual(first_offer.offerId, f"bankg_{raw_fare_id}")
        self.assertEqual(first_offer.price.amount, 4159.5)
        self.assertTrue(first_offer.price.verified)

        # Step B: flightReview using exact fareId extracted from airSearch
        rec_review = self.fixtures["flightReview_201"]
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(rec_review["http_status"], rec_review["body"]),
        ) as mock_post:
            summary_review = await executor.execute("review_fare", {"fareId": first_offer.offerId}, results)
            self.assertTrue(summary_review.success)
            # Ensure "bankg_" prefix was stripped before calling BankG
            self.assertEqual(mock_post.call_args.kwargs["json"], {"fareId": [raw_fare_id]})
            self.assertIn("xchecksum", mock_post.call_args.kwargs["headers"])

        rev = results.fareReview
        self.assertIsNotNone(rev)
        self.assertEqual(rev["bookingId"], "TJS106403262265")
        self.assertEqual(rev["reqId"], "9204425964")
        self.assertEqual(rev["priceId"], "R4-9204425964_0BOMDELSG252~5951152016899322")
        self.assertEqual(rev["searchPrice"], 4159.5)
        self.assertEqual(rev["reviewedPrice"], 5159.5)
        self.assertEqual(rev["totalFare"], 5159.5)
        self.assertEqual(rev["baseFare"], 2950.0)
        self.assertEqual(rev["totalTax"], 2209.5)
        self.assertEqual(rev["priceDelta"], 1000.0)
        self.assertEqual(rev["taxDelta"], 1000.0)
        self.assertEqual(rev["baseFareDelta"], 0.0)
        self.assertTrue(rev["priceChanged"])
        self.assertFalse(rev["originalPriceConfirmed"])
        self.assertEqual(rev["status"], "PRICE_CHANGED")
        self.assertIn("4159.50", rev["priceDiscrepancyNotice"])
        self.assertIn("5159.50", rev["priceDiscrepancyNotice"])
        # Original search offer price must NOT be overwritten
        self.assertEqual(first_offer.price.amount, 4159.5)
        self.assertEqual(len(rev["mealOptions"]), 21)
        self.assertEqual(len(rev["baggageOptions"]), 8)
        self.assertEqual(len(rev["cancellationRules"]), 2)
        self.assertEqual(len(rev["dateChangeRules"]), 1)
        self.assertEqual(len(rev["noShowRules"]), 1)

        # Step C: flightSeatMap using exact bookingId returned by flightReview
        rec_seat = self.fixtures["flightSeatMap_201"]
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(rec_seat["http_status"], rec_seat["body"]),
        ) as mock_post:
            summary_seat = await executor.execute("get_seat_map", {"bookingId": rev["bookingId"]}, results)
            self.assertTrue(summary_seat.success)
            self.assertEqual(mock_post.call_args.kwargs["json"], {"bookingId": "TJS106403262265"})
            self.assertIn("xchecksum", mock_post.call_args.kwargs["headers"])

        smap = results.seatMap
        self.assertEqual(smap["bookingId"], "TJS106403262265")
        self.assertEqual(len(smap["seatMaps"]), 1)
        self.assertEqual(smap["seatMaps"][0]["layout"], {"totalRows": 34, "totalColumns": 7})
        self.assertGreater(len(smap["seatMaps"][0]["seats"]), 0)

        # Step D1: fareRule using airSearch fareId
        rec_rule_a = self.fixtures["fareRule_from_airSearch_fareId_201"]
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(rec_rule_a["http_status"], rec_rule_a["body"]),
        ) as mock_post:
            summary_rule_a = await executor.execute("get_fare_rules", {"priceId": raw_fare_id}, results)
            self.assertTrue(summary_rule_a.success)
            self.assertEqual(mock_post.call_args.kwargs["json"], {"priceId": raw_fare_id})

        self.assertEqual(len(results.fareRules["fareRules"]), 1)
        tech_rule = results.fareRules["fareRules"][0]["technicalFareRule"]
        self.assertIn("CANCELLATION", tech_rule)
        self.assertIn("DATECHANGE", tech_rule)
        self.assertIn("NO_SHOW", tech_rule)

        # Step D2: fareRule using flightReview priceId
        rec_rule_b = self.fixtures["fareRule_from_flightReview_priceId_201"]
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(rec_rule_b["http_status"], rec_rule_b["body"]),
        ) as mock_post:
            summary_rule_b = await executor.execute("get_fare_rules", {"priceId": rev["priceId"]}, results)
            self.assertTrue(summary_rule_b.success)
            self.assertEqual(mock_post.call_args.kwargs["json"], {"priceId": rev["priceId"]})

    async def test_03_recorded_zero_result_response_distinguished_from_provider_error(self):
        """
        Live BankG returns responseCode='1' with description='No flight search results found...'
        when a route/cabin has 0 flights (e.g. FIRST class on BOM->JAI). Provider returns [] cleanly.
        """
        provider = BankGFlightProvider()
        rec_zero = self.fixtures["airSearch_zero_results_first_class_201"]
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(rec_zero["http_status"], rec_zero["body"]),
        ):
            offers = await provider.search_flights(
                origin="BOM", destination="JAI", departure_date="2099-11-10", adults=1, cabin_class="first"
            )
        self.assertEqual(offers, [])

    async def test_04_synthetic_fallback_fare_ids_rejected_before_network(self):
        """Synthetic fallback IDs ('bankg_0_0') must never be sent to flightReview or fareRule."""
        provider = BankGFlightProvider()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            with self.assertRaises(ValueError):
                await provider.review_fare("bankg_0_0")
            with self.assertRaises(ValueError):
                await provider.get_fare_rules("0_1")
            mock_post.assert_not_called()

    async def test_05_prohibited_and_unauthorized_operations_make_zero_network_calls(self):
        """Prohibited operations and unauthenticated lookups are rejected locally with 0 HTTP requests."""
        provider = BankGFlightProvider()
        executor = ToolExecutor(flight_provider=provider)
        results = ChatResults()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            for tool_name in ("flightBook", "book_flight", "submitAmendment", "submit_amendment", "fareValidate", "checkBalance"):
                res = await executor.execute(tool_name, {"bookingId": "TJS106403262265"}, results)
                self.assertFalse(res.success)
                self.assertIn("prohibited_operation", res.error)

            res_b = await executor.execute("get_booking_details", {"bookingId": "TJS106403262265"}, results, trusted_auth=None)
            self.assertFalse(res_b.success)
            self.assertIn("unauthorized_booking_lookup", res_b.error)

            res_a = await executor.execute("get_amendment_details", {"amendmentId": "7403197883"}, results, trusted_auth=None)
            self.assertFalse(res_a.success)
            self.assertIn("unauthorized_amendment_lookup", res_a.error)

            mock_post.assert_not_called()

    def test_06_fastapi_flights_search_endpoint_parses_recorded_response(self):
        """FastAPI POST /api/flights/search returns verified offers from recorded BankG response."""
        rec_search = self.fixtures["airSearch_201"]
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(rec_search["http_status"], rec_search["body"]),
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/flights/search",
                json={"origin": "Mumbai", "destination": "Delhi", "departureDate": "2099-11-10", "adults": 1, "cabin": "economy"},
            )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["source"], "bankg")
        self.assertGreaterEqual(payload["count"], 3)
        self.assertEqual(
            payload["results"][0]["observedDetails"]["fareId"],
            "4-9204425964_0BOMDELSG252~5951139211020472",
        )

    async def test_07_fare_comparison_identical_prices(self):
        """Priority 1: Identical search and reviewed prices mark status='LOCKED' and originalPriceConfirmed=True."""
        provider = BankGFlightProvider()
        rec_review = json.loads(json.dumps(self.fixtures["flightReview_201"]))
        # Set reviewed totalFare to 4159.5 (identical to search price)
        rec_review["body"]["options"]["data"]["flights"][0]["pricing"][0]["totalFare"] = 4159.5
        rec_review["body"]["options"]["data"]["flights"][0]["pricing"][0]["totalTax"] = 1209.5

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(201, rec_review["body"]),
        ):
            rev = await provider.review_fare(
                "4-9204425964_0BOMDELSG252~5951139211020472",
                search_price=4159.5,
                search_base_fare=2950.0,
                search_tax=1209.5,
                search_currency="INR",
            )

        self.assertEqual(rev["status"], "LOCKED")
        self.assertTrue(rev["priceVerified"])
        self.assertTrue(rev["fareIdMatched"])
        self.assertFalse(rev["priceChanged"])
        self.assertTrue(rev["originalPriceConfirmed"])
        self.assertEqual(rev["searchPrice"], 4159.5)
        self.assertEqual(rev["reviewedPrice"], 4159.5)
        self.assertEqual(rev["priceDelta"], 0.0)
        self.assertEqual(rev["taxDelta"], 0.0)
        self.assertIsNone(rev["priceDiscrepancyNotice"])

    async def test_08_fare_comparison_missing_prices(self):
        """Priority 1: Missing or zero reviewed price marks status='PRICE_MISSING' and never confirms original price."""
        provider = BankGFlightProvider()
        rec_missing = json.loads(json.dumps(self.fixtures["flightReview_201"]))
        rec_missing["body"]["options"]["data"]["flights"][0]["pricing"] = []

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(201, rec_missing["body"]),
        ):
            rev = await provider.review_fare(
                "4-9204425964_0BOMDELSG252~5951139211020472",
                search_price=4159.5,
                search_currency="INR",
            )

        self.assertEqual(rev["status"], "PRICE_MISSING")
        self.assertFalse(rev["priceVerified"])
        self.assertIsNone(rev["reviewedPrice"])
        self.assertIsNone(rev["totalFare"])
        self.assertFalse(rev["originalPriceConfirmed"])
        self.assertEqual(rev["searchPrice"], 4159.5)
        self.assertIn("valid positive totalFare", rev["priceDiscrepancyNotice"])

    async def test_09_fare_comparison_different_currencies_and_mismatched_fare_ids(self):
        """Priority 1: Currency mismatch and fareId mismatch are explicitly flagged and never confirm original price."""
        provider = BankGFlightProvider()

        # Case A: Currency mismatch (search INR vs review USD)
        rec_curr = json.loads(json.dumps(self.fixtures["flightReview_201"]))
        rec_curr["body"]["options"]["data"]["flights"][0]["pricing"][0]["currency"] = "USD"
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(201, rec_curr["body"]),
        ):
            rev_curr = await provider.review_fare(
                "4-9204425964_0BOMDELSG252~5951139211020472",
                search_price=4159.5,
                search_currency="INR",
            )
        self.assertEqual(rev_curr["status"], "CURRENCY_MISMATCH")
        self.assertTrue(rev_curr["currencyMismatch"])
        self.assertTrue(rev_curr["priceChanged"])
        self.assertIsNone(rev_curr["priceDelta"])
        self.assertFalse(rev_curr["originalPriceConfirmed"])

        # Case B: Mismatched fareId (requested SG252 on BOM->DEL, review returned IX1464 on NMI->DEL)
        rec_mismatch = json.loads(json.dumps(self.fixtures["flightReview_201"]))
        rec_mismatch["body"]["options"]["data"]["flights"][0]["pricing"][0][
            "priceId"
        ] = "R4-9204425964_0NMIDELIX1464~5951152016899322"
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=_make_httpx_response(201, rec_mismatch["body"]),
        ):
            rev_mm = await provider.review_fare(
                "4-9204425964_0BOMDELSG252~5951139211020472",
                search_price=4159.5,
                search_currency="INR",
            )
        self.assertEqual(rev_mm["status"], "FARE_ID_MISMATCH")
        self.assertFalse(rev_mm["fareIdMatched"])
        self.assertFalse(rev_mm["originalPriceConfirmed"])
        self.assertIn("does not match requested fareId", rev_mm["priceDiscrepancyNotice"])

    async def test_10_grounding_gate_and_facts_reflect_reviewed_price_and_reject_stale_confirmation(self):
        """Priority 1 & 6: Facts and GroundingGate expose reviewed price and block false confirmation of changed search price."""
        provider = BankGFlightProvider()
        executor = ToolExecutor(flight_provider=provider)
        results = ChatResults()
        session = create_initial_session("sess_p1_grounding")

        rec_search = self.fixtures["airSearch_201"]
        rec_review = self.fixtures["flightReview_201"]
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=[
                _make_httpx_response(rec_search["http_status"], rec_search["body"]),
                _make_httpx_response(rec_review["http_status"], rec_review["body"]),
            ],
        ):
            await executor.execute(
                "search_flights",
                {"origin": "BOM", "destination": "DEL", "departureDate": "2099-11-10", "adults": 1, "cabin": "economy"},
                results,
            )
            await executor.execute("review_fare", {"fareId": results.flights[0].offerId}, results)

        facts = build_facts(session, results)
        self.assertEqual(facts["fareReview.status"], "PRICE_CHANGED")
        self.assertEqual(facts["fareReview.searchPrice"], "₹4159.5")
        self.assertEqual(facts["fareReview.reviewedPrice"], "₹5159.5")
        self.assertEqual(facts["fareReview.priceChanged"], "true")
        self.assertEqual(facts["fareReview.originalPriceConfirmed"], "false")

        gate = GroundingGate.from_results_and_facts(results, facts)
        # 1. False claim that original ₹4,159.50 price is confirmed/locked must be REJECTED
        ok_bad, reason_bad = gate.is_sentence_grounded("Your SpiceJet flight SG 252 fare of ₹4,159.50 is confirmed and locked!")
        self.assertFalse(ok_bad)
        self.assertIn("Cannot claim original search price is confirmed", reason_bad)

        # 2. Honest explanation of the price change from ₹4,159.50 to ₹5,159.50 is APPROVED
        ok_good, _ = gate.is_sentence_grounded(
            "During fare review, the SpiceJet SG 252 fare changed from ₹4,159.50 to the reviewed price of ₹5,159.50."
        )
        self.assertTrue(ok_good)

        # 3. Confirmation of the reviewed price ₹5,159.50 is APPROVED
        ok_rev, _ = gate.is_sentence_grounded("The reviewed fare is locked at ₹5,159.50 for the next 15 minutes.")
        self.assertTrue(ok_rev)

    async def test_11_airport_and_route_identity_bom_vs_nmi_and_route_mismatch(self):
        """
        Priority 2: Verify BOM vs NMI route isolation, Mumbai city search, explicit NMI search,
        alternate airport opt-in, and exclusion of route mismatches.
        """
        provider = BankGFlightProvider()

        mixed_body = {
            "options": {
                "responseCode": "0",
                "description": "SUCCESS",
                "data": {
                    "journeys": [
                        {
                            "flights": [
                                {
                                    "airline": {"name": "Air India Express", "code": "IX"},
                                    "flightNumber": "1464",
                                    "aircraftType": "7M8",
                                    "departure": {
                                        "airportCode": "NMI",
                                        "airportName": "Navi Mumbai International Airport",
                                        "city": "Navi Mumbai",
                                        "terminal": "Terminal 1",
                                        "dateTime": "2099-11-10T10:15:00",
                                    },
                                    "arrival": {
                                        "airportCode": "DEL",
                                        "airportName": "Indira Gandhi Airport",
                                        "city": "Delhi",
                                        "terminal": "Terminal 1",
                                        "dateTime": "2099-11-10T12:35:00",
                                    },
                                    "durationMinutes": 140,
                                }
                            ],
                            "fares": [
                                {
                                    "fareId": "4-9204425964_0NMIDELIX1464~5951139175825525",
                                    "cabinClass": "ECONOMY",
                                    "price": {"baseFare": 2500.0, "tax": 748.5, "totalFare": 3248.5, "currency": "INR"},
                                    "baggage": {"checkIn": "15 Kg", "cabin": "7 Kg"},
                                }
                            ],
                        },
                        {
                            "flights": [
                                {
                                    "airline": {"name": "SpiceJet", "code": "SG"},
                                    "flightNumber": "252",
                                    "aircraftType": "737",
                                    "departure": {
                                        "airportCode": "BOM",
                                        "airportName": "Chhatrapati Shivaji",
                                        "city": "Mumbai",
                                        "terminal": "Terminal 1",
                                        "dateTime": "2099-11-10T07:15:00",
                                    },
                                    "arrival": {
                                        "airportCode": "DEL",
                                        "airportName": "Indira Gandhi Airport",
                                        "city": "Delhi",
                                        "terminal": "Terminal 3",
                                        "dateTime": "2099-11-10T09:35:00",
                                    },
                                    "durationMinutes": 140,
                                }
                            ],
                            "fares": [
                                {
                                    "fareId": "4-9204425964_0BOMDELSG252~5951139211020472",
                                    "cabinClass": "ECONOMY",
                                    "price": {"baseFare": 2950.0, "tax": 1209.5, "totalFare": 4159.5, "currency": "INR"},
                                    "baggage": {"checkIn": "15 Kg", "cabin": "7 Kg"},
                                }
                            ],
                        },
                        {
                            "flights": [
                                {
                                    "airline": {"name": "IndiGo", "code": "6E"},
                                    "flightNumber": "999",
                                    "departure": {
                                        "airportCode": "BLR",
                                        "airportName": "Kempegowda International",
                                        "city": "Bengaluru",
                                        "dateTime": "2099-11-10T06:00:00",
                                    },
                                    "arrival": {
                                        "airportCode": "DEL",
                                        "airportName": "Indira Gandhi Airport",
                                        "city": "Delhi",
                                        "dateTime": "2099-11-10T08:45:00",
                                    },
                                    "durationMinutes": 165,
                                }
                            ],
                            "fares": [
                                {
                                    "fareId": "4-9204425964_0BLRDEL6E999~5951139211020999",
                                    "cabinClass": "ECONOMY",
                                    "price": {"baseFare": 2000.0, "tax": 500.0, "totalFare": 2500.0, "currency": "INR"},
                                    "baggage": {"checkIn": "15 Kg", "cabin": "7 Kg"},
                                }
                            ],
                        },
                    ]
                },
            }
        }

        # 1. Explicit BOM search excludes NMI and BLR offers
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_make_httpx_response(201, mixed_body)):
            bom_offers = await provider.search_flights("BOM", "DEL", "2099-11-10")
        self.assertEqual(len(bom_offers), 1)
        self.assertEqual(bom_offers[0].outbound.origin, "BOM")
        self.assertEqual(bom_offers[0].outbound.destination, "DEL")
        self.assertEqual(bom_offers[0].outbound.flightNumber, "SG 252")
        self.assertEqual(bom_offers[0].observedDetails["originAirportCode"], "BOM")
        self.assertEqual(bom_offers[0].observedDetails["originAirportName"], "Chhatrapati Shivaji")
        self.assertTrue(bom_offers[0].observedDetails["routeMatched"])
        self.assertFalse(bom_offers[0].observedDetails["isAlternateAirport"])

        # 2. Mumbai city search resolves to BOM and excludes NMI
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_make_httpx_response(201, mixed_body)):
            mumbai_offers = await provider.search_flights("Mumbai", "Delhi", "2099-11-10")
        self.assertEqual(len(mumbai_offers), 1)
        self.assertEqual(mumbai_offers[0].outbound.origin, "BOM")
        self.assertNotEqual(mumbai_offers[0].outbound.origin, "NMI")

        # 3. Explicit NMI search returns ONLY NMI offers and excludes BOM and BLR
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_make_httpx_response(201, mixed_body)):
            nmi_offers = await provider.search_flights("NMI", "DEL", "2099-11-10")
        self.assertEqual(len(nmi_offers), 1)
        self.assertEqual(nmi_offers[0].outbound.origin, "NMI")
        self.assertEqual(nmi_offers[0].outbound.flightNumber, "IX 1464")
        self.assertEqual(nmi_offers[0].observedDetails["originAirportCode"], "NMI")
        self.assertEqual(nmi_offers[0].observedDetails["originAirportName"], "Navi Mumbai International Airport")
        self.assertTrue(nmi_offers[0].observedDetails["routeMatched"])

        # 4. Alternate airport opt-in (include_alternate_airports=True) includes co-terminal NMI flagged as alternate,
        # still excludes unrelated BLR, and keeps isCheapest=True on the exact BOM match!
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_make_httpx_response(201, mixed_body)):
            alt_offers = await provider.search_flights("BOM", "DEL", "2099-11-10", include_alternate_airports=True)
        self.assertEqual(len(alt_offers), 2)
        self.assertEqual(alt_offers[0].outbound.origin, "BOM")
        self.assertTrue(alt_offers[0].isCheapest)
        self.assertTrue(alt_offers[0].observedDetails["routeMatched"])
        self.assertFalse(alt_offers[0].observedDetails["isAlternateAirport"])

        self.assertEqual(alt_offers[1].outbound.origin, "NMI")
        self.assertFalse(alt_offers[1].isCheapest)
        self.assertFalse(alt_offers[1].observedDetails["routeMatched"])
        self.assertTrue(alt_offers[1].observedDetails["isAlternateAirport"])

    def test_12_location_resolver_city_vs_airport_selection(self):
        """Priority 2: Verify Mumbai vs Navi Mumbai and BOM vs NMI resolution in LocationResolver."""
        loc_mumbai = LocationResolver.resolve_city("Mumbai")
        self.assertIsNotNone(loc_mumbai)
        self.assertEqual(loc_mumbai.code, "BOM")
        self.assertEqual(loc_mumbai.name, "Mumbai")

        loc_bom = LocationResolver.resolve_city("BOM")
        self.assertIsNotNone(loc_bom)
        self.assertEqual(loc_bom.code, "BOM")

        loc_navi = LocationResolver.resolve_city("Navi Mumbai")
        self.assertIsNotNone(loc_navi)
        self.assertEqual(loc_navi.code, "NMI")
        self.assertEqual(loc_navi.name, "Navi Mumbai")

        loc_nmi = LocationResolver.resolve_city("NMI")
        self.assertIsNotNone(loc_nmi)
        self.assertEqual(loc_nmi.code, "NMI")

        self.assertEqual(_resolve_iata("Mumbai"), "BOM")
        self.assertEqual(_resolve_iata("BOM"), "BOM")
        self.assertEqual(_resolve_iata("Navi Mumbai"), "NMI")
        self.assertEqual(_resolve_iata("NMI"), "NMI")

        o1, d1, _ = LocationResolver.extract_route("Flights from BOM to DEL tomorrow")
        self.assertEqual(o1.code, "BOM")
        self.assertEqual(d1.code, "DEL")

        o2, d2, _ = LocationResolver.extract_route("Flights from NMI to DEL tomorrow")
        self.assertEqual(o2.code, "NMI")
        self.assertEqual(d2.code, "DEL")

        o3, d3, _ = LocationResolver.extract_route("Navi Mumbai se Delhi flight chahiye")
        self.assertEqual(o3.code, "NMI")
        self.assertEqual(d3.code, "DEL")


if __name__ == "__main__":
    unittest.main()

