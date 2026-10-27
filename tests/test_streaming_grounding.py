import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator, List

from fastapi import Request
from starlette.datastructures import Headers

from app.agent.grounding.sentence_buffer import SentenceBuffer
from app.agent.grounding.grounding_gate import GroundingGate
from app.agent.agent_service import AgentService, AgentResult
from app.schemas.chat import ChatResults, MessageRole
from app.schemas.flight import FlightOffer, FlightLeg, FlightSegment, Money
from app.schemas.train import TrainOption, TrainClassFare, Station
from app.schemas.hotel import HotelOffer
from app.api.chat import chat_stream_endpoint, chat_endpoint, ChatRequestDto, ChatMessageDto, agent_service
from app.config import settings


class TestSentenceBuffer(unittest.TestCase):
    def test_multiple_sentences_across_chunks(self):
        buf = SentenceBuffer()
        chunks = ["Hello ", "there! How ", "are you ", "doing today? I hope ", "all is well."]
        emitted = []
        for c in chunks:
            emitted.extend(buf.feed(c))
        emitted.extend(buf.finalize())
        self.assertEqual(
            emitted,
            ["Hello there!", "How are you doing today?", "I hope all is well."]
        )

    def test_periods_in_abbreviations(self):
        buf = SentenceBuffer()
        text = "The flight duration is approx. 2 hrs. and 15 min. Please arrive 2 hrs. before departure."
        emitted = buf.feed(text)
        emitted.extend(buf.finalize())
        self.assertEqual(len(emitted), 2)
        self.assertIn("approx. 2 hrs. and 15 min.", emitted[0])
        self.assertIn("Please arrive 2 hrs. before departure.", emitted[1])

    def test_times_not_split(self):
        buf = SentenceBuffer()
        text = "Flight AI-805 departs at 07:25 and arrives at 14:30. Boarding starts at 06:45."
        emitted = buf.feed(text)
        emitted.extend(buf.finalize())
        self.assertEqual(len(emitted), 2)
        self.assertIn("07:25", emitted[0])
        self.assertIn("14:30", emitted[0])

    def test_currency_values_not_split(self):
        buf = SentenceBuffer()
        text = "Flight fare is ₹4,500.00 per person. Train ticket is Rs. 1,200.50 only."
        emitted = buf.feed(text)
        emitted.extend(buf.finalize())
        self.assertEqual(len(emitted), 2)
        self.assertIn("₹4,500.00", emitted[0])
        self.assertIn("Rs. 1,200.50", emitted[1])

    def test_decimal_values_not_split(self):
        buf = SentenceBuffer()
        text = "The hotel rating is 4.8 out of 5 stars. It is located 3.5 km from the airport."
        emitted = buf.feed(text)
        emitted.extend(buf.finalize())
        self.assertEqual(len(emitted), 2)
        self.assertIn("4.8", emitted[0])
        self.assertIn("3.5", emitted[1])

    def test_hinglish_and_hindi_danda(self):
        buf = SentenceBuffer()
        text = "Yeh flight sabse sasti hai। Train ka option bhi badhiya hai॥"
        emitted = buf.feed(text)
        emitted.extend(buf.finalize())
        self.assertEqual(len(emitted), 2)
        self.assertIn("Yeh flight sabse sasti hai।", emitted[0])
        self.assertIn("Train ka option bhi badhiya hai॥", emitted[1])

    def test_short_final_sentence_on_finalize(self):
        buf = SentenceBuffer()
        buf.feed("Pehli flight book karein. Chalo")
        emitted = buf.finalize()
        self.assertIn("Chalo", emitted)

    def test_end_of_stream_partial_sentence(self):
        buf = SentenceBuffer()
        emitted = buf.feed("Aapka din shubh ho")
        self.assertEqual(emitted, [])
        final = buf.finalize()
        self.assertEqual(final, ["Aapka din shubh ho"])

    def test_json_envelope_stripping(self):
        buf = SentenceBuffer()
        stream = ['{"reply": "', 'IndiGo flight is available. ', 'Fares are starting from low."}']
        emitted = []
        for c in stream:
            emitted.extend(buf.feed(c))
        emitted.extend(buf.finalize())
        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[0], "IndiGo flight is available.")
        self.assertEqual(emitted[1], "Fares are starting from low.")
        # Ensure no JSON quotes or braces leaked
        for s in emitted:
            self.assertNotIn('{"reply"', s)
            self.assertNotIn('"}', s)


class TestGroundingGate(unittest.TestCase):
    def setUp(self):
        flight = FlightOffer(
            offerId="flt_1",
            provider="amadeus",
            price=Money(amount=4500.0, currency="INR"),
            outbound=FlightLeg(
                airline="Air India",
                origin="DEL",
                destination="BOM",
                departureLocal="2026-11-05T07:25:00",
                arrivalLocal="2026-11-05T09:45:00",
                segments=[
                    FlightSegment(
                        flightNumber="AI-805",
                        origin="DEL",
                        destination="BOM",
                        departureLocal="2026-11-05T07:25:00",
                        arrivalLocal="2026-11-05T09:45:00",
                    )
                ]
            )
        )
        train = TrainOption(
            trainNumber="12952",
            trainName="Mumbai Rajdhani",
            origin=Station(code="NDLS", name="New Delhi"),
            destination=Station(code="MMCT", name="Mumbai Central"),
            departureTime="16:55",
            arrivalTime="08:35",
            departureDate="2026-11-05",
            classFares=[
                TrainClassFare(classCode="3A", className="3rd AC", fare=1850.0)
            ]
        )
        self.results = ChatResults(flights=[flight], trains=[train])
        self.gate = GroundingGate.from_results_and_facts(self.results)

    def test_invented_fare_rejected(self):
        ok, reason = self.gate.is_sentence_grounded("Ticket sirf ₹2,999 mein mil sakti hai.")
        self.assertFalse(ok)
        self.assertIn("Unverified fare", reason)

    def test_invented_flight_number_rejected(self):
        ok, reason = self.gate.is_sentence_grounded("Aap IndiGo 6E-2134 se jaa sakte hain.")
        self.assertFalse(ok)
        self.assertIn("Unverified flight number", reason)

    def test_invented_train_number_rejected(self):
        ok, reason = self.gate.is_sentence_grounded("Train 19020 best schedule offer karti hai.")
        self.assertFalse(ok)
        self.assertIn("Unverified train number", reason)

    def test_invented_departure_time_rejected(self):
        ok, reason = self.gate.is_sentence_grounded("Flight subah 11:30 par rawana hogi.")
        self.assertFalse(ok)
        self.assertIn("Unverified schedule time", reason)

    def test_valid_supplier_details_accepted(self):
        s1 = "Air India flight AI-805 subah 07:25 baje niklegi aur fare ₹4,500 hai."
        ok, reason = self.gate.is_sentence_grounded(s1)
        self.assertTrue(ok, f"Failed with reason: {reason}")

        s2 = "Mumbai Rajdhani (12952) sham 16:55 baje chalegi aur 3A fare ₹1,850 hai."
        ok, reason = self.gate.is_sentence_grounded(s2)
        self.assertTrue(ok, f"Failed with reason: {reason}")

    def test_natural_conversational_sentences_accepted(self):
        ok, _ = self.gate.is_sentence_grounded("Maine aapke route ke liye live travel options dhoondh liye hain!")
        self.assertTrue(ok)

        ok, _ = self.gate.is_sentence_grounded("Aap kaunsi timing prefer karenge dost?")
        self.assertTrue(ok)


class TestSupplierOrchestration(unittest.IsolatedAsyncioTestCase):
    async def test_fast_supplier_emits_card_before_slow_supplier(self):
        agent = AgentService()
        cards_received = []

        async def fast_flights(*args, **kwargs):
            await asyncio.sleep(0.02)
            f = FlightOffer(
                offerId="f1",
                provider="amadeus",
                price=Money(amount=3500.0),
                outbound=FlightLeg(
                    airline="IndiGo",
                    origin="DEL",
                    destination="BOM",
                    departureLocal="2026-11-05T06:00:00",
                    arrivalLocal="2026-11-05T08:15:00",
                    segments=[FlightSegment(flightNumber="6E-101", origin="DEL", destination="BOM", departureLocal="2026-11-05T06:00:00", arrivalLocal="2026-11-05T08:15:00")]
                )
            )
            return [f]

        async def slow_trains(*args, **kwargs):
            await asyncio.sleep(0.1)
            t = TrainOption(
                trainNumber="12952",
                trainName="Rajdhani",
                origin=Station(code="NDLS", name="New Delhi"),
                destination=Station(code="MMCT", name="Mumbai Central"),
                departureTime="16:55",
                arrivalTime="08:35",
                classFares=[TrainClassFare(classCode="3A", className="3rd AC", fare=1800.0)]
            )
            return [t]

        with patch.object(agent.executor.flight_provider, "search_flights", side_effect=fast_flights), \
             patch.object(agent.executor.rail_provider, "search_trains", side_effect=slow_trains), \
             patch.object(agent.executor.hotel_provider, "search_hotels", new=AsyncMock(return_value=[])), \
             patch.object(agent.ai, "chat", new=AsyncMock(return_value=MagicMock(message=MagicMock(content="Found options!"), model="test-model"))):

            history = [MessageRole(role="user", content="Delhi to Mumbai on 5th Nov by flight or train")]
            result = await agent.run(
                history=history,
                on_card=lambda card_type, data: cards_received.append((card_type, data)),
            )

            self.assertTrue(len(cards_received) >= 2)
            # Fast flight card arrives first
            self.assertEqual(cards_received[0][0], "flight")
            self.assertEqual(cards_received[1][0], "train")
            self.assertEqual(len(result.results.flights), 1)
            self.assertEqual(len(result.results.trains), 1)

    async def test_single_supplier_failure_isolated(self):
        agent = AgentService()
        cards_received = []

        async def failing_flights(*args, **kwargs):
            raise RuntimeError("Amadeus API Down")

        async def working_trains(*args, **kwargs):
            t = TrainOption(
                trainNumber="12952",
                trainName="Rajdhani",
                origin=Station(code="NDLS", name="New Delhi"),
                destination=Station(code="MMCT", name="Mumbai Central"),
                departureTime="16:55",
                arrivalTime="08:35",
                classFares=[TrainClassFare(classCode="3A", className="3rd AC", fare=1800.0)]
            )
            return [t]

        with patch.object(agent.executor.flight_provider, "search_flights", side_effect=failing_flights), \
             patch.object(agent.executor.rail_provider, "search_trains", side_effect=working_trains), \
             patch.object(agent.executor.hotel_provider, "search_hotels", new=AsyncMock(return_value=[])), \
             patch.object(agent.ai, "chat", new=AsyncMock(return_value=MagicMock(message=MagicMock(content="Found trains!"), model="test-model"))):

            history = [MessageRole(role="user", content="Delhi to Mumbai on 5th Nov by flight or train")]
            result = await agent.run(
                history=history,
                on_card=lambda card_type, data: cards_received.append((card_type, data)),
            )

            # Train search still succeeded even though flight search failed
            self.assertEqual(len(cards_received), 1)
            self.assertEqual(cards_received[0][0], "train")
            self.assertEqual(len(result.results.trains), 1)
            self.assertEqual(len(result.results.flights), 0)


class TestApiAndEvents(unittest.IsolatedAsyncioTestCase):
    async def test_chat_stream_full_turn_event_sequence(self):
        dto = ChatRequestDto(
            messages=[ChatMessageDto(role="user", content="Delhi to Mumbai on 5th Nov")],
            sessionId="test_sess_123"
        )

        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=False)

        async def mock_chat_stream(*args, **kwargs):
            yield '{"reply": "Maine live options dekh li hain. '
            yield 'Air India ki flight AI-805 available hai."}'

        flight = FlightOffer(
            offerId="f1",
            provider="amadeus",
            price=Money(amount=4500.0),
            outbound=FlightLeg(
                airline="Air India",
                origin="DEL",
                destination="BOM",
                departureLocal="2026-11-05T07:25:00",
                arrivalLocal="2026-11-05T09:45:00",
                segments=[FlightSegment(flightNumber="AI-805", origin="DEL", destination="BOM", departureLocal="2026-11-05T07:25:00", arrivalLocal="2026-11-05T09:45:00")]
            )
        )

        with patch.object(settings, "LLM_TOKEN_STREAMING_ENABLED", True), \
             patch("app.api.chat.agent_service.ai.chat_stream", side_effect=mock_chat_stream), \
             patch.object(agent_service.executor.flight_provider, "search_flights", new=AsyncMock(return_value=[flight])), \
             patch.object(agent_service.executor.rail_provider, "search_trains", new=AsyncMock(return_value=[])), \
             patch.object(agent_service.executor.hotel_provider, "search_hotels", new=AsyncMock(return_value=[])):

            response = await chat_stream_endpoint(dto, mock_request)
            events = []
            async for chunk in response.body_iterator:
                line = chunk.strip() if isinstance(chunk, str) else chunk.decode("utf-8").strip()
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    events.append(ev)

            # Verify sequence
            types = [e["type"] for e in events]
            self.assertIn("turn_started", types)
            self.assertIn("status", types)
            self.assertIn("card", types)
            self.assertIn("sentence", types)
            self.assertIn("done", types)

            # Verify turnId is consistent across events
            turn_id = events[0]["turnId"]
            self.assertTrue(bool(turn_id))
            for e in events:
                if "turnId" in e:
                    self.assertEqual(e["turnId"], turn_id)

    async def test_cancellation_on_client_disconnect(self):
        dto = ChatRequestDto(
            messages=[ChatMessageDto(role="user", content="Find flights from Delhi to Mumbai")],
            sessionId="test_sess_cancel"
        )

        disconnect_calls = 0

        async def mock_is_disconnected():
            nonlocal disconnect_calls
            disconnect_calls += 1
            # Disconnect on 2nd check
            return disconnect_calls >= 2

        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = mock_is_disconnected

        async def slow_chat_stream(*args, **kwargs):
            for i in range(10):
                await asyncio.sleep(0.1)
                yield f"Sentence number {i}. "

        with patch.object(settings, "LLM_TOKEN_STREAMING_ENABLED", True), \
             patch("app.api.chat.agent_service.ai.chat_stream", side_effect=slow_chat_stream), \
             patch.object(agent_service.executor.flight_provider, "search_flights", new=AsyncMock(return_value=[])), \
             patch.object(agent_service.executor.rail_provider, "search_trains", new=AsyncMock(return_value=[])), \
             patch.object(agent_service.executor.hotel_provider, "search_hotels", new=AsyncMock(return_value=[])):

            response = await chat_stream_endpoint(dto, mock_request)
            events = []
            async for chunk in response.body_iterator:
                line = chunk.strip() if isinstance(chunk, str) else chunk.decode("utf-8").strip()
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    events.append(ev)

            types = [e["type"] for e in events]
            self.assertIn("turn_cancelled", types)
            self.assertNotIn("done", types)

    async def test_legacy_mode_preservation(self):
        dto = ChatRequestDto(
            messages=[ChatMessageDto(role="user", content="Hello")],
            sessionId="test_sess_legacy"
        )
        mock_request = MagicMock(spec=Request)
        mock_request.is_disconnected = AsyncMock(return_value=False)

        with patch.object(settings, "LLM_TOKEN_STREAMING_ENABLED", False), \
             patch.object(agent_service.ai, "chat", new=AsyncMock(return_value=MagicMock(message=MagicMock(content="Hello! How can I help?"), model="test-model"))), \
             patch.object(agent_service.executor.flight_provider, "search_flights", new=AsyncMock(return_value=[])), \
             patch.object(agent_service.executor.rail_provider, "search_trains", new=AsyncMock(return_value=[])), \
             patch.object(agent_service.executor.hotel_provider, "search_hotels", new=AsyncMock(return_value=[])):

            response = await chat_stream_endpoint(dto, mock_request)
            events = []
            async for chunk in response.body_iterator:
                line = chunk.strip() if isinstance(chunk, str) else chunk.decode("utf-8").strip()
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    events.append(ev)

            types = [e["type"] for e in events]
            self.assertIn("chunk", types)
            self.assertIn("narration", types)
            self.assertIn("results", types)
            self.assertIn("done", types)

    async def test_chat_endpoint_non_streaming_contract(self):
        dto = ChatRequestDto(
            messages=[ChatMessageDto(role="user", content="Hello")],
            sessionId="test_sess_sync"
        )

        with patch.object(agent_service.ai, "chat", new=AsyncMock(return_value=MagicMock(message=MagicMock(content="Namaste!"), model="test-model"))):
            resp = await chat_endpoint(dto)
            self.assertEqual(resp.message.role, "assistant")
            self.assertIn("Namaste", resp.message.content)
            self.assertIsNotNone(resp.metadata)
            self.assertIsNotNone(resp.results)


if __name__ == "__main__":
    unittest.main()
