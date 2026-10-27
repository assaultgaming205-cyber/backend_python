import asyncio
import unittest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

from app.agent.clarification import ClarificationStore, date_options, normalize_clarification
from app.agent.flight_selection import (
    NOT_PROVIDED, FlightLedger, compare_flights, get_ledger, offer_key, select_diverse,
)
from app.agent.search_preflight import (
    check_hotel_dates, expand_range, resolve_flight_date, today_ist,
)
from app.agent.tool_executor import ToolExecutor
from app.ai.stt_service import BoundedAudioBuffer, FailoverSttSession, SttEvent
from app.schemas.chat import ChatResults
from app.schemas.flight import FlightBags, FlightLeg, FlightOffer, FlightSegment, Money
from app.travel.airlines import canonical_airline_name

TODAY = date(2026, 10, 24)


def offer(code, name, price, day="2026-10-25", num="100", fare_id=None, cabin="economy", currency="INR"):
    seg = FlightSegment(carrierCode=code, carrierName=name, flightNumber=f"{code} {num}",
                        origin="BOM", destination="DEL",
                        departureLocal=f"{day}T08:00:00", arrivalLocal=f"{day}T10:00:00")
    return FlightOffer(
        offerId=fare_id or f"bankg_{code}{num}{price}", provider="BankG",
        price=Money(amount=price, currency=currency, verified=True),
        outbound=FlightLeg(airline=name, origin="BOM", destination="DEL",
                           departureLocal=seg.departureLocal, arrivalLocal=seg.arrivalLocal,
                           durationMinutes=120, stops=0, segments=[seg]),
        cabinClass=cabin, bags=FlightBags(carryOn=1, checked=None))


class TestAirlineNames(unittest.TestCase):
    def test_ai_express_alias_becomes_full_brand(self):
        r = canonical_airline_name("AI Express", "IX")
        self.assertEqual((r.display, r.ambiguous), ("Air India Express", False))

    def test_already_canonical_and_air_india_distinct(self):
        self.assertEqual(canonical_airline_name("Air India Express", "IX").display, "Air India Express")
        self.assertEqual(canonical_airline_name("Air India", "AI").display, "Air India")

    def test_codeshare_prefers_operating_carrier(self):
        r = canonical_airline_name("Air India", "AI", operating_name="Air India Express", operating_code="IX")
        self.assertEqual(r.display, "Air India Express")

    def test_operating_agrees(self):
        self.assertEqual(canonical_airline_name("Air India", "AI", "Air India", "AI").display, "Air India")

    def test_conflict_and_missing_are_conservative(self):
        c = canonical_airline_name("Air India", "IX")
        self.assertEqual(c.display, "Air India")  # supplier name kept, never silently rebranded
        self.assertTrue(c.ambiguous)
        m = canonical_airline_name(None, "AI")
        self.assertEqual(m.display, "AI")  # a bare code is not turned into a brand
        self.assertTrue(m.ambiguous)

    def test_unrelated_carrier_unchanged(self):
        self.assertEqual(canonical_airline_name("IndiGo", "6E").display, "IndiGo")

    def test_flight_number_independent_of_display(self):
        o = offer("IX", "AI Express", 4000)
        self.assertEqual(o.outbound.segments[0].flightNumber, "IX 100")


class TestDatePreflight(unittest.TestCase):
    def test_today_uses_india_local_date_near_midnight(self):
        utc_late = datetime(2026, 10, 24, 20, 0, tzinfo=ZoneInfo("UTC"))  # 01:30 IST on 25th
        self.assertEqual(today_ist(lambda: utc_late), date(2026, 10, 25))
        self.assertEqual(today_ist(lambda: datetime(2026, 10, 24, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))), TODAY)

    def test_default_is_tomorrow_and_disclosed(self):
        d = resolve_flight_date(None, None, TODAY)
        self.assertEqual((d.status, d.date, d.inferred, d.source), ("ok", "2026-10-25", True, "default"))
        self.assertIn("25 October", d.message)

    def test_explicit_context_and_not_inferred(self):
        self.assertFalse(resolve_flight_date("2026-10-27", None, TODAY).inferred)
        c = resolve_flight_date(None, "2026-10-29", TODAY)
        self.assertEqual((c.date, c.source, c.inferred), ("2026-10-29", "context", False))

    def test_past_invalid_ambiguous_clarify(self):
        for args in [("2026-10-01", None), ("2026-02-31", None)]:
            self.assertEqual(resolve_flight_date(*args, TODAY).status, "clarify")
        self.assertEqual(resolve_flight_date(None, None, TODAY, ambiguous=True).status, "clarify")

    def test_ranges_are_exact_and_bounded(self):
        self.assertEqual(expand_range("2026-10-25", 3, TODAY), ["2026-10-25", "2026-10-26", "2026-10-27"])
        self.assertEqual(len(expand_range("2026-10-25", 4, TODAY)), 4)
        self.assertIsNone(expand_range("2026-10-25", 30, TODAY))
        self.assertIsNone(expand_range("2026-10-01", 3, TODAY))

    def test_hotel_needs_both_dates(self):
        self.assertEqual(check_hotel_dates("2026-10-25", None, TODAY), "missing_dates")
        self.assertEqual(check_hotel_dates("2026-10-26", "2026-10-25", TODAY), "checkout_not_after_checkin")
        self.assertIsNone(check_hotel_dates("2026-10-25", "2026-10-27", TODAY))


class TestSupplierGating(unittest.IsolatedAsyncioTestCase):
    def make(self):
        f, t, h = MagicMock(), MagicMock(), MagicMock()
        f.search_flights = AsyncMock(return_value=[])
        t.search_trains = AsyncMock(return_value=[])
        h.search_hotels = AsyncMock(return_value=[])
        return ToolExecutor(f, t, h, today_fn=lambda: TODAY), f, t, h

    async def test_valid_date_calls_supplier_once_with_exact_date(self):
        ex, f, _, _ = self.make()
        s = await ex.execute("search_flights", {"origin": "BOM", "destination": "DEL", "departureDate": "2026-10-27"}, ChatResults())
        self.assertTrue(s.success)
        f.search_flights.assert_awaited_once()
        self.assertEqual(f.search_flights.await_args.kwargs["departure_date"], "2026-10-27")

    async def test_missing_invalid_past_dates_never_reach_supplier(self):
        ex, f, t, h = self.make()
        for d in ["", None, "garbage", "2026-10-01"]:
            s = await ex.execute("search_flights", {"origin": "BOM", "destination": "DEL", "departureDate": d}, ChatResults())
            self.assertFalse(s.success)
            self.assertTrue(s.error.startswith("blocked:"))
            await ex.execute("search_trains", {"origin": "A", "destination": "B", "date": d}, ChatResults())
        f.search_flights.assert_not_awaited()
        t.search_trains.assert_not_awaited()

    async def test_hotel_blocked_until_both_dates(self):
        ex, _, _, h = self.make()
        await ex.execute("search_hotels", {"destination": "Goa", "checkIn": "2026-10-25"}, ChatResults())
        h.search_hotels.assert_not_awaited()
        await ex.execute("search_hotels", {"destination": "Goa", "checkIn": "2026-10-25", "checkOut": "2026-10-27"}, ChatResults())
        h.search_hotels.assert_awaited_once()

    async def test_fanout_is_bounded_and_every_date_searched(self):
        ex, f, _, _ = self.make()
        running = peak = 0

        async def slow(**kw):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return []
        f.search_flights = AsyncMock(side_effect=slow)
        dates = [f"2026-10-{25 + i}" for i in range(6)]
        await asyncio.gather(*[ex.execute("search_flights", {"origin": "BOM", "destination": "DEL", "departureDate": d}, ChatResults()) for d in dates])
        self.assertEqual(f.search_flights.await_count, 6)
        self.assertLessEqual(peak, 3)

    async def test_one_date_failing_keeps_others(self):
        ex, f, _, _ = self.make()
        good = offer("6E", "IndiGo", 3000, day="2026-10-26")

        async def by_date(**kw):
            if kw["departure_date"] == "2026-10-25":
                raise RuntimeError("supplier down")
            return [good]
        f.search_flights = AsyncMock(side_effect=by_date)
        res = ChatResults()
        a = await ex.execute("search_flights", {"origin": "B", "destination": "D", "departureDate": "2026-10-25"}, res)
        b = await ex.execute("search_flights", {"origin": "B", "destination": "D", "departureDate": "2026-10-26"}, res)
        self.assertFalse(a.success)
        self.assertTrue(b.success)
        self.assertEqual(len(res.flights), 1)


class TestDiversityLedgerCompare(unittest.TestCase):
    def test_distinct_brands_per_date_independent(self):
        offers = ([offer("6E", "IndiGo", 3000 + i, num=str(i)) for i in range(5)]
                  + [offer("AI", "Air India", 5000), offer("IX", "AI Express", 4000), offer("QP", "Akasa Air", 4500),
                     offer("SG", "SpiceJet", 6000)]
                  + [offer("6E", "IndiGo", 3100, day="2026-10-26"), offer("AI", "Air India", 5200, day="2026-10-26")])
        sel, short = select_diverse(offers)
        d1 = [o for o in sel if o.outbound.departureLocal.startswith("2026-10-25")]
        self.assertEqual(len(d1), 4)
        self.assertEqual(len({o.outbound.airline for o in d1}), 4)
        self.assertNotIn("2026-10-25", short)
        self.assertEqual(short, {"2026-10-26": 2})  # honest shortfall, no fake offers
        self.assertEqual(len([o for o in sel if o.outbound.departureLocal.startswith("2026-10-26")]), 2)

    def test_dedupe_equivalent_offers(self):
        a, b = offer("6E", "IndiGo", 3000, fare_id="x1"), offer("6E", "IndiGo", 3000, fare_id="x2")
        self.assertEqual(offer_key(a), offer_key(b))
        self.assertEqual(len(select_diverse([a, b])[0]), 1)

    def test_show_more_excludes_shown_and_rejected_and_is_exhaustible(self):
        led = FlightLedger()
        offers = [offer("6E", "IndiGo", 3000 + i, num=str(i)) for i in range(7)]
        led.remember(offers)
        first = led.next_batch()
        self.assertEqual(len(first), 3)
        led.reject(offers[3])
        second = led.next_batch()
        keys = {offer_key(o) for o in first}
        self.assertTrue(keys.isdisjoint({offer_key(o) for o in second}))
        self.assertNotIn(offer_key(offers[3]), {offer_key(o) for o in second})
        self.assertEqual(len(second), 3)
        self.assertEqual(led.next_batch(), [])  # exhausted: nothing manufactured
        recovered = led.next_batch(include_rejected=[offer_key(offers[3])])
        self.assertEqual([offer_key(o) for o in recovered], [offer_key(offers[3])])

    def test_explicit_rejected_recovery(self):
        led = FlightLedger()
        o = offer("6E", "IndiGo", 3000)
        led.remember([o])
        led.reject(o)
        self.assertEqual(led.next_batch(), [])
        self.assertEqual(len(led.next_batch(include_rejected=[offer_key(o)])), 1)

    def test_multi_date_pagination_keeps_dates(self):
        led = FlightLedger()
        led.remember([offer("6E", "IndiGo", 3000 + i, num=str(i), day=d) for i in range(4) for d in ("2026-10-25", "2026-10-26")])
        out = led.next_batch(per_date=3)
        self.assertEqual({o.outbound.departureLocal[:10] for o in out}, {"2026-10-25", "2026-10-26"})
        self.assertEqual(len(out), 6)

    def test_conversations_isolated(self):
        o = offer("6E", "IndiGo", 3000)
        get_ledger("conv-A").reject(o)
        self.assertNotIn(offer_key(o), get_ledger("conv-B").rejected)

    def test_comparison_uses_only_supplier_fields(self):
        c = compare_flights([offer("6E", "IndiGo", 3000), offer("IX", "AI Express", 4000)])
        row = c["rows"][1]
        self.assertEqual(row["airline"], "Air India Express")
        self.assertEqual(row["refundPolicy"], NOT_PROVIDED)
        self.assertEqual(row["checkedBags"], NOT_PROVIDED)  # checked=None -> not fabricated
        self.assertEqual(row["cabinBags"], 1)
        self.assertTrue(c["priceComparable"])
        self.assertEqual(c["cheapestOfferId"], c["rows"][0]["offerId"])

    def test_different_currency_or_cabin_not_ranked(self):
        c = compare_flights([offer("6E", "IndiGo", 3000), offer("AI", "Air India", 90, currency="USD")])
        self.assertFalse(c["priceComparable"])
        self.assertIsNone(c["cheapestOfferId"])
        self.assertIsNotNone(c["note"])
        c2 = compare_flights([offer("6E", "IndiGo", 3000), offer("AI", "Air India", 9000, cabin="business")])
        self.assertIsNone(c2["cheapestOfferId"])


class TestClarification(unittest.TestCase):
    def test_options_capped_at_three_with_separate_labels(self):
        raw = {"question": "Kab?", "field": "date",
               "options": [{"label": f"L{i}", "value": f"v{i}"} for i in range(6)] + ["v1"]}
        card = normalize_clarification(raw, "t1")
        self.assertEqual(len(card["options"]), 3)
        self.assertEqual(card["options"][0], {"label": "L0", "value": "v0"})
        self.assertTrue(card["allowCustom"])
        self.assertEqual(card["type"], "clarification")

    def test_invalid_card_rejected_and_junk_options_ignored(self):
        self.assertIsNone(normalize_clarification({"question": "", "field": "date"}, "t"))
        card = normalize_clarification({"question": "q", "field": "cabin", "options": [None, 5, {"label": "", "value": ""}]}, "t")
        self.assertEqual(card["options"], [])

    def test_date_options_are_real_future_dates(self):
        opts = date_options(TODAY)
        self.assertEqual([o["value"] for o in opts], ["2026-10-25", "2026-10-26", "2026-10-31"])
        self.assertLessEqual(len(opts), 3)

    def test_answer_flow_stale_duplicate_invalid(self):
        s = ClarificationStore()
        card = normalize_clarification({"question": "Kab?", "field": "travelDate"}, "turn-1")
        s.open("c1", card)
        qid = card["questionId"]
        self.assertEqual(s.answer("c1", qid, "old-turn", "2026-10-27", TODAY)["status"], "stale")
        self.assertEqual(s.answer("c1", qid, "turn-1", "2026-10-01", TODAY)["status"], "invalid")
        self.assertEqual(s.answer("c2", qid, "turn-1", "2026-10-27", TODAY)["status"], "stale")  # other conversation
        ok = s.answer("c1", qid, "turn-1", "2026-10-27", TODAY)
        self.assertEqual((ok["status"], ok["value"]), ("ok", "2026-10-27"))
        self.assertEqual(s.answer("c1", qid, "turn-1", "2026-10-27", TODAY)["status"], "stale")  # duplicate click

    def test_newer_card_supersedes_and_cancel_clears(self):
        s = ClarificationStore()
        a = normalize_clarification({"question": "q", "field": "date"}, "t1")
        s.open("c", a)
        b = normalize_clarification({"question": "q2", "field": "date"}, "t2")
        s.open("c", b)
        self.assertEqual(s.answer("c", a["questionId"], "t1", "2026-10-27", TODAY)["status"], "stale")
        s.cancel("c")
        self.assertEqual(s.answer("c", b["questionId"], "t2", "2026-10-27", TODAY)["status"], "stale")


class TestAudioBufferPrefix(unittest.IsolatedAsyncioTestCase):
    def test_keeps_beginning_and_flags_overflow(self):
        b = BoundedAudioBuffer(max_bytes=10)
        self.assertTrue(b.append(b"AAAA"))
        self.assertTrue(b.append(b"BBBB"))
        self.assertFalse(b.append(b"CCCC"))
        self.assertTrue(b.overflowed)
        self.assertEqual(b.get_all(), [b"AAAA", b"BBBB"])  # beginning kept, tail not substituted
        b.clear()
        self.assertFalse(b.overflowed)
        self.assertEqual(b.get_all(), [])

    def test_default_capacity_is_60s_of_16k_mono_pcm16(self):
        self.assertEqual(BoundedAudioBuffer().max_bytes, 60 * 16000 * 2)

    async def test_failover_replays_full_utterance_prefix_first(self):
        sess = FailoverSttSession(sarvam_enabled=False)
        sent = []
        sess.parakeet = MagicMock()
        sess.parakeet.send_audio = AsyncMock(side_effect=lambda f: sent.append(f) or True)
        sess.parakeet.start = AsyncMock(return_value=True)
        sess.parakeet.stop = AsyncMock()
        sess.sarvam = MagicMock()
        sess.sarvam.cancel = AsyncMock()
        for i in range(200):  # 200 x 100ms frames = 20s, well beyond the old 5s rolling window
            sess._audio_buffer.append(bytes([i % 256]) * 3200)
        sess._stopping = True
        await sess._failover_to_parakeet()
        self.assertEqual(len(sent), 200)
        self.assertEqual(sent[0], bytes([0]) * 3200)  # first words of the utterance survive
        sess.parakeet.stop.assert_awaited_once()  # stop arrived mid-failover; replacement is finalized

    async def test_overflow_blocks_tail_only_replay_with_friendly_error(self):
        sess = FailoverSttSession(sarvam_enabled=False)
        sess._audio_buffer = BoundedAudioBuffer(max_bytes=3200)
        sess.parakeet = MagicMock()
        sess.parakeet.send_audio = AsyncMock()
        sess.parakeet.start = AsyncMock(return_value=True)
        sess.sarvam = MagicMock()
        sess.sarvam.cancel = AsyncMock()
        sess._audio_buffer.append(b"\x00" * 3200)
        sess._audio_buffer.append(b"\x01" * 3200)
        await sess._failover_to_parakeet()
        sess.parakeet.send_audio.assert_not_awaited()
        events = []
        while not sess._event_queue.empty():
            events.append(sess._event_queue.get_nowait())
        err = [e for e in events if isinstance(e, SttEvent) and e.kind == "error"]
        self.assertEqual(err[0].error_code, "utterance_too_long")


if __name__ == "__main__":
    unittest.main()
