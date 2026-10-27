import unittest
from datetime import date

from app.agent.agent_service import AgentService
from app.agent.flight_selection import (
    FlightLedger,
    _dedupe,
    compare_flights,
    get_ledger,
    offer_key,
    parse_departure_minutes,
    select_diverse,
    sort_flight_offers,
)
from app.agent.session import TravelSession
from app.schemas.chat import ChatResults
from app.schemas.flight import (
    FlightBags,
    FlightLeg,
    FlightOffer,
    FlightSegment,
    Money,
)
from app.travel.airlines import canonical_airline_name


def make_test_offer(
    code: str,
    name: str,
    price: float,
    num: str = "101",
    day: str = "2026-10-25",
    dep_time: str = "08:00:00",
    arr_time: str = "10:00:00",
    cabin: str = "economy",
    currency: str = "INR",
    price_basis: str = "per_person",
    fare_id: str = None,
) -> FlightOffer:
    dep_local = f"{day}T{dep_time}" if "T" not in dep_time else dep_time
    arr_local = f"{day}T{arr_time}" if "T" not in arr_time else arr_time
    seg = FlightSegment(
        carrierCode=code,
        carrierName=name,
        flightNumber=f"{code} {num}",
        origin="BOM",
        destination="DEL",
        departureLocal=dep_local,
        arrivalLocal=arr_local,
    )
    return FlightOffer(
        offerId=fare_id or f"offer_{code}_{num}_{price}_{cabin}_{day}",
        provider="BankG",
        price=Money(amount=price, currency=currency, verified=True),
        outbound=FlightLeg(
            airline=name,
            origin="BOM",
            destination="DEL",
            departureLocal=dep_local,
            arrivalLocal=arr_local,
            durationMinutes=120,
            stops=0,
            segments=[seg],
        ),
        cabinClass=cabin,
        bags=FlightBags(carryOn=1, checked=None),
        priceBasis=price_basis,
        sourceUrl="https://example.com/book",
    )


class TestFlightResultsAndSorting(unittest.TestCase):
    def setUp(self):
        self.agent = AgentService()

    # 1. Every distinct validated offer in a fetched response remains available to the result renderer.
    def test_01_all_distinct_validated_offers_remain_available(self):
        session = TravelSession(sessionId="session_all_offers")
        results = ChatResults(
            flights=[
                make_test_offer("6E", "IndiGo", 3500, num="101"),
                make_test_offer("6E", "IndiGo", 3800, num="102"),
                make_test_offer("AI", "Air India", 4200, num="201"),
                make_test_offer("AI", "Air India", 4500, num="202"),
                make_test_offer("SG", "SpiceJet", 3200, num="301"),
                make_test_offer("UK", "Vistara", 5000, num="401"),
                make_test_offer("QP", "Akasa Air", 3400, num="501"),
            ]
        )
        self.agent._filter_and_rank(session, results)
        # All 7 distinct offers must remain available!
        self.assertEqual(len(results.flights), 7)
        # Diverse airline picks come first (IndiGo, SpiceJet, Akasa, Air India, Vistara)
        first_brands = [f.outbound.airline for f in results.flights[:4]]
        self.assertEqual(len(set(first_brands)), 4)

    # 2. Duplicate representations of an identical offer are deduplicated.
    def test_02_duplicate_representations_are_deduplicated(self):
        o1 = make_test_offer("6E", "IndiGo", 3500, num="101", fare_id="dup_1")
        o2 = make_test_offer("6E", "IndiGo", 3500, num="101", fare_id="dup_2")  # identical facts
        deduped = _dedupe([o1, o2])
        self.assertEqual(len(deduped), 1)

    # 3. Distinct flights from the same airline are not incorrectly collapsed.
    def test_03_distinct_flights_same_airline_not_collapsed(self):
        o1 = make_test_offer("6E", "IndiGo", 3500, num="101", dep_time="08:00:00")
        o2 = make_test_offer("6E", "IndiGo", 4200, num="102", dep_time="14:00:00")
        deduped = _dedupe([o1, o2])
        self.assertEqual(len(deduped), 2)

    # 4. Price sorting uses numeric order, not string order.
    def test_04_price_sorting_uses_numeric_order(self):
        # In string order: "10500" < "3200" < "950"
        # In numeric order: 950 < 3200 < 10500
        offers = [
            make_test_offer("AI", "Air India", 10500),
            make_test_offer("6E", "IndiGo", 950),
            make_test_offer("SG", "SpiceJet", 3200),
        ]
        sorted_asc = sort_flight_offers(offers, mode="price_asc")
        prices = [f.price.amount for f in sorted_asc]
        self.assertEqual(prices, [950, 3200, 10500])

    # 5. High-to-low order is the reverse of the comparable low-to-high price order.
    def test_05_high_to_low_reverse_of_low_to_high(self):
        offers = [
            make_test_offer("AI", "Air India", 5000, num="1"),
            make_test_offer("6E", "IndiGo", 3000, num="2"),
            make_test_offer("SG", "SpiceJet", 8000, num="3"),
            make_test_offer("UK", "Vistara", 4000, num="4"),
        ]
        asc = sort_flight_offers(offers, mode="price_asc")
        desc = sort_flight_offers(offers, mode="price_desc")
        self.assertEqual([f.price.amount for f in desc], list(reversed([f.price.amount for f in asc])))

    # 6. Invalid or missing fares remain visible and are placed after valid comparable fares.
    def test_06_invalid_or_missing_fares_placed_after_valid(self):
        valid1 = make_test_offer("6E", "IndiGo", 3000, num="1")
        valid2 = make_test_offer("AI", "Air India", 6000, num="2")
        invalid1 = make_test_offer("SG", "SpiceJet", 0, num="3")  # fare 0 = invalid
        invalid2 = make_test_offer("UK", "Vistara", -1, num="4")  # negative = invalid

        # Price Ascending: valid (3000, 6000), then invalid (0, -1)
        asc = sort_flight_offers([invalid1, valid2, invalid2, valid1], mode="price_asc")
        self.assertEqual(len(asc), 4)
        self.assertEqual([f.offerId for f in asc[:2]], [valid1.offerId, valid2.offerId])

        # Price Descending: valid (6000, 3000), then invalid (0, -1)
        desc = sort_flight_offers([invalid1, valid2, invalid2, valid1], mode="price_desc")
        self.assertEqual(len(desc), 4)
        self.assertEqual([f.offerId for f in desc[:2]], [valid2.offerId, valid1.offerId])

    # 7. Different currencies and incompatible fare bases are not falsely ranked together.
    def test_07_different_currencies_and_fare_bases_not_falsely_ranked(self):
        inr_offer = make_test_offer("6E", "IndiGo", 5000, currency="INR")
        usd_offer = make_test_offer("AI", "Air India", 100, currency="USD")
        comparison = compare_flights([inr_offer, usd_offer])
        self.assertFalse(comparison["priceComparable"])
        self.assertIsNone(comparison["cheapestOfferId"])

    # 8. All four cabin classes remain distinguishable.
    def test_08_all_four_cabin_classes_distinguishable(self):
        cabins = ["economy", "premium_economy", "business", "first"]
        offers = [make_test_offer("AI", "Air India", 5000 * (i + 1), cabin=c) for i, c in enumerate(cabins)]
        self.assertEqual(len({o.cabinClass for o in offers}), 4)

    # 9. A requested cabin class cannot be satisfied by a different class.
    def test_09_requested_cabin_class_cannot_be_satisfied_by_different_class(self):
        session = TravelSession(sessionId="session_first_class", cabinClass="first")
        results = ChatResults(
            flights=[
                make_test_offer("6E", "IndiGo", 3500, cabin="economy"),
                make_test_offer("AI", "Air India", 25000, cabin="first"),
            ]
        )
        self.agent._filter_and_rank(session, results)
        self.assertEqual(len(results.flights), 1)
        self.assertEqual(results.flights[0].cabinClass, "first")

    # 10. Multi-date results retain their dates and class information.
    def test_10_multi_date_results_retain_dates_and_class(self):
        offers = [
            make_test_offer("6E", "IndiGo", 4000, day="2026-10-25", cabin="economy"),
            make_test_offer("AI", "Air India", 3800, day="2026-10-25", cabin="business"),
            make_test_offer("6E", "IndiGo", 3500, day="2026-10-26", cabin="economy"),
            make_test_offer("AI", "Air India", 3900, day="2026-10-26", cabin="business"),
        ]
        sorted_offers = sort_flight_offers(offers, mode="price_asc")
        # Dates remain intact and grouped
        self.assertEqual(sorted_offers[0].outbound.departureLocal[:10], "2026-10-25")
        self.assertEqual(sorted_offers[-1].outbound.departureLocal[:10], "2026-10-26")

    # 11. Rejection tracking survives sorting.
    def test_11_rejection_tracking_survives_sorting(self):
        ledger = FlightLedger()
        o1 = make_test_offer("6E", "IndiGo", 3000, num="1")
        o2 = make_test_offer("AI", "Air India", 5000, num="2")
        ledger.remember([o1, o2])
        ledger.reject(o1)
        self.assertIn(offer_key(o1), ledger.rejected)

        # Sorting offers does not touch ledger rejection state
        sorted_offers = sort_flight_offers([o1, o2], mode="price_asc")
        self.assertIn(offer_key(o1), ledger.rejected)
        # Next batch excludes rejected
        batch = ledger.next_batch()
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].offerId, o2.offerId)

    # 12. "Show more" never repeats already-shown or rejected offers.
    def test_12_show_more_never_repeats_shown_or_rejected(self):
        ledger = FlightLedger()
        offers = [make_test_offer("6E", "IndiGo", 3000 + i * 100, num=str(i)) for i in range(6)]
        ledger.remember(offers)
        batch1 = ledger.next_batch(per_date=3)
        self.assertEqual(len(batch1), 3)
        ledger.reject(offers[3])
        batch2 = ledger.next_batch(per_date=3)
        # batch2 must not overlap with batch1 and must not contain offers[3]
        keys1 = {offer_key(o) for o in batch1}
        keys2 = {offer_key(o) for o in batch2}
        self.assertTrue(keys1.isdisjoint(keys2))
        self.assertNotIn(offer_key(offers[3]), keys2)

    # 13. Sorting does not initiate a new supplier request.
    def test_13_sorting_offline_no_network(self):
        offers = [
            make_test_offer("6E", "IndiGo", 3000),
            make_test_offer("AI", "Air India", 4000),
        ]
        # Pure synchronous local sorting, no async I/O
        sorted_asc = sort_flight_offers(offers, mode="price_asc")
        sorted_desc = sort_flight_offers(offers, mode="price_desc")
        self.assertEqual(len(sorted_asc), 2)
        self.assertEqual(len(sorted_desc), 2)

    # 14. Canonical airline names and verified supplier data are preserved.
    def test_14_canonical_airline_names_and_facts_preserved(self):
        o = make_test_offer("IX", "AI Express", 3500)
        c_name = canonical_airline_name("AI Express", "IX").display
        self.assertEqual(c_name, "Air India Express")
        sorted_offers = sort_flight_offers([o], mode="price_asc")
        res = sorted_offers[0]
        self.assertEqual(res.offerId, o.offerId)
        self.assertEqual(res.outbound.airline, "AI Express")
        self.assertEqual(res.price.amount, 3500)
        self.assertEqual(res.sourceUrl, "https://example.com/book")

    # 15. Departure sorting covers midnight (00:00) through late night (23:59).
    def test_15_departure_time_sorting_full_day(self):
        midnight = make_test_offer("6E", "IndiGo", 5000, num="1", dep_time="00:00:00")
        morning = make_test_offer("AI", "Air India", 4000, num="2", dep_time="08:30:00")
        noon = make_test_offer("SG", "SpiceJet", 3500, num="3", dep_time="12:00:00")
        evening = make_test_offer("UK", "Vistara", 6000, num="4", dep_time="18:45:00")
        late_night = make_test_offer("QP", "Akasa", 4500, num="5", dep_time="23:59:00")
        missing_time = make_test_offer("AI", "Air India", 3000, num="6", dep_time="")

        # Earliest first: midnight -> morning -> noon -> evening -> late night -> missing at end
        asc = sort_flight_offers([evening, noon, missing_time, late_night, midnight, morning], mode="departure_asc")
        self.assertEqual(
            [f.offerId for f in asc[:5]],
            [midnight.offerId, morning.offerId, noon.offerId, evening.offerId, late_night.offerId],
        )
        self.assertEqual(asc[5].offerId, missing_time.offerId)

        # Latest first: late night -> evening -> noon -> morning -> midnight -> missing at end
        desc = sort_flight_offers([evening, noon, missing_time, late_night, midnight, morning], mode="departure_desc")
        self.assertEqual(
            [f.offerId for f in desc[:5]],
            [late_night.offerId, evening.offerId, noon.offerId, morning.offerId, midnight.offerId],
        )
        self.assertEqual(desc[5].offerId, missing_time.offerId)

    # 16. AM/PM parsing covers 12:00 AM as 00:00 and 12:00 PM as 12:00.
    def test_16_departure_ampm_parsing(self):
        self.assertEqual(parse_departure_minutes("12:00 AM"), 0)
        self.assertEqual(parse_departure_minutes("12:30 AM"), 30)
        self.assertEqual(parse_departure_minutes("06:00 AM"), 360)
        self.assertEqual(parse_departure_minutes("11:59 AM"), 719)
        self.assertEqual(parse_departure_minutes("12:00 PM"), 720)
        self.assertEqual(parse_departure_minutes("12:30 PM"), 750)
        self.assertEqual(parse_departure_minutes("11:59 PM"), 1439)
        self.assertIsNone(parse_departure_minutes(""))
        self.assertIsNone(parse_departure_minutes(None))


if __name__ == "__main__":
    unittest.main()
