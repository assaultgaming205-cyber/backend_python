import unittest
from app.agent.grounding.grounding_gate import GroundingGate
from app.schemas.chat import ChatResults
from app.schemas.flight import FlightOffer, FlightLeg, FlightSegment, Money
from app.schemas.train import TrainOption, TrainClassFare, Station
from app.schemas.hotel import HotelOffer, HotelPrice


class TestGroundingGateAdversarial(unittest.TestCase):
    """60 deterministic adversarial grounding test cases verifying strict fail-closed safety."""

    def setUp(self):
        self.verified_flight = FlightOffer(
            offerId="flt-1",
            provider="amadeus",
            price=Money(amount=4500.0, currency="INR"),
            outbound=FlightLeg(
                departureLocal="2026-11-15T14:30:00",
                arrivalLocal="2026-11-15T16:45:00",
                origin="DEL",
                destination="BOM",
                airline="IndiGo",
                durationMinutes=135,
                segments=[
                    FlightSegment(
                        departureLocal="2026-11-15T14:30:00",
                        arrivalLocal="2026-11-15T16:45:00",
                        origin="DEL",
                        destination="BOM",
                        carrierCode="6E",
                        flightNumber="6E-204",
                        carrierName="IndiGo",
                    )
                ]
            )
        )

        self.verified_train = TrainOption(
            trainNumber="12951",
            trainName="Mumbai Rajdhani",
            origin=Station(code="NDLS", name="New Delhi"),
            destination=Station(code="MMCT", name="Mumbai Central"),
            departureTime="16:55",
            arrivalTime="08:35",
            classFares=[
                TrainClassFare(classCode="3A", className="3rd AC", fare=2850.0),
                TrainClassFare(classCode="2A", className="2nd AC", fare=3990.0),
            ]
        )

        self.verified_hotel = HotelOffer(
            hotelId="htl-1",
            name="Taj Lands End",
            price=HotelPrice(amount=12500.0, currency="INR"),
            address="Bandra, Mumbai",
        )

        self.float_price_flight = FlightOffer(
            offerId="flt-float",
            provider="amadeus",
            price=Money(amount=4999.80, currency="INR"),
            outbound=FlightLeg(
                departureLocal="2026-11-15T09:15:00",
                arrivalLocal="2026-11-15T11:30:00",
                origin="DEL",
                destination="BOM",
                airline="Air India",
                durationMinutes=135,
                segments=[
                    FlightSegment(
                        departureLocal="2026-11-15T09:15:00",
                        arrivalLocal="2026-11-15T11:30:00",
                        origin="DEL",
                        destination="BOM",
                        carrierCode="AI",
                        flightNumber="AI-805",
                        carrierName="Air India",
                    )
                ]
            )
        )

        self.decimal_cents_flight = FlightOffer(
            offerId="flt-decimal",
            provider="amadeus",
            price=Money(amount=4500.99, currency="INR"),
            outbound=FlightLeg(
                departureLocal="2026-11-15T06:00:00",
                arrivalLocal="2026-11-15T08:15:00",
                origin="DEL",
                destination="BOM",
                airline="Akasa",
                durationMinutes=135,
                segments=[
                    FlightSegment(
                        departureLocal="2026-11-15T06:00:00",
                        arrivalLocal="2026-11-15T08:15:00",
                        origin="DEL",
                        destination="BOM",
                        carrierCode="QP",
                        flightNumber="QP-1102",
                        carrierName="Akasa",
                    )
                ]
            )
        )

    # --- CATEGORY 1: Empty Supplier Flights (Fail-closed on flight numbers) ---
    def test_adv_01_empty_flights_rejects_6e204(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Flight 6E-204 is available today.")
        self.assertFalse(ok)

    def test_adv_02_empty_flights_rejects_ai101(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("You can take AI-101 to Mumbai.")
        self.assertFalse(ok)

    def test_adv_03_empty_flights_rejects_sg819(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("SpiceJet flight SG-819 departs in the evening.")
        self.assertFalse(ok)

    def test_adv_04_empty_flights_rejects_uk955(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Vistara UK-955 leaves at night.")
        self.assertFalse(ok)

    def test_adv_05_empty_flights_rejects_qp1102(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Akasa Air QP-1102 is the earliest flight.")
        self.assertFalse(ok)

    # --- CATEGORY 2: Flight Number Verification ---
    def test_adv_06_verified_flight_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Flight 6E-204 is available today.")
        self.assertTrue(ok)

    def test_adv_07_invented_flight_rejected_when_another_verified(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Flight AI-101 is available today.")
        self.assertFalse(ok)

    def test_adv_08_flight_spacing_variation_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("IndiGo flight 6E 204 is direct.")
        self.assertTrue(ok)

    def test_adv_09_flight_lowercase_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Aap 6e-204 book kar sakte hain.")
        self.assertTrue(ok)

    def test_adv_10_airline_name_alone_without_number(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("IndiGo provides non-stop flights on this route.")
        self.assertTrue(ok)

    # --- CATEGORY 3: Empty Supplier Trains (Fail-closed on train numbers) ---
    def test_adv_11_empty_trains_rejects_12951(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Train 12951 Rajdhani Express is scheduled.")
        self.assertFalse(ok)

    def test_adv_12_empty_trains_rejects_12002(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Shatabdi Express train 12002 has seats.")
        self.assertFalse(ok)

    def test_adv_13_empty_trains_rejects_22436(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Vande Bharat train 22436 runs fast.")
        self.assertFalse(ok)

    def test_adv_14_empty_trains_rejects_12301(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Howrah Rajdhani 12301 is running on time.")
        self.assertFalse(ok)

    def test_adv_15_empty_trains_rejects_leading_zero_train(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Special train 02132 is available for booking.")
        self.assertFalse(ok)

    # --- CATEGORY 4: Train Number Verification ---
    def test_adv_16_verified_train_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(trains=[self.verified_train]))
        ok, _ = gate.is_sentence_grounded("Train 12951 Mumbai Rajdhani is available.")
        self.assertTrue(ok)

    def test_adv_17_invented_train_rejected_when_another_verified(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(trains=[self.verified_train]))
        ok, _ = gate.is_sentence_grounded("Train 12002 Shatabdi is available.")
        self.assertFalse(ok)

    def test_adv_18_train_with_hash_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(trains=[self.verified_train]))
        ok, _ = gate.is_sentence_grounded("Option train #12951 departs from New Delhi.")
        self.assertTrue(ok)

    def test_adv_19_train_number_in_hinglish_sentence(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(trains=[self.verified_train]))
        ok, _ = gate.is_sentence_grounded("Aap train number 12951 se travel kar sakte hain.")
        self.assertTrue(ok)

    def test_adv_20_train_name_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(trains=[self.verified_train]))
        ok, _ = gate.is_sentence_grounded("Mumbai Rajdhani is an excellent option.")
        self.assertTrue(ok)

    # --- CATEGORY 5: Schedule Times (Empty Supplier Times) ---
    def test_adv_21_empty_schedule_rejects_1430(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("The flight departs at 14:30.")
        self.assertFalse(ok)

    def test_adv_22_empty_schedule_rejects_0915(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Arrival is expected at 09:15.")
        self.assertFalse(ok)

    def test_adv_23_empty_schedule_rejects_230_pm(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Departure time is 2:30 PM.")
        self.assertFalse(ok)

    def test_adv_24_empty_schedule_rejects_1800(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("The train leaves at 18:00 sharp.")
        self.assertFalse(ok)

    def test_adv_25_empty_schedule_rejects_2145(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Landing will be at 21:45.")
        self.assertFalse(ok)

    # --- CATEGORY 6: Verified Schedule Times (24h & 12h Formats) ---
    def test_adv_26_verified_24h_time_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Flight 6E-204 departs at 14:30.")
        self.assertTrue(ok)

    def test_adv_27_verified_arrival_time_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("It reaches Mumbai at 16:45.")
        self.assertTrue(ok)

    def test_adv_28_verified_12h_format_pm_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Flight 6E-204 leaves at 2:30 PM.")
        self.assertTrue(ok)

    def test_adv_29_invented_time_rejected_when_another_verified(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Flight 6E-204 departs at 18:00.")
        self.assertFalse(ok)

    def test_adv_30_invented_12h_am_time_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Flight 6E-204 departs at 2:30 AM.")
        self.assertFalse(ok)

    # --- CATEGORY 7: Missing Fares (Empty Supplier Fares) ---
    def test_adv_31_empty_fares_rejects_rupee_4500(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("The total price is ₹4,500.")
        self.assertFalse(ok)

    def test_adv_32_empty_fares_rejects_rs_1200(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Ticket fare is Rs. 1,200 per passenger.")
        self.assertFalse(ok)

    def test_adv_33_empty_fares_rejects_5000_inr(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Cost is 5000 INR for one-way journey.")
        self.assertFalse(ok)

    def test_adv_34_empty_fares_rejects_999(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Special fare starting at ₹999.")
        self.assertFalse(ok)

    def test_adv_35_empty_fares_rejects_decimal_fare(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Fare comes to ₹4,500.50 with taxes.")
        self.assertFalse(ok)

    # --- CATEGORY 8: Decimal Precision & Unsafe Truncation Prevention ---
    def test_adv_36_decimal_mismatch_rejected_integer_supplier(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))  # 4500.0
        ok, _ = gate.is_sentence_grounded("The flight ticket is ₹4,500.99.")
        self.assertFalse(ok, "Claiming ₹4,500.99 when supplier is 4500.0 must be rejected")

    def test_adv_37_decimal_mismatch_half_rupee_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))  # 4500.0
        ok, _ = gate.is_sentence_grounded("The flight ticket is ₹4,500.50.")
        self.assertFalse(ok)

    def test_adv_38_exact_decimal_match_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.decimal_cents_flight]))  # 4500.99
        ok, _ = gate.is_sentence_grounded("Akasa QP-1102 costs ₹4,500.99.")
        self.assertTrue(ok)

    def test_adv_39_decimal_truncated_claim_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.decimal_cents_flight]))  # 4500.99
        ok, _ = gate.is_sentence_grounded("Akasa QP-1102 costs ₹4,500.")
        self.assertFalse(ok, "Truncating ₹4,500.99 to ₹4,500 is unsafe")

    def test_adv_40_trailing_zeroes_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))  # 4500.0
        ok, _ = gate.is_sentence_grounded("The ticket costs ₹4,500.00.")
        self.assertTrue(ok)

    # --- CATEGORY 9: No Arbitrary ±1 Rounding Tolerance ---
    def test_adv_41_plus_one_rupee_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))  # 4500.0
        ok, _ = gate.is_sentence_grounded("The flight fare is ₹4,501.")
        self.assertFalse(ok)

    def test_adv_42_minus_one_rupee_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))  # 4500.0
        ok, _ = gate.is_sentence_grounded("The flight fare is ₹4,499.")
        self.assertFalse(ok)

    def test_adv_43_plus_two_rupees_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))  # 4500.0
        ok, _ = gate.is_sentence_grounded("The flight fare is ₹4,502.")
        self.assertFalse(ok)

    def test_adv_44_train_fare_plus_one_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(trains=[self.verified_train]))  # 2850.0
        ok, _ = gate.is_sentence_grounded("3A fare is ₹2,851.")
        self.assertFalse(ok)

    def test_adv_45_train_fare_minus_one_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(trains=[self.verified_train]))  # 2850.0
        ok, _ = gate.is_sentence_grounded("3A fare is ₹2,849.")
        self.assertFalse(ok)

    # --- CATEGORY 10: Defined Display/Rounding Policy on Float Supplier Prices ---
    def test_adv_46_float_rounded_up_to_nearest_integer_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.float_price_flight]))  # 4999.80
        ok, _ = gate.is_sentence_grounded("Air India AI-805 costs ₹5,000.")
        self.assertTrue(ok)

    def test_adv_47_float_exact_decimal_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.float_price_flight]))  # 4999.80
        ok, _ = gate.is_sentence_grounded("Air India AI-805 costs ₹4,999.80.")
        self.assertTrue(ok)

    def test_adv_48_float_invented_rounding_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.float_price_flight]))  # 4999.80
        ok, _ = gate.is_sentence_grounded("Air India AI-805 costs ₹5,010.")
        self.assertFalse(ok)

    # --- CATEGORY 11: Alternative Currency Formatting With Valid Fares ---
    def test_adv_49_rs_dot_format_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Fare is Rs. 4,500 only.")
        self.assertTrue(ok)

    def test_adv_50_rs_no_dot_format_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Fare is Rs 4,500 only.")
        self.assertTrue(ok)

    def test_adv_51_inr_format_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Booking charge is 4500 INR.")
        self.assertTrue(ok)

    def test_adv_52_rupees_word_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("Iska ticket 4500 rupees ka hai.")
        self.assertTrue(ok)

    # --- CATEGORY 12: Conversational Sentences & Safe Fallbacks ---
    def test_adv_53_conversational_intro_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Main aapke liye best travel options check kar raha hoon.")
        self.assertTrue(ok)

    def test_adv_54_polite_closing_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Aap inme se kaunsa option prefer karenge?")
        self.assertTrue(ok)

    def test_adv_55_missing_evidence_fallback_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults())
        ok, _ = gate.is_sentence_grounded("Mujhe is route ke verified flight numbers nahi mile, kripya cards check karein.")
        self.assertTrue(ok)

    # --- CATEGORY 13: Combined Multi-Claim Safety ---
    def test_adv_56_all_verified_facts_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        ok, _ = gate.is_sentence_grounded("IndiGo flight 6E-204 departs at 14:30 for ₹4,500.")
        self.assertTrue(ok)

    def test_adv_57_one_invented_fact_in_multi_claim_rejects(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(flights=[self.verified_flight]))
        # 6E-204 and 14:30 are valid, but fare ₹9,999 is invented
        ok, _ = gate.is_sentence_grounded("IndiGo flight 6E-204 departs at 14:30 for ₹9,999.")
        self.assertFalse(ok)

    def test_adv_58_hotel_verified_price_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(hotels=[self.verified_hotel]))
        ok, _ = gate.is_sentence_grounded("Taj Lands End costs ₹12,500 per night.")
        self.assertTrue(ok)

    def test_adv_59_hotel_invented_price_rejected(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(hotels=[self.verified_hotel]))
        ok, _ = gate.is_sentence_grounded("Taj Lands End costs ₹7,500 per night.")
        self.assertFalse(ok)

    def test_adv_60_hotel_address_verified_accepted(self):
        gate = GroundingGate.from_results_and_facts(ChatResults(hotels=[self.verified_hotel]))
        ok, _ = gate.is_sentence_grounded("Taj Lands End is situated in Bandra, Mumbai.")
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
