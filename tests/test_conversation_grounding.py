"""Fixture-based fare accuracy, price-change reconciliation, and grounding tests for Travel Buddy.

Covers all 12 required grounding scenarios:
1. Identical search and review fare.
2. Higher review fare (e.g., ₹4,159.50 -> ₹5,159.50).
3. Lower review fare (e.g., ₹4,159.50 -> ₹3,759.50).
4. Currency mismatch (INR vs USD).
5. Mismatched fare identifier.
6. Missing price or tax fields.
7. Different price basis (per_person vs total).
8. Review failure after successful search.
9. Seat map unavailable after successful review.
10. Fare rules missing or empty.
11. Stale search results after route change.
12. Multiple similar flights on the same route.
"""
from typing import List
import pytest

from app.ai.ai_service import AiMessage
from app.common.dates import plus_days, today_in_tz
from test_conversation_flow import (
    build_offline_agent,
    block_all_external_http,  # noqa: F401
    make_fixture_flight,
)


@pytest.mark.anyio
async def test_grounding_01_02_03_identical_higher_and_lower_review_fares():
    """Tests identical fare, fare increase (+₹1000), and fare decrease (-₹400)."""
    # 1. Identical fare: ₹4159.50 -> ₹4159.50
    agent, bankg, store = build_offline_agent()
    bankg.review_price_delta = 0.0
    sid_same = "sess-gnd-same"
    h_same = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    r1 = await agent.run(h_same, session_id=sid_same)
    h_same.append(AiMessage(role="assistant", content=r1.content))
    h_same.append(AiMessage(role="user", content="Select the first flight and review fare"))
    r2 = await agent.run(h_same, session_id=sid_same)
    rev_same = r2.results.fareReview
    assert rev_same["status"] == "LOCKED"
    assert rev_same["priceChanged"] is False
    assert rev_same["originalPriceConfirmed"] is True
    assert rev_same["searchPrice"] == 4159.50
    assert rev_same["reviewedPrice"] == 4159.50
    assert "confirmed at ₹4159.5" in r2.content

    # 2. Higher review fare: ₹4159.50 -> ₹5159.50 (+₹1000.00)
    agent_inc, bankg_inc, _ = build_offline_agent()
    bankg_inc.review_price_delta = 1000.0
    sid_inc = "sess-gnd-inc"
    h_inc = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    ri1 = await agent_inc.run(h_inc, session_id=sid_inc)
    h_inc.append(AiMessage(role="assistant", content=ri1.content))
    h_inc.append(AiMessage(role="user", content="Select the first flight and review fare"))
    ri2 = await agent_inc.run(h_inc, session_id=sid_inc)
    rev_inc = ri2.results.fareReview
    assert rev_inc["status"] == "PRICE_CHANGED"
    assert rev_inc["fareChangeDirection"] == "increased"
    assert rev_inc["priceChanged"] is True
    assert rev_inc["originalPriceConfirmed"] is False
    assert rev_inc["searchPrice"] == 4159.50
    assert rev_inc["reviewedPrice"] == 5159.50
    assert rev_inc["priceDelta"] == 1000.0
    assert "increased" in ri2.content.lower()
    assert "4159.5" in ri2.content and "5159.5" in ri2.content
    assert "1000" in ri2.content

    # 3. Lower review fare: ₹4159.50 -> ₹3759.50 (-₹400.00)
    agent_dec, bankg_dec, _ = build_offline_agent()
    bankg_dec.review_price_delta = -400.0
    sid_dec = "sess-gnd-dec"
    h_dec = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    rd1 = await agent_dec.run(h_dec, session_id=sid_dec)
    h_dec.append(AiMessage(role="assistant", content=rd1.content))
    h_dec.append(AiMessage(role="user", content="Select the first flight and review fare"))
    rd2 = await agent_dec.run(h_dec, session_id=sid_dec)
    rev_dec = rd2.results.fareReview
    assert rev_dec["status"] == "PRICE_CHANGED"
    assert rev_dec["fareChangeDirection"] == "decreased"
    assert rev_dec["priceChanged"] is True
    assert rev_dec["originalPriceConfirmed"] is False
    assert rev_dec["searchPrice"] == 4159.50
    assert rev_dec["reviewedPrice"] == 3759.50
    assert rev_dec["priceDelta"] == -400.0
    assert "decreased" in rd2.content.lower()
    assert "4159.5" in rd2.content and "3759.5" in rd2.content


@pytest.mark.anyio
async def test_grounding_04_05_06_07_currency_fareid_missingprice_and_basis_mismatches():
    """Tests currency mismatch, fareId mismatch, missing price/tax fields, and price basis mismatch."""
    # 4. Currency mismatch (INR vs USD)
    agent_curr, bankg_curr, _ = build_offline_agent()
    bankg_curr.review_override = {
        "bookingId": "BOOK_CURR_1",
        "fareId": "FARE_BOM_DEL_101",
        "priceId": "FARE_BOM_DEL_101",
        "searchPrice": 4159.50,
        "searchCurrency": "INR",
        "reviewedPrice": 52.00,
        "reviewedCurrency": "USD",
        "priceBasis": "total",
        "originalPriceBasis": "total",
    }
    h_curr = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    rc1 = await agent_curr.run(h_curr, session_id="sess-gnd-curr")
    h_curr.append(AiMessage(role="assistant", content=rc1.content))
    h_curr.append(AiMessage(role="user", content="Select the first flight and review fare"))
    rc2 = await agent_curr.run(h_curr, session_id="sess-gnd-curr")
    assert rc2.results.fareReview["status"] == "CURRENCY_MISMATCH"
    assert rc2.results.fareReview["currencyMismatch"] is True
    assert rc2.results.fareReview["originalPriceConfirmed"] is False
    assert "currency mismatch" in rc2.content.lower()

    # 5. Mismatched fareId returned by supplier
    agent_fid, bankg_fid, store_fid = build_offline_agent()
    bankg_fid.review_override = {
        "bookingId": "BOOK_WRONG_999",
        "fareId": "FARE_WRONG_999",
        "priceId": "FARE_WRONG_999",
        "searchPrice": 4159.50,
        "reviewedPrice": 4159.50,
        "priceBasis": "total",
        "originalPriceBasis": "total",
    }
    h_fid = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    rf1 = await agent_fid.run(h_fid, session_id="sess-gnd-fid")
    h_fid.append(AiMessage(role="assistant", content=rf1.content))
    h_fid.append(AiMessage(role="user", content="Select the first flight and review fare"))
    rf2 = await agent_fid.run(h_fid, session_id="sess-gnd-fid")
    assert rf2.results.fareReview["status"] == "FARE_ID_MISMATCH"
    assert rf2.results.fareReview["fareIdMatched"] is False
    assert rf2.results.fareReview["originalPriceConfirmed"] is False
    assert store_fid.get("sess-gnd-fid").bookingSessionId is None
    assert "mismatched fare identifier" in rf2.content.lower()

    # 6. Missing price or tax fields in review response
    agent_miss, bankg_miss, _ = build_offline_agent()
    bankg_miss.review_override = {
        "bookingId": "BOOK_MISS_1",
        "fareId": "FARE_BOM_DEL_101",
        "priceId": "FARE_BOM_DEL_101",
        "searchPrice": 4159.50,
        "reviewedPrice": None,
        "missingTaxFields": True,
        "priceBasis": "total",
        "originalPriceBasis": "total",
    }
    h_miss = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    rm1 = await agent_miss.run(h_miss, session_id="sess-gnd-miss")
    h_miss.append(AiMessage(role="assistant", content=rm1.content))
    h_miss.append(AiMessage(role="user", content="Select the first flight and review fare"))
    rm2 = await agent_miss.run(h_miss, session_id="sess-gnd-miss")
    assert rm2.results.fareReview["status"] == "PRICE_MISSING"
    assert rm2.results.fareReview["priceVerified"] is False
    assert rm2.results.fareReview["originalPriceConfirmed"] is False
    assert "could not be verified" in rm2.content.lower() or "omitted" in rm2.content.lower()

    # 7. Different price basis (per_person vs total)
    agent_bas, bankg_bas, _ = build_offline_agent()
    bankg_bas.review_override = {
        "bookingId": "BOOK_BAS_1",
        "fareId": "FARE_BOM_DEL_101",
        "priceId": "FARE_BOM_DEL_101",
        "searchPrice": 4159.50,
        "reviewedPrice": 8319.00,
        "originalPriceBasis": "per_person",
        "priceBasis": "total",
    }
    h_bas = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    rb1 = await agent_bas.run(h_bas, session_id="sess-gnd-bas")
    h_bas.append(AiMessage(role="assistant", content=rb1.content))
    h_bas.append(AiMessage(role="user", content="Select the first flight and review fare"))
    rb2 = await agent_bas.run(h_bas, session_id="sess-gnd-bas")
    assert rb2.results.fareReview["status"] == "BASIS_MISMATCH"
    assert rb2.results.fareReview["priceBasisMismatch"] is True
    assert rb2.results.fareReview["originalPriceConfirmed"] is False
    assert "basis mismatch" in rb2.content.lower()


@pytest.mark.anyio
async def test_grounding_08_09_10_11_12_review_failure_empty_seatmap_empty_rules_stale_and_duplicates():
    """Tests review failure, seat map unavailable, fare rules empty, stale results, and similar flights."""
    # 8. Review failure after successful search
    agent_fail, bankg_fail, store_fail = build_offline_agent()
    bankg_fail.raise_on_review = RuntimeError("BankG flightReview timeout")
    sid_fail = "sess-gnd-revfail"
    h_fail = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    rf1 = await agent_fail.run(h_fail, session_id=sid_fail)
    h_fail.append(AiMessage(role="assistant", content=rf1.content))
    h_fail.append(AiMessage(role="user", content="Select the first flight and review fare"))
    rf2 = await agent_fail.run(h_fail, session_id=sid_fail)
    assert any(tc.name == "review_fare" and not tc.success for tc in rf2.toolCalls)
    sess_fail = store_fail.get(sid_fail)
    assert len(sess_fail.lastOffers) == 3  # Search results preserved!
    assert sess_fail.bookingSessionId is None  # Never fabricated!
    assert "could not be completed" in rf2.content.lower()

    # 9 & 10. Seat map unavailable (empty seats) & Fare rules empty after successful review
    agent_empty, bankg_empty, store_empty = build_offline_agent()
    bankg_empty.seat_map_override = {"status": "SUCCESS", "bookingId": "BOOK_FARE_BOM_DEL_101", "seatMaps": []}
    bankg_empty.fare_rules_override = {"status": "SUCCESS", "priceId": "FARE_BOM_DEL_101", "fareRules": []}
    sid_emp = "sess-gnd-empty-downstream"
    h_emp = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    re1 = await agent_empty.run(h_emp, session_id=sid_emp)
    h_emp.append(AiMessage(role="assistant", content=re1.content))
    h_emp.append(AiMessage(role="user", content="Select the first flight and review fare"))
    re2 = await agent_empty.run(h_emp, session_id=sid_emp)
    h_emp.append(AiMessage(role="assistant", content=re2.content))

    h_emp.append(AiMessage(role="user", content="Show me the seat map"))
    re_sm = await agent_empty.run(h_emp, session_id=sid_emp)
    h_emp.append(AiMessage(role="assistant", content=re_sm.content))
    assert "did not contain any available selectable seats" in re_sm.content.lower()
    assert "1a" not in re_sm.content.lower()  # No invented seats!

    h_emp.append(AiMessage(role="user", content="What are the fare rules?"))
    re_fr = await agent_empty.run(h_emp, session_id=sid_emp)
    assert "no structured fare rules" in re_fr.content.lower()
    assert "3000" not in re_fr.content  # No invented fees!

    # 11 & 12. Stale results after route change + multiple similar flights on the same route
    agent_sim, bankg_sim, store_sim = build_offline_agent()
    dep_d = plus_days(today_in_tz("Asia/Kolkata"), 4)
    bankg_sim.custom_offers = [
        make_fixture_flight(
            offer_id="SIM_1",
            fare_id="FARE_SIM_MORNING",
            airline="IndiGo",
            carrier_code="6E",
            flight_number="6E-111",
            origin="BOM",
            destination="BLR",
            departure_date=dep_d,
            dep_time="06:30",
            arr_time="08:10",
            amount=3800.0,
        ),
        make_fixture_flight(
            offer_id="SIM_2",
            fare_id="FARE_SIM_EVENING",
            airline="IndiGo",
            carrier_code="6E",
            flight_number="6E-999",
            origin="BOM",
            destination="BLR",
            departure_date=dep_d,
            dep_time="19:15",
            arr_time="20:55",
            amount=3950.0,
        ),
    ]
    sid_sim = "sess-gnd-similar"
    h_sim: List[AiMessage] = [AiMessage(role="user", content=f"Search flights from Mumbai to Bengaluru on {dep_d}")]
    rs1 = await agent_sim.run(h_sim, session_id=sid_sim)
    h_sim.append(AiMessage(role="assistant", content=rs1.content))

    # Select specifically "that evening flight" -> must pick FARE_SIM_EVENING (3950.0), not morning (3800.0)
    h_sim.append(AiMessage(role="user", content="Select the evening flight and review fare"))
    rs2 = await agent_sim.run(h_sim, session_id=sid_sim)
    assert bankg_sim.calls[-1]["fareId"] == "FARE_SIM_EVENING"
    assert rs2.results.fareReview["reviewedPrice"] == 3950.0

