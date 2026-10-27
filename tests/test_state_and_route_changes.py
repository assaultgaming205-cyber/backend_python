"""State management, route/date/passenger/cabin change matrix, stale ID invalidation, and cross-session isolation tests."""
from typing import List
import pytest

from app.ai.ai_service import AiMessage
from app.common.dates import plus_days, today_in_tz
from test_conversation_flow import build_offline_agent, block_all_external_http  # noqa: F401


@pytest.mark.anyio
async def test_origin_destination_and_both_route_changes_invalidate_downstream_state():
    """Tests changing origin, destination, and both after fare review + seat map."""
    agent, bankg, store = build_offline_agent()
    sid = "sess-state-route-matrix"
    dep_d = plus_days(today_in_tz("Asia/Kolkata"), 3)
    history: List[AiMessage] = []

    # Step 1: Search BOM -> DEL on dep_d for 2 adults in business class
    history.append(AiMessage(role="user", content=f"Search business class flights from Mumbai to Delhi on {dep_d} for 2 adults"))
    r1 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r1.content))
    assert bankg.calls[-1]["origin"] == "BOM"
    assert bankg.calls[-1]["destination"] == "DEL"
    assert bankg.calls[-1]["adults"] == 2
    assert bankg.calls[-1]["cabin"] == "business"

    # Step 2: Review first flight & get seat map
    history.append(AiMessage(role="user", content="Select the first flight and review fare"))
    r2 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r2.content))
    history.append(AiMessage(role="user", content="Show me the seat map"))
    r3 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r3.content))

    sess = store.get(sid)
    assert sess.selectedFareId == "FARE_BOM_DEL_101"
    assert sess.bookingSessionId == "BOOK_FARE_BOM_DEL_101"
    assert sess.lastSeatMap is not None

    # Step 3: Change destination only ("Change destination to Bengaluru")
    # Must preserve origin=BOM, departureDate=dep_d, adults=2, cabin=business, and clear old review/seatmap
    history.append(AiMessage(role="user", content="Actually, change destination to Bengaluru"))
    r4 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r4.content))
    sess = store.get(sid)
    assert sess.origin.code == "BOM"
    assert sess.destination.code == "BLR"
    assert sess.dates.departureDate == dep_d
    assert sess.passengers.adults == 2
    assert sess.cabinClass == "business"
    assert sess.selectedFareId is None
    assert sess.bookingSessionId is None
    assert sess.reviewedFareData is None
    assert sess.lastSeatMap is None
    assert bankg.calls[-1]["origin"] == "BOM"
    assert bankg.calls[-1]["destination"] == "BLR"

    # Attempting to review the old BOM->DEL fareId on the new BOM->BLR search must be rejected
    history.append(AiMessage(role="user", content="Review fareId=FARE_BOM_DEL_101"))
    r_stale = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r_stale.content))
    assert r_stale.toolCalls == []
    assert "does not match" in r_stale.content.lower()

    # Step 4: Change origin only ("Change origin to Pune")
    history.append(AiMessage(role="user", content="Change origin to Pune"))
    r5 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r5.content))
    sess = store.get(sid)
    assert sess.origin.code == "PNQ"
    assert sess.destination.code == "BLR"
    assert sess.dates.departureDate == dep_d
    assert sess.passengers.adults == 2
    assert sess.cabinClass == "business"
    assert bankg.calls[-1]["origin"] == "PNQ"
    assert bankg.calls[-1]["destination"] == "BLR"

    # Step 5: Negated full route swap ("Not Pune to Bengaluru, Kolkata to Chennai")
    history.append(AiMessage(role="user", content="Not Pune to Bengaluru, Kolkata to Chennai"))
    r6 = await agent.run(history, session_id=sid)
    sess = store.get(sid)
    assert sess.origin.code == "CCU"
    assert sess.destination.code == "MAA"
    assert bankg.calls[-1]["origin"] == "CCU"
    assert bankg.calls[-1]["destination"] == "MAA"


@pytest.mark.anyio
async def test_date_passenger_cabin_and_triptype_changes_invalidate_review_state():
    """Tests date change, return date / roundtrip change, passenger change, and cabin class change."""
    agent, bankg, store = build_offline_agent()
    sid = "sess-state-params"
    d1 = plus_days(today_in_tz("Asia/Kolkata"), 2)
    d2 = plus_days(today_in_tz("Asia/Kolkata"), 5)
    d_ret = plus_days(today_in_tz("Asia/Kolkata"), 9)
    history: List[AiMessage] = []

    # Initial search & fare review
    history.append(AiMessage(role="user", content=f"Search flights from Mumbai to Delhi on {d1}"))
    r1 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r1.content))

    history.append(AiMessage(role="user", content="Select the first flight and review fare"))
    r2 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r2.content))
    sess = store.get(sid)
    assert sess.bookingSessionId == "BOOK_FARE_BOM_DEL_101"

    # 1. Departure date change -> preserves BOM->DEL, clears bookingSessionId, searches d2
    history.append(AiMessage(role="user", content=f"Move the travel date to {d2}"))
    r3 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r3.content))
    sess = store.get(sid)
    assert sess.origin.code == "BOM"
    assert sess.destination.code == "DEL"
    assert sess.dates.departureDate == d2
    assert sess.bookingSessionId is None
    assert sess.selectedFareId is None
    assert bankg.calls[-1]["departureDate"] == d2

    # 2. Switch to round-trip with return date -> updates tripType="roundtrip" and returnDate
    history.append(AiMessage(role="user", content=f"Make it a round trip returning on {d_ret}"))
    r4 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r4.content))
    sess = store.get(sid)
    assert sess.tripType == "roundtrip"
    assert sess.dates.returnDate == d_ret

    # 3. Switch back to one-way and change passengers to 3 adults, 1 child, 1 infant
    history.append(AiMessage(role="user", content="Actually make it one way for 3 adults, 1 child and 1 infant"))
    r5 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r5.content))
    sess = store.get(sid)
    assert sess.tripType == "oneway"
    assert sess.dates.returnDate is None
    assert sess.passengers.adults == 3
    assert sess.passengers.children == 1
    assert sess.passengers.infantsOnLap == 1
    assert bankg.calls[-1]["adults"] == 3
    assert bankg.calls[-1]["child"] == 1
    assert bankg.calls[-1]["infant"] == 1

    # 4. Change cabin class from economy to business
    history.append(AiMessage(role="user", content="Switch to business class"))
    r6 = await agent.run(history, session_id=sid)
    sess = store.get(sid)
    assert sess.cabinClass == "business"
    assert bankg.calls[-1]["cabin"] == "business"


@pytest.mark.anyio
async def test_airport_identity_bom_vs_nmi_and_preference_updates():
    """Tests explicit NMI vs BOM airport identity and preference updates (fastest, nonstop)."""
    agent, bankg, store = build_offline_agent()
    sid = "sess-nmi-bom"
    history: List[AiMessage] = []

    # Search explicitly from Navi Mumbai (NMI) to Delhi tomorrow
    history.append(AiMessage(role="user", content="Search flights from NMI to DEL tomorrow"))
    r1 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r1.content))
    sess = store.get(sid)
    assert sess.origin.code == "NMI"
    assert bankg.calls[-1]["origin"] == "NMI"

    # Follow-up preference: "Show me the fastest flight"
    history.append(AiMessage(role="user", content="Show me the fastest flight"))
    r2 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=r2.content))
    sess = store.get(sid)
    assert sess.rankingPreference == "fastest"
    assert r2.results.flights[0].outbound.durationMinutes == 120  # AI-806 is 120m vs 135m

    # Follow-up filter: "Only nonstop flights"
    history.append(AiMessage(role="user", content="Filter to only nonstop flights"))
    r3 = await agent.run(history, session_id=sid)
    sess = store.get(sid)
    assert sess.nonstopOnly is True
    assert all(f.outbound.stops == 0 for f in r3.results.flights)


@pytest.mark.anyio
async def test_cross_session_isolation_concurrent_sessions():
    """Verifies that two concurrent sessions never leak route, passenger, offer, or booking state."""
    agent, bankg, store = build_offline_agent()
    sid_a = "sess-concurrent-A"
    sid_b = "sess-concurrent-B"

    d_a = plus_days(today_in_tz("Asia/Kolkata"), 2)
    d_b = plus_days(today_in_tz("Asia/Kolkata"), 6)

    hist_a = [AiMessage(role="user", content=f"Search business class flights from Mumbai to Delhi on {d_a} for 2 adults")]
    hist_b = [AiMessage(role="user", content=f"Search economy flights from Bengaluru to Goa on {d_b} for 1 adult")]

    res_a1 = await agent.run(hist_a, session_id=sid_a)
    res_b1 = await agent.run(hist_b, session_id=sid_b)

    hist_a.append(AiMessage(role="assistant", content=res_a1.content))
    hist_b.append(AiMessage(role="assistant", content=res_b1.content))

    # Review first flight in session A
    hist_a.append(AiMessage(role="user", content="Select the first flight and review fare"))
    res_a2 = await agent.run(hist_a, session_id=sid_a)

    sess_a = store.get(sid_a)
    sess_b = store.get(sid_b)

    assert sess_a.origin.code == "BOM" and sess_a.destination.code == "DEL"
    assert sess_a.passengers.adults == 2 and sess_a.cabinClass == "business"
    assert sess_a.bookingSessionId == "BOOK_FARE_BOM_DEL_101"

    assert sess_b.origin.code == "BLR" and sess_b.destination.code == "GOI"
    assert sess_b.passengers.adults == 1 and sess_b.cabinClass == "economy"
    assert sess_b.bookingSessionId is None
    assert all(f.outbound.origin == "BLR" and f.outbound.destination == "GOI" for f in sess_b.lastOffers)

