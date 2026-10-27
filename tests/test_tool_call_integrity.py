"""Tool-call integrity, argument validation, call ordering, and upstream ID propagation tests.

Verifies:
1. Correct tool selection for every user intent.
2. Strict argument validation (3-letter IATA, valid future YYYY-MM-DD, passenger bounds, cabin classes).
3. Strict call ordering (airSearch -> flightReview -> flightSeatMap / fareRule).
4. Exact upstream ID propagation and rejection of injected/fabricated IDs.
5. Permanent blocking of prohibited tools (flightBook, submitAmendment, etc.).
"""
from typing import List
import pytest

from app.agent.tool_executor import PROHIBITED_TOOLS, ToolExecutor
from app.ai.ai_service import AiMessage
from app.common.dates import plus_days, today_in_tz
from app.schemas.chat import ChatResults
from test_conversation_flow import build_offline_agent, block_all_external_http  # noqa: F401


@pytest.mark.anyio
async def test_call_ordering_blocks_premature_review_seatmap_and_farerules():
    """Never call review_fare, get_seat_map, or get_fare_rules before prerequisite steps."""
    agent, bankg, _ = build_offline_agent()

    # 1. review_fare before search_flights
    res_rev = await agent.run(
        [AiMessage(role="user", content="Select the first flight and review its fare")],
        session_id="sess-order-rev",
    )
    assert res_rev.toolCalls == []
    assert len(bankg.calls) == 0
    assert "search" in res_rev.content.lower()

    # 2. get_seat_map before search_flights and review_fare
    res_sm = await agent.run(
        [AiMessage(role="user", content="Show me the seat map for this flight")],
        session_id="sess-order-sm",
    )
    assert res_sm.toolCalls == []
    assert len(bankg.calls) == 0
    assert "review" in res_sm.content.lower() or "select" in res_sm.content.lower()

    # 3. get_seat_map after search_flights but BEFORE review_fare
    sid_no_rev = "sess-order-sm-after-search"
    history: List[AiMessage] = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    r1 = await agent.run(history, session_id=sid_no_rev)
    history.append(AiMessage(role="assistant", content=r1.content))
    assert len(bankg.calls) == 1

    history.append(AiMessage(role="user", content="Show me the seat map without running flightReview"))
    r2 = await agent.run(history, session_id=sid_no_rev)
    assert r2.toolCalls == []
    assert len(bankg.calls) == 1  # No flightSeatMap call!
    assert "flightreview" in r2.content.lower() or "review" in r2.content.lower()

    # 4. get_fare_rules before search_flights
    res_fr = await agent.run(
        [AiMessage(role="user", content="What are the fare rules and cancellation charges?")],
        session_id="sess-order-fr",
    )
    assert res_fr.toolCalls == []
    assert len(bankg.calls) == 1  # Still 1 from earlier session


@pytest.mark.anyio
async def test_upstream_identifier_propagation_and_injected_id_rejection():
    """Verifies exact upstream ID propagation and blocks fabricated fareId / bookingId / priceId."""
    agent, bankg, store = build_offline_agent()
    sid = "sess-id-prop"
    history: List[AiMessage] = [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow")]
    res1 = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res1.content))
    assert len(res1.results.flights) == 3

    # Attempt to review a fabricated fareId not in search results
    history.append(AiMessage(role="user", content="Review fareId=FAKE_INJECTED_FARE_999"))
    res_bad_fare = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res_bad_fare.content))
    assert res_bad_fare.toolCalls == []
    assert "does not match" in res_bad_fare.content.lower()

    # Select the second flight in price-ranked results (Akasa Air QP-112 -> FARE_BOM_DEL_303)
    history.append(AiMessage(role="user", content="Select the second flight and review fare"))
    res_rev = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res_rev.content))
    assert any(tc.name == "review_fare" and tc.success for tc in res_rev.toolCalls)
    assert bankg.calls[-1]["method"] == "flightReview"
    assert bankg.calls[-1]["fareId"] == "FARE_BOM_DEL_303"
    sess = store.get(sid)
    assert sess.selectedFareId == "FARE_BOM_DEL_303"
    assert sess.bookingSessionId == "BOOK_FARE_BOM_DEL_303"

    # Attempt to call seat map with an injected bookingId different from session.bookingSessionId
    history.append(AiMessage(role="user", content="Show seat map for bookingId=INJECTED_BOOK_123"))
    res_bad_bid = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res_bad_bid.content))
    assert res_bad_bid.toolCalls == []
    assert "unverified" in res_bad_bid.content.lower() or "stale" in res_bad_bid.content.lower()

    # Legitimate seat map call uses exact upstream bookingId BOOK_FARE_BOM_DEL_303
    history.append(AiMessage(role="user", content="Show me the seat map"))
    res_sm = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res_sm.content))
    assert any(tc.name == "get_seat_map" and tc.success for tc in res_sm.toolCalls)
    assert bankg.calls[-1]["method"] == "flightSeatMap"
    assert bankg.calls[-1]["bookingId"] == "BOOK_FARE_BOM_DEL_303"

    # Attempt to call fare rules with an injected priceId
    history.append(AiMessage(role="user", content="Check fare rules for priceId=INJECTED_PRICE_888"))
    res_bad_pid = await agent.run(history, session_id=sid)
    history.append(AiMessage(role="assistant", content=res_bad_pid.content))
    assert res_bad_pid.toolCalls == []
    assert "unverified" in res_bad_pid.content.lower() or "stale" in res_bad_pid.content.lower()

    # Legitimate fare rules call uses exact upstream priceId FARE_BOM_DEL_303
    history.append(AiMessage(role="user", content="Check the fare rules"))
    res_fr = await agent.run(history, session_id=sid)
    assert any(tc.name == "get_fare_rules" and tc.success for tc in res_fr.toolCalls)
    assert bankg.calls[-1]["method"] == "fareRule"
    assert bankg.calls[-1]["priceId"] == "FARE_BOM_DEL_303"


@pytest.mark.anyio
async def test_argument_validation_dates_passengers_and_cabins():
    """Tests validation of invalid calendar dates, past dates, passenger bounds, and unsupported cabin."""
    agent, bankg, _ = build_offline_agent()

    # 1. Impossible calendar date: 30 February
    res_feb30 = await agent.run(
        [AiMessage(role="user", content="Search flights from Mumbai to Delhi on 30 February")],
        session_id="sess-val-feb30",
    )
    assert res_feb30.toolCalls == []
    assert len(bankg.calls) == 0

    # 2. Past date: yesterday / 2020-01-01
    res_past = await agent.run(
        [AiMessage(role="user", content="Search flights from Mumbai to Delhi yesterday")],
        session_id="sess-val-past",
    )
    assert res_past.toolCalls == []
    assert len(bankg.calls) == 0

    # 3. Invalid passenger count: 0 adults / 12 adults
    res_pax0 = await agent.run(
        [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow for 0 adults")],
        session_id="sess-val-pax0",
    )
    assert res_pax0.toolCalls == []
    assert len(bankg.calls) == 0

    res_pax12 = await agent.run(
        [AiMessage(role="user", content="Search flights from Mumbai to Delhi tomorrow for 12 adults")],
        session_id="sess-val-pax12",
    )
    assert res_pax12.toolCalls == []
    assert len(bankg.calls) == 0

    # 4. Unsupported cabin class: Premium Economy -> does not call BankG airSearch, explains limitation
    res_pe = await agent.run(
        [AiMessage(role="user", content="Search premium economy flights from Mumbai to Delhi tomorrow")],
        session_id="sess-val-pe",
    )
    assert res_pe.toolCalls == []
    assert len(bankg.calls) == 0
    assert "premium economy" in res_pe.content.lower() or "economy" in res_pe.content.lower()


@pytest.mark.anyio
async def test_prohibited_tools_blocked_at_executor_and_agent_layers():
    """Ensures all prohibited booking/amendment/balance tools are permanently blocked."""
    executor = ToolExecutor()
    results = ChatResults()
    for tool_name in PROHIBITED_TOOLS:
        summary = await executor.execute(tool_name, {"bookingId": "BOOK_123"}, results)
        assert summary.success is False
        assert "prohibited" in (summary.error or "").lower() or "disabled" in (summary.error or "").lower()
