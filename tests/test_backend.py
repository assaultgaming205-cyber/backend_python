import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.agent.intent.guard import TravelDomainGuard


def test_domain_guard_refusal():
    # 1. Non-travel gadgets / shopping
    res1 = TravelDomainGuard.evaluate("what is the price of iphone 15 pro in flipkart bbd sale?", character="meera")
    assert res1.isOutOfScope is True
    assert "travel agent" in res1.refusalMessage.lower() or "travel buddy" in res1.refusalMessage.lower()
    assert "nahi bata sakti" in res1.refusalMessage

    # Kabir gender check
    res_kabir = TravelDomainGuard.evaluate("what is the price of iphone 15?", character="kabir")
    assert res_kabir.isOutOfScope is True
    assert "nahi bata sakta" in res_kabir.refusalMessage

    # 2. Non-travel coding
    res2 = TravelDomainGuard.evaluate("write a python code to reverse a binary tree", character="meera")
    assert res2.isOutOfScope is True

    # 3. Non-travel recipe / food
    res3 = TravelDomainGuard.evaluate("how to make biryani recipe banane ki vidhi", character="meera")
    assert res3.isOutOfScope is True

    # 4. Pure general question without travel
    res4 = TravelDomainGuard.evaluate("who won the cricket match yesterday?", character="meera")
    assert res4.isOutOfScope is True

    # 5. Legitimate travel should NOT be refused
    res_travel1 = TravelDomainGuard.evaluate("Delhi se Mumbai flight chahiye kal", character="meera")
    assert res_travel1.isOutOfScope is False

    res_travel2 = TravelDomainGuard.evaluate("train from Bangalore to Chennai on 25th", character="meera")
    assert res_travel2.isOutOfScope is False

    res_greeting = TravelDomainGuard.evaluate("Hello Meera kaise ho?", character="meera")
    assert res_greeting.isOutOfScope is False
    print("✅ All TravelDomainGuard tests passed!")


@pytest.mark.anyio
async def test_fastapi_endpoints():
    from unittest.mock import AsyncMock, patch
    from app.api.chat import agent_service
    from app.ai.ai_service import AiResponsePayload
    from app.schemas.chat import ToolCallSummary

    async def fake_execute(name, args, results):
        return ToolCallSummary(name=name, args=args, success=True, count=0, durationMs=1)

    fake_ai = AsyncMock(
        return_value=AiResponsePayload(
            message={"role": "assistant", "content": "Main aapki travel buddy hoon! Delhi se Mumbai ke options check kiye hain."},
            content="Main aapki travel buddy hoon! Delhi se Mumbai ke options check kiye hain.",
            model="offline-mock",
            raw="",
        )
    )

    transport = ASGITransport(app=app)
    with patch.object(agent_service.executor, "execute", side_effect=fake_execute), patch.object(
        agent_service.ai, "chat", fake_ai
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Test Health
            resp_health = await client.get("/api/health")
            assert resp_health.status_code == 200
            data_health = resp_health.json()
            assert data_health["service"] == "travel-buddy-python"

            # Test Chat with non-travel query (domain guard refusal filter)
            chat_payload = {
                "messages": [
                    {"role": "user", "content": "Can you give me the specs and discount on iPhone 15?"}
                ],
                "character": "meera",
            }
            resp_chat = await client.post("/api/chat", json=chat_payload)
            assert resp_chat.status_code == 200
            chat_data = resp_chat.json()
            assert "travel" in chat_data["message"]["content"].lower()

            # Test Chat with legitimate travel query (100% offline mocked)
            travel_payload = {
                "messages": [
                    {"role": "user", "content": "Delhi se Mumbai flight aur train dono compare karo kal"}
                ],
                "character": "meera",
            }
            resp_travel = await client.post("/api/chat", json=travel_payload)
            assert resp_travel.status_code == 200
            travel_data = resp_travel.json()
            assert "message" in travel_data


if __name__ == "__main__":
    test_domain_guard_refusal()
    asyncio.run(test_fastapi_endpoints())
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
