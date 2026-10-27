import re
import json
from typing import Dict, Any, Optional
from .grounded_reply import Facts, parse_reply_json


def validate_draft(
    raw: str,
    facts: Optional[Facts] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = context or {}
    raw_clean = raw.strip()
    if not raw_clean:
        return {"valid": False, "reason": "Empty draft"}

    parsed = parse_reply_json(raw_clean)
    reply = parsed.get("reply", "").strip()

    if not reply or len(reply) > 2000:
        return {"valid": False, "reason": "Reply must contain 1-2000 characters."}

    # Don't invent today/tomorrow if past mistake
    if context.get("pastMistake") and not context.get("departureDate") and re.search(r"\b(?:kal|kaal|tomorrow|today|aaj)\b", reply, re.IGNORECASE):
        return {
            "valid": False,
            "reason": "Departure date is unknown. The user forgot yesterday; do not reinterpret that as travel today or tomorrow."
        }

    # Booking confirmation claim check
    prose = re.sub(r"\[\[[^\]]+\]\]", "", reply)
    if re.search(
        r"\b(?:your (?:ticket|flight|hotel|booking) (?:is|has been) (?:booked|confirmed)|(?:successfully|already) booked|payment (?:is )?(?:successful|complete)|PNR\s*(?:is|:))\b|बुकिंग\s*(?:हो गई|कन्फर्म)",
        prose,
        re.IGNORECASE
    ):
        return {"valid": False, "reason": "This app cannot confirm bookings or payments."}

    return {"valid": True, "reply": reply}
