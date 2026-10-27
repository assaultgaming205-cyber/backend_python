"""Clarification cards: backend-enforced normalization (max 3 options) and answer validation."""
import re
import uuid
from collections import OrderedDict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from .search_preflight import check_future_iso

MAX_OPTIONS = 3
MAX_CONVERSATIONS = 500


def normalize_clarification(raw: Dict[str, Any], turn_id: str) -> Optional[Dict[str, Any]]:
    """Validate model/agent supplied card data. Options: label and value are separate, <=3, unique."""
    question = str((raw or {}).get("question") or "").strip()
    field = str((raw or {}).get("field") or "").strip()
    if not question or not field:
        return None
    options: List[Dict[str, str]] = []
    seen = set()
    for opt in (raw.get("options") or []):
        if isinstance(opt, str):
            opt = {"label": opt, "value": opt}
        if not isinstance(opt, dict):
            continue
        value = str(opt.get("value") or "").strip()
        label = str(opt.get("label") or value).strip()
        if value and label and value not in seen:
            seen.add(value)
            options.append({"label": label[:60], "value": value[:80]})
        if len(options) == MAX_OPTIONS:
            break
    card: Dict[str, Any] = {
        "type": "clarification", "turnId": turn_id,
        "questionId": str(raw.get("questionId") or uuid.uuid4()),
        "field": field, "question": question[:200], "options": options, "allowCustom": True,
    }
    if raw.get("context"):
        card["context"] = str(raw["context"])[:200]
    return card


def date_options(today: date) -> List[Dict[str, str]]:
    """Real calendar dates (tomorrow, day after, upcoming Saturday), de-duplicated."""
    cands = [("Tomorrow", today + timedelta(days=1)), ("Day after tomorrow", today + timedelta(days=2))]
    sat = today + timedelta(days=(5 - today.weekday()) % 7 or 7)
    cands.append(("This weekend", sat))
    out, seen = [], set()
    for label, d in cands:
        if d not in seen:
            seen.add(d)
            out.append({"label": f"{label} — {d.day} {d.strftime('%b')}", "value": d.isoformat()})
    return out[:MAX_OPTIONS]


class ClarificationStore:
    """Per-conversation pending questions. Answers are validated; stale/duplicate ones are ignored."""

    def __init__(self) -> None:
        self._pending: "OrderedDict[str, Dict[str, Dict[str, Any]]]" = OrderedDict()

    def open(self, conversation_id: str, card: Dict[str, Any]) -> None:
        # One focused question at a time: a new card supersedes the previous one.
        self._pending[conversation_id] = {card["questionId"]: {"turnId": card["turnId"], "field": card["field"]}}
        self._pending.move_to_end(conversation_id)
        while len(self._pending) > MAX_CONVERSATIONS:
            self._pending.popitem(last=False)

    def cancel(self, conversation_id: str) -> None:
        self._pending.pop(conversation_id, None)

    def answer(self, conversation_id: str, question_id: str, turn_id: str, value: str,
               today: date) -> Dict[str, Any]:
        """Returns {"status": "ok", "field", "value"} or {"status": "stale"|"invalid"}."""
        q = self._pending.get(conversation_id, {}).get(question_id)
        if not q or q["turnId"] != turn_id:
            return {"status": "stale"}  # unknown, already answered, cancelled, or wrong turn
        value = (value or "").strip()
        if not value or len(value) > 80:
            return {"status": "invalid"}
        if q["field"].lower().endswith("date") and check_future_iso(value, today):
            return {"status": "invalid"}
        if q["field"] in ["origin", "destination"]:
            from .resolvers.location_resolver import LocationResolver
            if not LocationResolver.is_supported_location(value):
                return {"status": "invalid"}
        if re.search(r"[<>{}]", value):
            return {"status": "invalid"}
        del self._pending[conversation_id][question_id]  # a repeated click cannot trigger a second search
        return {"status": "ok", "field": q["field"], "value": value}


clarification_store = ClarificationStore()
