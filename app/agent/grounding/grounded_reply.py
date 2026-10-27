import re
import json
from typing import Dict, Any, Optional
from ...schemas.chat import ChatResults
from ..session import TravelSession

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "minLength": 1, "maxLength": 1800}
    },
    "required": ["reply"],
    "additionalProperties": False,
}

Facts = Dict[str, str]

# Chain-of-thought / prompt-echo markers; shortcut: phrase list, upgrade if new leak styles appear.
_LEAK = re.compile(
    r"\b(the user (?:wants|said|sent|asked|specified)|system prompt|server search outcome|I should (?:respond|ask|say)|the instruction says)\b",
    re.IGNORECASE,
)


def build_facts(s: TravelSession, r: ChatResults) -> Facts:
    f: Facts = {}

    def put(key: str, value: Any):
        if value is not None and value != "" and not isinstance(value, (dict, list)):
            f[key] = str(value)

    if s.origin:
        put("trip.origin", s.origin.name)
    if s.destination:
        put("trip.destination", s.destination.name)
        if not s.destination.hasAirport:
            put("trip.destinationHasNoAirport", "true")
            put("trip.destinationNearestAirport", s.destination.nearestAirportName or s.destination.code)
            put("trip.destinationNearestAirportCode", s.destination.nearestAirportCode or s.destination.code)
    put("trip.region", s.destinationRegion)
    put("trip.date", s.dates.departureDate)
    put("trip.return", s.dates.returnDate)
    put("trip.budget", s.constraints.maxBudget)
    put("trip.adults", s.passengers.adults)
    put("trip.mode", s.mode)
    put("mode", s.mode)
    put("passengers", s.passengers.adults)

    # 1. Flights
    for i, o in enumerate(r.flights[:8]):
        k = f"flight.{i+1}"
        put(f"{k}.airline", o.outbound.airline)
        put(f"{k}.cabin", o.cabinClass.replace("_", " ") if o.cabinClass else "economy")
        if o.price.amount > 0:
            cur = "₹" if o.price.currency == "INR" else f"{o.price.currency} "
            put(f"{k}.fare", f"{o.price.currency} {o.price.amount}")
            put(f"{k}.fareWithCurrency", f"{cur}{o.price.amount}")
            put(f"{k}.fareInr", f"{cur}{o.price.amount}")
            put(f"{k}.fareAmount", f"{o.price.amount}")
        put(f"{k}.priceBasis", o.priceBasis or "unknown")
        put(f"{k}.provider", o.provider)
        put(f"{k}.date", o.outbound.departureLocal[:10] if o.outbound.departureLocal else None)
        put(f"{k}.departure", o.outbound.departureLocal)
        put(f"{k}.arrival", o.outbound.arrivalLocal)
        if o.outbound.durationMinutes is not None:
            put(f"{k}.duration", f"{o.outbound.durationMinutes} minutes")
        put(f"{k}.stops", o.outbound.stops)

    # 2. Trains
    for i, t in enumerate(r.trains[:5]):
        k = f"train.{i+1}"
        put(f"{k}.name", t.trainName)
        put(f"{k}.number", t.trainNumber)
        put(f"{k}.departure", t.departureTime)
        put(f"{k}.arrival", t.arrivalTime)
        put(f"{k}.date", t.departureDate)
        if t.durationMinutes is not None:
            put(f"{k}.duration", f"{t.durationMinutes} minutes")
        if t.classes:
            put(f"{k}.classes", ", ".join(t.classes))
        for cf in t.classFares or []:
            put(f"{k}.fare.{cf.classCode}", f"{cf.fare}")
            put(f"{k}.fareWithCurrency.{cf.classCode}", f"₹{cf.fare}")
            put(f"{k}.fareInr.{cf.classCode}", f"INR {cf.fare}")
            if cf.status:
                put(f"{k}.status.{cf.classCode}", cf.status)

    # 3. Hotels
    for i, h in enumerate(r.hotels[:5]):
        k = f"hotel.{i+1}"
        put(f"{k}.name", h.name)
        put(f"{k}.stars", h.stars)
        if h.rating:
            put(f"{k}.rating", h.rating.score)
        put(f"{k}.address", h.address)
        put(f"{k}.room", h.roomName)
        if h.price:
            put(f"{k}.fare", f"{h.price.currency} {h.price.amount}")
            put(f"{k}.fareWithCurrency", f"₹{h.price.amount}")
        put(f"{k}.priceBasis", h.priceBasis or "unknown")

    # 4. Comparison (Flight vs Train)
    if r.flights and r.trains:
        min_train_fare = float("inf")
        min_train = None
        min_train_class = ""
        for t in r.trains:
            for cf in t.classFares or []:
                if cf.fare and cf.fare > 0 and cf.fare < min_train_fare:
                    min_train_fare = cf.fare
                    min_train = t
                    min_train_class = cf.classCode

        sorted_flights = sorted(r.flights, key=lambda x: x.price.amount if x.price else float("inf"))
        min_flight = sorted_flights[0] if sorted_flights else None

        if min_train and min_flight and min_train_fare < float("inf") and min_flight.price and min_flight.price.amount:
            cheaper_mode = "train" if min_train_fare < min_flight.price.amount else "flight"
            diff = int(abs(min_flight.price.amount - min_train_fare))
            put("comparison.cheaperMode", cheaper_mode)
            put("comparison.cheapestTrain.name", min_train.trainName)
            put("comparison.cheapestTrain.number", min_train.trainNumber)
            put("comparison.cheapestTrain.fare", f"₹{int(min_train_fare)}")
            put("comparison.cheapestTrain.class", min_train_class)
            put("comparison.cheapestFlight.airline", min_flight.outbound.airline)
            put("comparison.cheapestFlight.fare", f"₹{int(min_flight.price.amount)}")
            put("comparison.fareDifference", f"₹{diff}")

    # 5. Cabin classes comparison (for 'sabki batao')
    cabins_found = {}
    for o in r.flights:
        c = (o.cabinClass or "economy").lower()
        if c not in cabins_found or (o.price and o.price.amount < cabins_found[c].price.amount):
            cabins_found[c] = o

    if len(cabins_found) > 1 or s.compareAllCabins or s.cabinClass == "all":
        put("cabins.count", len(cabins_found))
        for c, o in cabins_found.items():
            c_label = c.replace("_", " ").title()
            put(f"cabin.{c}.name", c_label)
            put(f"cabin.{c}.airline", o.outbound.airline)
            put(f"cabin.{c}.departure", o.outbound.departureLocal)
            put(f"cabin.{c}.fare", f"₹{o.price.amount}")
            if o.outbound.durationMinutes:
                put(f"cabin.{c}.duration", f"{o.outbound.durationMinutes} min")

    return f


def clean_reply_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    cleaned = text
    cleaned = re.sub(r"^\{[\s\S]*?\"reply\"\s*:\s*\"", "", cleaned)
    cleaned = re.sub(r"\"\s*\}$", "", cleaned)
    cleaned = re.sub(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ", cleaned)
    cleaned = re.sub(r"[\u2010-\u2015]", "-", cleaned)
    return cleaned.strip()


def parse_reply_json(raw: str) -> Dict[str, str]:
    if not raw or not isinstance(raw, str):
        return {"reply": ""}

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            if "reply" in obj and isinstance(obj["reply"], str):
                return {"reply": clean_reply_text(obj["reply"])}
            text_val = obj.get("message") or obj.get("response") or obj.get("answer") or obj.get("content") or obj.get("text")
            if isinstance(text_val, str):
                return {"reply": clean_reply_text(text_val)}
    except Exception:
        m = re.search(r'\{[\s\S]*?"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
        if m:
            try:
                val = json.loads(f'"{m.group(1)}"')
            except Exception:
                val = m.group(1)
            return {"reply": clean_reply_text(val)}

        trunc = re.search(r'\{[\s\S]*?"reply"\s*:\s*"((?:[^"\\]|\\.)*)$', cleaned)
        if trunc:
            return {"reply": clean_reply_text(re.sub(r"\\+$", "", trunc.group(1)))}

    return {"reply": clean_reply_text(cleaned)}


def publish_reply(raw: str, facts: Optional[Facts] = None, last_user: str = "", session: Optional[TravelSession] = None) -> str:
    facts = facts or {}
    obj = parse_reply_json(raw)
    reply = obj.get("reply", "").strip()
    if not reply or _LEAK.search(reply):
        return "Main aapki travel planning mein madad kar sakti hoon. Kahan jana chahte hain?"

    # Replace [[fact_key]] placeholders
    def replace_token(match):
        key = match.group(1)
        return facts.get(key, "")

    published = re.sub(r"\[\[([a-zA-Z0-9_.]+)\]\]", replace_token, reply)
    return published.strip()
