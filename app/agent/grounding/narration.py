import re
from typing import List, Dict, Any, Optional
from ...schemas.chat import ChatResults, NarrationSection
from ..session import TravelSession
from .grounded_reply import Facts, parse_reply_json, publish_reply


def build_narration(
    raw: str,
    facts: Optional[Facts] = None,
    results: Optional[ChatResults] = None,
    last_user: str = "",
    session: Optional[TravelSession] = None,
) -> List[NarrationSection]:
    try:
        facts = facts or {}
        results = results or ChatResults()
        obj = parse_reply_json(raw)
        full_text = (obj.get("reply") or "").strip()
        if not full_text:
            return []

        pieces = [p.strip() for p in re.split(r"\n+", full_text) if p.strip()]
        sections: List[NarrationSection] = []

        for piece in pieces:
            # Fact tokens [[kind.index.*]]
            matches = re.findall(r"\[\[(flight|hotel|train)\.(\d+)\.[^\]]+\]\]", piece)
            flights: List[str] = []
            hotels: List[str] = []
            trains: List[str] = []

            for kind, idx_str in matches:
                idx = int(idx_str) - 1
                if kind == "flight" and 0 <= idx < len(results.flights):
                    flights.append(results.flights[idx].offerId)
                elif kind == "train" and 0 <= idx < len(results.trains):
                    trains.append(results.trains[idx].trainNumber)
                elif kind == "hotel" and 0 <= idx < len(results.hotels):
                    hotels.append(results.hotels[idx].hotelId)

            # Mentioned train by number or name
            if not trains and results.trains:
                for t in results.trains:
                    if t.trainNumber and t.trainNumber in piece:
                        trains.append(t.trainNumber)
                    elif t.trainName and t.trainName.lower() in piece.lower():
                        trains.append(t.trainNumber)

            # Mentioned flight by airline
            if not flights and results.flights:
                for f in results.flights:
                    if f.outbound.airline and f.outbound.airline.lower() in piece.lower():
                        flights.append(f.offerId)

            # Mentioned hotel by name
            if not hotels and results.hotels:
                for h in results.hotels:
                    if h.name and h.name.lower() in piece.lower():
                        hotels.append(h.hotelId)

            text = publish_reply(piece, facts, last_user, session)
            sections.append(
                NarrationSection(
                    text=text,
                    flightOfferIds=list(dict.fromkeys(flights)),
                    hotelIds=list(dict.fromkeys(hotels)),
                    trainNumbers=list(dict.fromkeys(trains)),
                )
            )

        return sections
    except Exception:
        return []
