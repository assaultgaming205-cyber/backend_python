"""Airline-diverse selection, per-conversation shown/rejected ledger, and grounded comparisons.

All facts come from FlightOffer fields; missing values are reported as NOT_PROVIDED, never guessed.
"""
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Set

from ..schemas.flight import FlightOffer
from ..travel.airlines import canonical_airline_name

NOT_PROVIDED = "Not provided by supplier"
MAX_CONVERSATIONS = 500


def offer_date(o: FlightOffer) -> str:
    return (o.outbound.departureLocal or "")[:10]


def brand_of(o: FlightOffer) -> str:
    seg = o.outbound.segments[0] if o.outbound.segments else None
    return canonical_airline_name(
        (seg.carrierName if seg else None) or o.outbound.airline, seg.carrierCode if seg else None
    ).display


def offer_key(o: FlightOffer) -> str:
    """Deterministic key: date + provider + flight numbers + departure + cabin + fare.

    offerId is not used alone: BankG falls back to an index-based id when fareId is missing.
    """
    nums = ",".join(s.flightNumber or "" for s in o.outbound.segments)
    return "|".join([offer_date(o), o.provider, nums, o.outbound.departureLocal or "",
                     (o.cabinClass or "").lower(), f"{o.price.amount:.2f}{o.price.currency}"])


def _dedupe(offers: Iterable[FlightOffer]) -> List[FlightOffer]:
    seen: Set[str] = set()
    out = []
    for o in offers:
        k = offer_key(o)
        if k not in seen:
            seen.add(k)
            out.append(o)
    return out


def select_diverse(offers: List[FlightOffer], max_brands: int = 4, min_brands: int = 3):
    """Per date: cheapest offer per canonical brand, up to max_brands.

    Returns (selected, shortfall) where shortfall maps date -> distinct brands found when < min_brands.
    """
    by_date: Dict[str, List[FlightOffer]] = {}
    for o in _dedupe(offers):
        by_date.setdefault(offer_date(o), []).append(o)
    selected: List[FlightOffer] = []
    shortfall: Dict[str, int] = {}
    for d in sorted(by_date):
        best: Dict[str, FlightOffer] = {}
        for o in sorted(by_date[d], key=lambda x: x.price.amount):
            best.setdefault(brand_of(o), o)
        picks = sorted(best.values(), key=lambda x: x.price.amount)[:max_brands]
        if len(best) < min_brands:
            shortfall[d] = len(best)
        selected += picks
    return selected, shortfall


class FlightLedger:
    """Shown/rejected/selected offers for ONE conversation, plus its unseen pool."""

    def __init__(self) -> None:
        self.shown: Set[str] = set()
        self.rejected: Set[str] = set()
        self.selected: Set[str] = set()
        self.pool: "OrderedDict[str, FlightOffer]" = OrderedDict()

    def remember(self, offers: Iterable[FlightOffer]) -> None:
        for o in offers:
            self.pool.setdefault(offer_key(o), o)

    def mark_shown(self, offers: Iterable[FlightOffer]) -> None:
        self.shown.update(offer_key(o) for o in offers)

    def reject(self, offer: FlightOffer) -> None:
        self.rejected.add(offer_key(offer))

    def next_batch(self, per_date: int = 3, dates: Optional[List[str]] = None,
                   include_rejected: Iterable[str] = ()) -> List[FlightOffer]:
        """Next unseen, non-rejected offers per date from the pool (cheapest first, new brands first)."""
        allow = set(include_rejected)
        out: List[FlightOffer] = []
        for d in sorted({offer_date(o) for o in self.pool.values()}):
            if dates is not None and d not in dates:
                continue
            fresh = sorted(
                (o for k, o in self.pool.items()
                 if offer_date(o) == d and k not in self.shown and (k not in self.rejected or k in allow)),
                key=lambda x: x.price.amount)
            shown_brands = {brand_of(o) for o in self.pool.values() if offer_key(o) in self.shown}
            fresh.sort(key=lambda x: brand_of(x) in shown_brands)  # stable: unseen brands first
            out += fresh[:per_date]
        self.mark_shown(out)
        return out


def build_search_context_id(
    origin: Optional[str],
    destination: Optional[str],
    departure_date: Optional[str] = None,
    date_range_start: Optional[str] = None,
    date_range_end: Optional[str] = None,
    cabin: Optional[str] = "economy",
    adults: int = 1,
    dates: Optional[List[str]] = None,
) -> str:
    """Explicit identity representing effective parameters of a search context."""
    orig = (origin or "").upper().strip()
    dest = (destination or "").upper().strip()
    if dates and len(dates) > 1:
        dates_part = f"{min(dates)}..{max(dates)}"
    elif date_range_start and date_range_end:
        dates_part = f"{date_range_start}..{date_range_end}"
    else:
        dates_part = departure_date or ""
    c = (cabin or "economy").lower().strip()
    return f"{orig}:{dest}:{dates_part}:{c}:{max(1, adults)}"


_LEDGERS: "OrderedDict[str, FlightLedger]" = OrderedDict()


def get_ledger(conversation_id: str, context_id: Optional[str] = None) -> FlightLedger:
    """Isolated per conversation and search context; bounded LRU so state cannot grow without limit."""
    key = f"{conversation_id}:{context_id}" if context_id else conversation_id
    if key in _LEDGERS:
        _LEDGERS.move_to_end(key)
    else:
        _LEDGERS[key] = FlightLedger()
        while len(_LEDGERS) > MAX_CONVERSATIONS:
            _LEDGERS.popitem(last=False)
    return _LEDGERS[key]


def compare_flights(offers: List[FlightOffer]) -> Dict[str, Any]:
    """Structured comparison from supplier fields only. Prices are ranked only on a shared basis."""
    rows = []
    for o in offers:
        segs = o.outbound.segments
        obs = o.observedDetails or {}
        bag = o.bags
        rows.append({
            "offerId": o.offerId,
            "airline": brand_of(o),
            "flightNumbers": [s.flightNumber for s in segs if s.flightNumber] or NOT_PROVIDED,
            "date": offer_date(o) or NOT_PROVIDED,
            "from": o.outbound.origin, "to": o.outbound.destination,
            "departure": o.outbound.departureLocal, "arrival": o.outbound.arrivalLocal,
            "durationMinutes": o.outbound.durationMinutes if o.outbound.durationMinutes else NOT_PROVIDED,
            "stops": o.outbound.stops if o.outbound.stops is not None else NOT_PROVIDED,
            "layovers": [s.origin for s in segs[1:]] if len(segs) > 1 else [],
            "fare": f"{o.price.currency} {o.price.amount:g}",
            "priceBasis": o.priceBasis or NOT_PROVIDED,
            "cabinClass": o.cabinClass or NOT_PROVIDED,
            "checkedBags": (bag.checked if bag and bag.checked is not None else NOT_PROVIDED),
            "cabinBags": (bag.carryOn if bag and bag.carryOn is not None else NOT_PROVIDED),
            "fareBreakdown": obs.get("fareBreakdown") or NOT_PROVIDED,
            "refundPolicy": NOT_PROVIDED, "changePolicy": NOT_PROVIDED,  # BankG adapter exposes neither
            "provider": o.provider, "bookingUrl": o.sourceUrl or NOT_PROVIDED,
            "observedAt": o.observedAt or NOT_PROVIDED,
        })
    keys = {(r["priceBasis"], o.price.currency, o.cabinClass, offer_date(o)) for r, o in zip(rows, offers)}
    comparable = len(keys) <= 1
    cheapest = min(offers, key=lambda x: x.price.amount).offerId if offers and comparable else None
    return {"rows": rows, "priceComparable": comparable, "cheapestOfferId": cheapest,
            "note": None if comparable else
            "Fares differ in currency, cabin, date or basis, so they are not ranked by price."}


def parse_departure_minutes(dep_str: Optional[str]) -> Optional[int]:
    """Parse local departure time into minutes from midnight (0..1439).

    12:00 AM -> 0 (00:00)
    12:01 AM -> 1
    11:59 AM -> 719
    12:00 PM -> 720 (12:00)
    12:01 PM -> 721
    11:59 PM -> 1439 (23:59)
    Returns None if missing or invalid.
    """
    if not dep_str:
        return None
    s = dep_str.strip()
    if not s:
        return None

    import re
    # 1. 12-hour AM/PM time e.g. "12:00 AM", "06:30 PM"
    ampm_m = re.search(r"(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM)", s, re.IGNORECASE)
    if ampm_m:
        h = int(ampm_m.group(1)) % 12
        m = int(ampm_m.group(2))
        if ampm_m.group(3).upper() == "PM":
            h += 12
        return h * 60 + m

    # 2. ISO timestamp or 24-hour time: "2026-10-25T12:00:00" or "12:00"
    time_part = s.split("T")[-1] if "T" in s else (s.split()[-1] if " " in s else s)
    h24_m = re.match(r"^(\d{1,2}):(\d{2})", time_part)
    if h24_m:
        h = int(h24_m.group(1))
        m = int(h24_m.group(2))
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m

    return None


def sort_flight_offers(offers: List[FlightOffer], mode: str = "price_asc") -> List[FlightOffer]:
    """Sort flight offers numerically by price or chronologically by departure time.

    Modes:
    - "price_asc": Low to High
    - "price_desc": High to Low
    - "departure_asc": Earliest First (00:00 through 23:59)
    - "departure_desc": Latest First (23:59 through 00:00)

    Guarantees:
    - Deduplicates identical offers.
    - Preserves multi-date grouping (chronological by date, sorted within each date).
    - Preserves incompatible currencies without falsely comparing amounts.
    - Missing or invalid values (prices <= 0 or missing times) are placed AFTER valid comparable values in BOTH ascending and descending order.
    - Stable tie-breaking using offerId.
    """
    deduped = _dedupe(offers)
    if not deduped:
        return []

    # Group by date and currency so comparisons are valid
    by_group: Dict[tuple, List[FlightOffer]] = {}
    for o in deduped:
        d = offer_date(o)
        cur = (o.price.currency if o.price else "INR") or "INR"
        basis = o.priceBasis or "per_person"
        # In multi-date, group primarily by date
        grp = (d, cur, basis) if mode.startswith("price") else (d,)
        by_group.setdefault(grp, []).append(o)

    sorted_result: List[FlightOffer] = []
    # Sort groups chronologically by date
    for grp in sorted(by_group.keys(), key=lambda g: (1 if not g[0] else 0, g)):
        group_offers = by_group[grp]
        if mode == "price_asc":
            valid = [o for o in group_offers if o.price and o.price.amount is not None and o.price.amount > 0]
            invalid = [o for o in group_offers if not (o.price and o.price.amount is not None and o.price.amount > 0)]
            valid.sort(key=lambda o: (float(o.price.amount), o.outbound.departureLocal or "", o.offerId))
            invalid.sort(key=lambda o: (o.outbound.departureLocal or "", o.offerId))
            sorted_result.extend(valid + invalid)
        elif mode == "price_desc":
            valid = [o for o in group_offers if o.price and o.price.amount is not None and o.price.amount > 0]
            invalid = [o for o in group_offers if not (o.price and o.price.amount is not None and o.price.amount > 0)]
            valid.sort(key=lambda o: (-float(o.price.amount), o.outbound.departureLocal or "", o.offerId))
            invalid.sort(key=lambda o: (o.outbound.departureLocal or "", o.offerId))
            sorted_result.extend(valid + invalid)
        elif mode == "departure_asc":
            valid_with_time = []
            invalid = []
            for o in group_offers:
                mins = parse_departure_minutes(o.outbound.departureLocal)
                if mins is not None:
                    valid_with_time.append((mins, o))
                else:
                    invalid.append(o)
            valid_with_time.sort(key=lambda x: (x[0], float(x[1].price.amount) if x[1].price and x[1].price.amount else float("inf"), x[1].offerId))
            invalid.sort(key=lambda o: (float(o.price.amount) if o.price and o.price.amount else float("inf"), o.offerId))
            sorted_result.extend([x[1] for x in valid_with_time] + invalid)
        elif mode == "departure_desc":
            valid_with_time = []
            invalid = []
            for o in group_offers:
                mins = parse_departure_minutes(o.outbound.departureLocal)
                if mins is not None:
                    valid_with_time.append((mins, o))
                else:
                    invalid.append(o)
            valid_with_time.sort(key=lambda x: (-x[0], float(x[1].price.amount) if x[1].price and x[1].price.amount else float("inf"), x[1].offerId))
            invalid.sort(key=lambda o: (float(o.price.amount) if o.price and o.price.amount else float("inf"), o.offerId))
            sorted_result.extend([x[1] for x in valid_with_time] + invalid)
        else:
            sorted_result.extend(group_offers)

    return sorted_result
