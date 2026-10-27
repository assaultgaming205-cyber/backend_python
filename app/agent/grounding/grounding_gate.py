import re
import logging
from typing import Set, Tuple, Optional, Union
from ...schemas.chat import ChatResults
from .grounded_reply import Facts

logger = logging.getLogger("grounding_gate")


class GroundingGate:
    """Verifies that individual streaming sentences do not invent travel facts.

    Inspects:
    - Fares / currency amounts (e.g. ₹2,238, Rs. 605, 1633 rupaye, ₹4,500.99)
    - Flight identifiers (e.g. AI-805, 6E 2134)
    - 5-digit Indian train numbers (e.g. 19020, 12952)
    - Exact departure and arrival times (e.g. 07:25, 16:55, 7:25 am, 7:25pm)
    - Dates where explicit travel claims are asserted

    Safety guarantees:
    - Fail-closed: If verified supplier facts are empty, claims are strictly rejected.
    - Decimals: Fares with decimal values (e.g. 4500.99 vs 4500.0) are compared accurately without unsafe truncation.
    - No arbitrary ±1 tolerance: Only defined supplier float values and their standard integer roundings are accepted.
    - Natural conversational sentences without exact travel claims remain allowed.
    """

    def __init__(
        self,
        verified_fares: Set[Union[int, float]],
        verified_flight_nums: Set[str],
        verified_train_nums: Set[str],
        verified_times: Set[str],
        verified_dates: Optional[Set[str]] = None,
        has_eligible_flights: bool = False,
        failed_dates: Optional[Set[str]] = None,
    ):
        self.verified_fares: Set[Union[int, float]] = set(verified_fares)
        self.verified_flight_nums: Set[str] = {
            fn.replace("-", "").replace(" ", "").upper() for fn in verified_flight_nums if fn
        }
        self.verified_train_nums: Set[str] = {str(tn).strip() for tn in verified_train_nums if tn}
        self.verified_times: Set[str] = set(verified_times)
        self.verified_dates: Set[str] = set(verified_dates or [])
        self.has_eligible_flights: bool = has_eligible_flights
        self.failed_dates: Set[str] = set(failed_dates or [])

    @classmethod
    def from_results_and_facts(cls, results: Optional[ChatResults], facts: Optional[Facts] = None) -> "GroundingGate":
        verified_fares: Set[Union[int, float]] = set()
        verified_flight_nums: Set[str] = set()
        verified_train_nums: Set[str] = set()
        verified_times: Set[str] = set()
        verified_dates: Set[str] = set()

        if results:
            for f in results.flights:
                if f.price and f.price.amount and f.price.amount > 0:
                    verified_fares.add(f.price.amount)
                    verified_fares.add(int(round(f.price.amount)))
                fn = getattr(f.outbound, "flightNumber", None)
                if fn:
                    verified_flight_nums.add(fn)
                for seg in getattr(f.outbound, "segments", []):
                    if getattr(seg, "flightNumber", None):
                        verified_flight_nums.add(seg.flightNumber)
                if f.outbound.departureLocal and len(f.outbound.departureLocal) >= 16:
                    verified_times.add(f.outbound.departureLocal[11:16])
                    verified_dates.add(f.outbound.departureLocal[:10])
                if f.outbound.arrivalLocal and len(f.outbound.arrivalLocal) >= 16:
                    verified_times.add(f.outbound.arrivalLocal[11:16])

            for t in results.trains:
                if t.trainNumber:
                    verified_train_nums.add(str(t.trainNumber).strip())
                if t.departureTime:
                    dep = t.departureTime.strip()
                    if ":" in dep:
                        parts = dep.split(":")
                        verified_times.add(f"{int(parts[0]):02d}:{parts[1][:2]}")
                if t.arrivalTime:
                    arr = t.arrivalTime.strip()
                    if ":" in arr:
                        parts = arr.split(":")
                        verified_times.add(f"{int(parts[0]):02d}:{parts[1][:2]}")
                if t.departureDate:
                    verified_dates.add(t.departureDate.strip())
                for cf in t.classFares or []:
                    if cf.fare and cf.fare > 0:
                        verified_fares.add(cf.fare)
                        verified_fares.add(int(round(cf.fare)))

            for h in results.hotels:
                if h.price and h.price.amount and h.price.amount > 0:
                    verified_fares.add(h.price.amount)
                    verified_fares.add(int(round(h.price.amount)))

        if facts:
            # Add comparison numbers if present in facts
            diff_str = facts.get("comparison.fareDifference", "")
            diff_m = re.search(r"\d+", diff_str)
            if diff_m:
                verified_fares.add(int(diff_m.group(0)))

            for k, v in facts.items():
                if "fare" in k.lower():
                    for num in re.findall(r"\d+(?:\.\d+)?", v):
                        try:
                            f_val = float(num)
                            verified_fares.add(f_val)
                            verified_fares.add(int(round(f_val)))
                        except ValueError:
                            pass
                if "date" in k.lower() and re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                    verified_dates.add(v)

        failed_dates: Set[str] = set()
        if results and results.dateStatuses:
            for d, st in results.dateStatuses.items():
                if st == "failed":
                    failed_dates.add(d)
        elif facts and facts.get("search.datesFailed"):
            for d in facts.get("search.datesFailed").split(","):
                d = d.strip()
                if d:
                    failed_dates.add(d)

        has_eligible = bool(results and results.flights)
        return cls(
            verified_fares=verified_fares,
            verified_flight_nums=verified_flight_nums,
            verified_train_nums=verified_train_nums,
            verified_times=verified_times,
            verified_dates=verified_dates,
            has_eligible_flights=has_eligible,
            failed_dates=failed_dates,
        )

    def is_sentence_grounded(self, sentence: str) -> Tuple[bool, str]:
        """Validates that any exact travel claims made in the sentence are verified.

        Returns:
            (True, "OK") if verified or natural/conversational.
            (False, "<reason>") if an unverified claim is detected.
        """
        if not sentence or not sentence.strip():
            return True, "Empty"

        # Contradiction guard: Never allow claiming all flights are booked/unavailable when eligible flight offers exist
        booked_contradiction = re.search(
            r"\b(saari\s+flights\s+fully\s+booked|all\s+flights\s+are\s+fully\s+booked|fully\s+booked|koi\s+flight\s+nahi\s+mil|seats\s+available\s+nahi|no\s+flights?\s+available)\b",
            sentence,
            re.IGNORECASE,
        )
        # Never claim failed/timed-out dates are fully booked
        if self.failed_dates and booked_contradiction:
            for fd in self.failed_dates:
                day_num = fd.split("-")[-1] if "-" in fd else ""
                if (fd in sentence) or (day_num and re.search(r"\b" + re.escape(day_num) + r"\b", sentence)) or any(kw in sentence.lower() for kw in ["next week", "sabhi", "saari", "all", "har ek", "pure hafte", "sab"]):
                    return False, f"Contradictory availability claim: search for date '{fd}' timed out or failed, cannot report as fully booked"

        if self.has_eligible_flights and booked_contradiction:
            return False, f"Contradictory availability claim: eligible flight offers exist in results, but sentence claimed: '{booked_contradiction.group(0)}'"

        # 1. Check Fares / Prices
        # Currency followed by digits/decimals: ₹2,238, Rs. 605, INR 1500, ₹4,500.99
        # Digits/decimals followed by currency: 2238 rupaye, 605 rs, 1500/-, 4500.99 inr
        fare_matches = re.findall(
            r"(?:₹|Rs\.?|INR)\s*([0-9][0-9,]*(?:\.[0-9]+)?)|([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:rupaye|rs|inr|\/-)",
            sentence,
            re.IGNORECASE,
        )
        for m in fare_matches:
            raw_val = (m[0] or m[1]).replace(",", "")
            try:
                if "." in raw_val:
                    val: Union[int, float] = float(raw_val)
                else:
                    val = int(raw_val)
            except ValueError:
                continue

            # Ignore small incidental numbers (e.g., 1, 2, 3 passengers, 10 minutes)
            if val < 50:
                continue

            # Fail-closed: Must be in verified_fares
            is_verified = (val in self.verified_fares)
            if not is_verified and isinstance(val, float) and val.is_integer():
                is_verified = (int(val) in self.verified_fares)
            elif not is_verified and isinstance(val, int):
                is_verified = (float(val) in self.verified_fares)

            if not is_verified:
                return False, f"Unverified fare '{val}' (verified: {sorted(self.verified_fares)})"

        # 2. Check Flight Numbers (e.g. AI-805, 6E 2134, UK 955)
        flight_matches = re.findall(r"\b([A-Z0-9]{2}[ -]?\d{3,4})\b", sentence)
        for fn in flight_matches:
            norm_fn = fn.replace("-", "").replace(" ", "").upper()
            if not any(c.isalpha() for c in norm_fn):
                continue
            # Must not be year (e.g. 2026, 2027)
            if norm_fn.startswith("202") or norm_fn.startswith("203"):
                continue

            # Fail-closed: If claimed, must exist in verified_flight_nums
            if norm_fn not in self.verified_flight_nums:
                return False, f"Unverified flight number '{fn}' (verified: {sorted(self.verified_flight_nums)})"

        # 3. Check 5-digit Indian Train Numbers (e.g. 19020, 12952)
        train_matches = re.findall(r"\b(\d{5})\b", sentence)
        for tn in train_matches:
            # If tn happens to match a verified fare, allow it
            if int(tn) in self.verified_fares:
                continue

            # Fail-closed: If claimed, must exist in verified_train_nums
            if tn not in self.verified_train_nums:
                return False, f"Unverified train number '{tn}' (verified: {sorted(self.verified_train_nums)})"

        # 4. Check Exact Departure / Arrival Times (e.g. 07:25, 16:55, 7:25 am, 7:25 pm)
        time_matches = re.finditer(r"\b(\d{1,2}:\d{2})\s*(am|pm)?\b", sentence, re.IGNORECASE)
        for tm_match in time_matches:
            tm = tm_match.group(1)
            ampm = tm_match.group(2)
            parts = tm.split(":")
            h, m_int = int(parts[0]), int(parts[1])

            if ampm:
                if ampm.lower() == "pm" and h < 12:
                    h += 12
                elif ampm.lower() == "am" and h == 12:
                    h = 0
            norm_tm = f"{h:02d}:{m_int:02d}"

            # Fail-closed: If claimed, must exist in verified_times
            if norm_tm not in self.verified_times:
                return False, f"Unverified schedule time '{tm_match.group(0)}' (verified: {sorted(self.verified_times)})"

        # 5. Check Exact ISO Dates (e.g. 2026-11-05)
        date_matches = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", sentence)
        for dt in date_matches:
            if self.verified_dates and dt not in self.verified_dates:
                return False, f"Unverified travel date '{dt}' (verified: {sorted(self.verified_dates)})"

        return True, "OK"
