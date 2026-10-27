"""Canonical airline display names derived from supplier carrier evidence.

Air India (IATA AI) and Air India Express (IATA IX) are distinct brands. The display
name is derived from supplier-provided names first (operating carrier preferred over
marketing); an IATA code alone is never used to guess a brand. Raw supplier data is
never modified by callers; this only produces a display label.
"""
import re
from dataclasses import dataclass
from typing import Optional

AIR_INDIA = "Air India"
AIR_INDIA_EXPRESS = "Air India Express"

# Documented aliases of the Air India Express brand (IX).
_EXPRESS_RE = re.compile(r"^(air\s*india\s*express|ai\s*express|airindia\s*express|air\s*india\s*exp\.?)$", re.I)
_AIR_INDIA_RE = re.compile(r"^(air\s*india|airindia)$", re.I)
_CODE_BRAND = {"AI": AIR_INDIA, "IX": AIR_INDIA_EXPRESS}
_GENERIC = {"", "airline", "carrier", "unknown"}


@dataclass(frozen=True)
class CarrierDisplay:
    display: str
    ambiguous: bool = False  # supplier fields contradict each other or are missing


def _brand_from_name(name: Optional[str]) -> Optional[str]:
    n = re.sub(r"\s+", " ", (name or "").strip())
    if _EXPRESS_RE.match(n):
        return AIR_INDIA_EXPRESS
    if _AIR_INDIA_RE.match(n):
        return AIR_INDIA
    return None


def canonical_airline_name(
    name: Optional[str],
    code: Optional[str] = None,
    operating_name: Optional[str] = None,
    operating_code: Optional[str] = None,
) -> CarrierDisplay:
    """Display name for a carrier. Operating-carrier fields win when the supplier gives them."""
    if (operating_name or "").strip().lower() not in _GENERIC:
        name, code = operating_name, operating_code or code
    clean = re.sub(r"\s+", " ", (name or "").strip())
    code = (code or "").strip().upper()

    if clean.lower() in _GENERIC:
        # Never guess a brand from a bare code; surface the code itself.
        return CarrierDisplay(code or "Airline", ambiguous=True)

    brand = _brand_from_name(clean)
    if brand is None:
        return CarrierDisplay(clean)  # unrelated carrier: unchanged
    code_brand = _CODE_BRAND.get(code)
    # Supplier name is explicit, so it wins; flag a contradicting designator.
    return CarrierDisplay(brand, ambiguous=bool(code_brand and code_brand != brand))
