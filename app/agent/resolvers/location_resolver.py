import re
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from ..session import ResolvedLocation

logger = logging.getLogger("location_resolver")


class CityDefinition:
    def __init__(
        self,
        code: str,
        name: str,
        aliases: List[str],
        station: Optional[str] = None,
        hindi_name: Optional[str] = None,
        has_airport: bool = True,
        nearest_airport_code: Optional[str] = None,
        nearest_airport_name: Optional[str] = None,
    ):
        self.code = code  # IATA code (or nearest airport code for flight searches)
        self.name = name  # Standard English name
        self.station = station  # Primary IR station code
        self.hindi_name = hindi_name
        self.aliases = aliases
        self.has_airport = has_airport
        self.nearest_airport_code = nearest_airport_code or code
        self.nearest_airport_name = nearest_airport_name or name


_DATA_PATH = Path(__file__).parent / "known_cities.json"


def _load_known_cities() -> List[CityDefinition]:
    if _DATA_PATH.exists():
        try:
            with open(_DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [CityDefinition(**row) for row in data]
        except Exception as e:
            logger.error(f"Failed to load known_cities.json: {e}")
    return []


KNOWN_CITIES: List[CityDefinition] = _load_known_cities()

# Build fast lookup indexes (airport cities take precedence for codes/aliases if shared)
_ALIAS_TO_CITY: Dict[str, CityDefinition] = {}
_CODE_TO_CITY: Dict[str, CityDefinition] = {}
_NAME_TO_CITY: Dict[str, CityDefinition] = {}

for _c in KNOWN_CITIES:
    if not _c.has_airport:
        _CODE_TO_CITY[_c.code.lower()] = _c
        _NAME_TO_CITY[_c.name.lower()] = _c
        for _a in _c.aliases:
            _ALIAS_TO_CITY[_a.lower()] = _c

for _c in KNOWN_CITIES:
    if _c.has_airport:
        _CODE_TO_CITY[_c.code.lower()] = _c
        _NAME_TO_CITY[_c.name.lower()] = _c
        for _a in _c.aliases:
            _ALIAS_TO_CITY[_a.lower()] = _c

# Precompile a single regex pattern across all unique aliases for sub-millisecond matching
_SORTED_ALIASES = sorted(_ALIAS_TO_CITY.keys(), key=lambda x: -len(x))
_ROUTE_PATTERN = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(a) for a in _SORTED_ALIASES) + r")(?![\w])",
    re.IGNORECASE,
) if _SORTED_ALIASES else None


KNOWN_REGIONS = [
    {"name": "Rajasthan", "aliases": ["rajasthan", "rajastan", "rajsthan", "राजस्थान"]},
    {"name": "Kerala", "aliases": ["kerala", "keralam", "केरल"]},
    {"name": "Himachal Pradesh", "aliases": ["himachal", "himachal pradesh", "hp", "हिमाचल", "हिमाचल प्रदेश"]},
    {"name": "Uttarakhand", "aliases": ["uttarakhand", "uttaranchal", "उत्तराखंड", "उत्तरांचल"]},
    {"name": "Kashmir", "aliases": ["kashmir", "jammu and kashmir", "j&k", "कश्मीर", "जम्मू और कश्मीर"]},
    {"name": "Gujarat", "aliases": ["gujarat", "gujrat", "गुजरात"]},
    {"name": "Punjab", "aliases": ["punjab", "पंजाब"]},
    {"name": "Maharashtra", "aliases": ["maharashtra", "महाराष्ट्र"]},
    {"name": "Karnataka", "aliases": ["karnataka", "कर्नाटक"]},
    {"name": "Tamil Nadu", "aliases": ["tamil nadu", "tamilnadu", "तमिलनाडु"]},
]


class LocationResolver:
    @staticmethod
    def resolve_city(raw: str) -> Optional[ResolvedLocation]:
        if not raw or not isinstance(raw, str):
            return None
        clean = raw.strip().lower()
        if len(clean) < 2:
            return None

        # 1. Direct alias match
        city = _ALIAS_TO_CITY.get(clean)
        # 2. Direct name match
        if not city:
            city = _NAME_TO_CITY.get(clean)
        # 3. Direct airport/station code match (e.g. DEL, BOM, NDLS)
        if not city:
            city = _CODE_TO_CITY.get(clean)

        if city:
            return ResolvedLocation(
                code=city.code,
                name=city.name,
                station=city.station,
                isExplicit=True,
                rawInput=raw,
                hasAirport=city.has_airport,
                nearestAirportCode=city.nearest_airport_code,
                nearestAirportName=city.nearest_airport_name,
            )
        return None

    @staticmethod
    def is_supported_location(raw: str) -> bool:
        """Deterministic check: True only if raw resolves to a known supported city/station/code."""
        return LocationResolver.resolve_city(raw) is not None

    @staticmethod
    def resolve_region(text: str) -> Optional[str]:
        if not text:
            return None
        clean = text.lower()
        for reg in KNOWN_REGIONS:
            for alias in reg["aliases"]:
                pattern = rf"(?<![\w]){re.escape(alias)}(?![\w])"
                if re.search(pattern, clean, re.IGNORECASE):
                    return reg["name"]
        return None

    @staticmethod
    def extract_route(text: str, mode: Optional[str] = None) -> Tuple[Optional[ResolvedLocation], Optional[ResolvedLocation], Optional[str]]:
        lower = text.lower()
        matches = []
        if _ROUTE_PATTERN:
            for m in _ROUTE_PATTERN.finditer(lower):
                alias = m.group(1).lower()
                city = _ALIAS_TO_CITY.get(alias)
                if city:
                    matches.append({"city": city, "start": m.start(), "end": m.end(), "alias": alias})

        matches.sort(key=lambda x: (x["start"], -len(x["alias"])))
        accepted = []
        for m in matches:
            if any(m["start"] < a["end"] and m["end"] > a["start"] for a in accepted):
                continue
            before = lower[max(0, m["start"] - 20):m["start"]]
            after = lower[m["end"]:m["end"] + 15]
            if re.search(r"(?:not|mat)\s*$", before) or re.search(r"^\s*(?:nahi|not|mat|नहीं)", after):
                continue
            accepted.append(m)

        origin_match = None
        for m in accepted:
            before = lower[:m["start"]]
            after = lower[m["end"]:]
            if re.search(r"(?:from|frm)\s*$", before) or re.search(r"^\s*(?:se\b|say\b|से|to\b|→|->)", after):
                origin_match = m
                break

        if not origin_match and len(accepted) == 2 and re.search(r"\b(flight|flights|train|trains|travel|business|economy)\b|jana|jaana|जाना|ट्रेन|फ्लाइट", lower) and not re.search(r"hotel|होटल", lower):
            origin_match = accepted[0]

        dest_match = None
        for m in accepted:
            if m == origin_match:
                continue
            before = lower[:m["start"]]
            after = lower[m["end"]:]
            if re.search(r"(?:to|towards|visit|in)\s*(?:the\s*)?$", before) or re.search(r"^\s*(?:jana|jaana|jane|jaane|ko|जाना|जाने|को)", after):
                dest_match = m
                break

        if not dest_match and origin_match:
            dest_candidates = [m for m in accepted if m != origin_match and m["start"] > origin_match["start"]]
            if dest_candidates:
                dest_match = dest_candidates[0]

        if not dest_match and not origin_match and accepted:
            dest_match = accepted[-1]

        def to_resolved(m) -> Optional[ResolvedLocation]:
            if not m:
                return None
            city = m["city"]
            return ResolvedLocation(
                code=city.code,
                name=city.name,
                station=city.station,
                isExplicit=True,
                rawInput=m["alias"],
                hasAirport=city.has_airport,
                nearestAirportCode=city.nearest_airport_code,
                nearestAirportName=city.nearest_airport_name,
            )

        origin = to_resolved(origin_match)
        destination = to_resolved(dest_match)
        region = LocationResolver.resolve_region(text) if not destination else None
        return origin, destination, region

    @staticmethod
    def get_flight_code(loc: Any) -> str:
        if not loc:
            return ""
        if isinstance(loc, str):
            code_str = loc.strip().upper()
            name_str = loc.strip().lower()
            nearest = None
        else:
            nearest = getattr(loc, "nearestAirportCode", None)
            code_str = (getattr(loc, "code", "") or "").strip().upper()
            name_str = (getattr(loc, "name", "") or "").strip().lower()
        if nearest:
            return nearest.upper()
        city = _CODE_TO_CITY.get(code_str.lower()) or _NAME_TO_CITY.get(name_str.lower())
        if city:
            return (city.nearest_airport_code or city.code).upper()
        return code_str or name_str.upper()

    @staticmethod
    def get_train_station(loc: Any) -> str:
        if not loc:
            return ""
        if isinstance(loc, str):
            code_str = loc.strip().upper()
            name_str = loc.strip().lower()
            stn = None
        else:
            stn = getattr(loc, "station", None)
            code_str = (getattr(loc, "code", "") or "").strip().upper()
            name_str = (getattr(loc, "name", "") or "").strip().lower()
        if stn:
            return stn.upper()
        city = _CODE_TO_CITY.get(code_str.lower()) or _NAME_TO_CITY.get(name_str.lower())
        if city and city.station:
            return city.station.upper()
        return code_str or name_str.upper()
