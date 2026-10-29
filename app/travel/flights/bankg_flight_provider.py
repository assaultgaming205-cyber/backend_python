import re
import time
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import httpx

from ...schemas.flight import FlightOffer, FlightLeg, FlightSegment, FlightBags, Money
from ...agent.resolvers.location_resolver import LocationResolver
from ...config import settings
from ..airlines import canonical_airline_name
from .bankg_crypto import generate_bankg_xchecksum

logger = logging.getLogger("bankg_flight_provider")

GATEWAY_URL = "https://bankg-stagging-api.testuat.com/travel/v1/api/travel-master/airSearch"

AIRLINE_BOOKING_URLS = {
    "AI": "https://www.airindia.com",
    "6E": "https://www.goindigo.in",
    "SG": "https://www.spicejet.com",
    "IX": "https://www.airindiaexpress.com",
    "QP": "https://www.akasaair.com",
    "UK": "https://www.airindia.com",
}


class OperationDisabledError(RuntimeError):
    """Raised when an operation is prohibited by security policy or feature flags."""
    pass


def _resolve_iata(city_or_code: str) -> str:
    """Resolve city name or code to 3-letter IATA code. Never defaults or guesses."""
    if not city_or_code or not str(city_or_code).strip():
        raise ValueError("Travel route origin and destination are required.")
    clean = city_or_code.strip()
    if len(clean) == 3 and clean.isalpha():
        return clean.upper()
    resolved = LocationResolver.resolve_city(clean)
    if resolved and resolved.code and len(resolved.code) == 3 and resolved.code.isalpha():
        return resolved.code.upper()
    raise ValueError(f"Unresolvable location: '{city_or_code}'. Must be a recognized city or valid 3-letter IATA airport code.")


def _normalize_date(departure_date: Optional[str]) -> str:
    """Ensure departure date is a valid calendar date in YYYY-MM-DD."""
    if not departure_date:
        raise ValueError("Travel date is required and must be in YYYY-MM-DD format.")
    raw = departure_date.split("T")[0].strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if match:
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        datetime(y, m, d)
        return raw
    # Check DD-MM-YYYY or DD/MM/YYYY
    match_rev = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", raw)
    if match_rev:
        d, m, y = int(match_rev.group(1)), int(match_rev.group(2)), int(match_rev.group(3))
        datetime(y, m, d)
        return f"{y:04d}-{m:02d}-{d:02d}"
    raise ValueError(f"Invalid travel date '{departure_date}'. Must be a valid date in YYYY-MM-DD format.")



def _normalize_cabin(cabin_class: Optional[str]) -> str:
    """Map cabin class strictly to BankG Gateway v3.2 supported enums: ECONOMY, BUSINESS, FIRST."""
    c = (cabin_class or "economy").lower().strip()
    if "premium" in c or c in ("premium_economy", "premium economy", "pe"):
        raise ValueError("Documented cabinClass values are ECONOMY, BUSINESS, FIRST. Premium Economy is not supported by BankG gateway v3.2.")
    if "business" in c:
        return "BUSINESS"
    if "first" in c:
        return "FIRST"
    if "economy" in c:
        return "ECONOMY"
    raise ValueError(f"Unsupported cabinClass: '{cabin_class}'. Documented values are ECONOMY, BUSINESS, FIRST.")


def _unwrap_bankg_envelope(data: Any) -> tuple[str, str, Dict[str, Any]]:
    """
    Unwraps BankG Gateway v3.2 response payloads.
    Live staging responses wrap the payload in {"options": {"responseCode": "0", "description": "SUCCESS", "data": {...}}},
    while PDF sample snippets show top-level {"responseCode": "0", "description": "SUCCESS", "data": {...}}.
    """
    if not isinstance(data, dict):
        return "", "", {}
    opts = data.get("options") if isinstance(data.get("options"), dict) else {}
    res_code = str(data["responseCode"] if "responseCode" in data else opts.get("responseCode", ""))
    desc = str(data.get("description") or opts.get("description") or "").strip()
    inner = data["data"] if "data" in data else opts.get("data", {})
    return res_code, desc, inner if isinstance(inner, dict) else {}


def _clean_supplier_id(raw_id: str, field_name: str) -> str:
    """Validate and normalize a supplier fareId/priceId, stripping any internal 'bankg_' offerId prefix."""
    if not raw_id or not str(raw_id).strip():
        raise ValueError(f"{field_name} is required")
    val = str(raw_id).strip()
    if val.startswith("bankg_"):
        val = val[len("bankg_"):]
    if not val or re.fullmatch(r"\d+_\d+", val):
        raise ValueError(f"Valid supplier-returned {field_name} is required; synthetic fallback IDs cannot be used.")
    return val


_METRO_AIRPORT_GROUPS = [
    {"BOM", "NMI"},
    {"DEL", "HDO", "DXN"},
    {"GOI", "GOX"},
]


def _are_co_terminal_airports(code_a: str, code_b: str) -> bool:
    if not code_a or not code_b:
        return False
    a, b = code_a.strip().upper(), code_b.strip().upper()
    if a == b:
        return True
    return any(a in grp and b in grp for grp in _METRO_AIRPORT_GROUPS)


def _extract_fare_route_flight_token(identifier: str) -> str:
    """
    Extracts the middle route+flight token from BankG fareId / priceId strings,
    stripping the leading journey index (e.g., '13' in airSearch vs '0' in flightReview):
    e.g. '22-9151590572_13BOMIXBIX1176IXBDELIX1080~5953792485673275' -> 'BOMIXBIX1176IXBDELIX1080'
         'R22-9151590572_0BOMIXBIX1176IXBDELIX1080~5953821025123619' -> 'BOMIXBIX1176IXBDELIX1080'
    """
    if not identifier or "_" not in identifier:
        return ""
    after_underscore = identifier.split("_", 1)[1]
    raw_token = after_underscore.split("~", 1)[0].strip().upper()
    return re.sub(r"^\d+", "", raw_token)


def _verify_fare_id_correlation(requested_fare_id: str, returned_price_id: str, returned_req_id: str = "") -> bool:
    """
    Verifies that the flightReview response priceId/reqId corresponds to the requested fareId.
    - If both have structured BankG route+flight tokens (`_<token>~`), their tokens must match.
    - If returned_req_id is present and requested_fare_id has a `<prefix>_` token, the core numeric reqId must match.
    """
    req_clean = (requested_fare_id or "").strip()
    price_clean = (returned_price_id or "").strip()
    if not req_clean or not price_clean:
        return False

    req_token = _extract_fare_route_flight_token(req_clean)
    price_token = _extract_fare_route_flight_token(price_clean)
    if req_token and price_token and req_token != price_token:
        return False

    if "_" in req_clean and "_" in price_clean:
        req_prefix = req_clean.split("_", 1)[0].lstrip("R")
        price_prefix = price_clean.split("_", 1)[0].lstrip("R")
        if req_prefix and price_prefix and req_prefix != price_prefix:
            return False

    return True


class BankGFlightProvider:
    def __init__(self, timeout: float = 45.0):
        self.timeout = timeout

    def _build_headers(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        merchant_key = getattr(settings, "BANKG_MERCHANT_KEY", "") or ""
        if merchant_key and payload is not None:
            xchecksum = generate_bankg_xchecksum(payload, merchant_key)
        else:
            xchecksum = getattr(settings, "BANKG_CHECKSUM", "DUMMY_ENCRYPTED_CHECKSUM_SAMPLE")
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "xchecksum": xchecksum,
        }

    async def check_health(self) -> Dict[str, Any]:
        """
        Gateway v3.2 Section 1: GET /health liveness probe.
        Note: On the BankG UAT staging server, GET /travel/v1/api/health returns HTTP 404
        ("Cannot GET /travel/v1/api/health") even when POST /travel-master/airSearch is live.
        Therefore, a 404 on /health proves that the gateway host is reachable over HTTPS while the
        /health route is unmounted, and must NEVER be treated as a flight-search failure.
        """
        url = f"{self.base_url}/health"
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0, read=10.0)) as client:
                resp = await client.get(url, headers={"Accept": "application/json"})
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
            return {
                "url": url,
                "httpStatus": None,
                "elapsedMs": elapsed_ms,
                "gatewayReachable": False,
                "healthRouteMounted": False,
                "healthy": False,
                "status": "UNREACHABLE",
                "error": str(exc),
            }
        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}
        res_code, desc, inner = _unwrap_bankg_envelope(body)
        is_up = resp.status_code in (200, 201) and (
            (isinstance(body, dict) and str(body.get("status", "")).upper() == "UP") or res_code == "0"
        )
        route_mounted = resp.status_code != 404
        return {
            "url": url,
            "httpStatus": resp.status_code,
            "elapsedMs": elapsed_ms,
            "gatewayReachable": resp.status_code < 500,
            "healthRouteMounted": route_mounted,
            "healthy": is_up,
            "status": "UP" if is_up else ("HEALTH_ROUTE_UNMOUNTED" if resp.status_code == 404 else f"HTTP_{resp.status_code}"),
            "responseCode": res_code,
            "description": desc,
            "response": body,
        }

    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1,
        cabin_class: str = "economy",
        **kwargs,
    ) -> List[FlightOffer]:
        orig_iata = _resolve_iata(origin)
        dest_iata = _resolve_iata(destination)

        if not re.match(r"^[A-Z]{3}$", orig_iata):
            raise ValueError(f"Origin must be a 3-letter IATA code, got '{origin}' -> '{orig_iata}'")
        if not re.match(r"^[A-Z]{3}$", dest_iata):
            raise ValueError(f"Destination must be a 3-letter IATA code, got '{destination}' -> '{dest_iata}'")

        valid_date = _normalize_date(departure_date)
        try:
            datetime.strptime(valid_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format '{valid_date}'. Must be YYYY-MM-DD.")

        from ...agent.search_preflight import today_ist
        today = today_ist(tz=getattr(settings, "APP_TIMEZONE", "Asia/Kolkata"))
        if valid_date < today.isoformat():
            logger.warning(f"BankG search skipped: {valid_date} is in the past (today is {today.isoformat()})")
            return []

        api_cabin = _normalize_cabin(cabin_class)

        if not isinstance(adults, int) or adults < 1 or adults > 9:
            raise ValueError(f"adults count must be an integer between 1 and 9 inclusive, got {adults}")

        child = kwargs.get("child", 0)
        if not isinstance(child, int) or child < 0 or child > 8:
            raise ValueError(f"child count must be an integer between 0 and 8 inclusive, got {child}")

        infant = kwargs.get("infant", 0)
        if not isinstance(infant, int) or infant < 0 or infant > 4:
            raise ValueError(f"infant count must be an integer between 0 and 4 inclusive, got {infant}")

        segments_payload = [
            {
                "origin": orig_iata,
                "destination": dest_iata,
                "date": valid_date,
            }
        ]

        search_modifiers = {
            "isDirectFlight": bool(kwargs.get("isDirectFlight", True)),
            "isConnectingFlight": bool(kwargs.get("isConnectingFlight", True)),
            "pft": str(kwargs.get("pft", "REGULAR")),
        }

        payload = {
            "cabinClass": api_cabin,
            "adults": adults,
            "child": child,
            "infant": infant,
            "segments": segments_payload,
            "searchModifiers": search_modifiers,
        }

        gateway_url = getattr(settings, "BANKG_GATEWAY_URL", GATEWAY_URL)
        headers = self._build_headers(payload)

        start_mono = time.monotonic()
        reached_supplier = False
        status_code = None
        usable_response = False
        sanitized_error = None
        timeout_config = httpx.Timeout(timeout=self.timeout, connect=5.0, read=self.timeout)

        logger.info(f"Querying BankG Flight Gateway: {orig_iata} -> {dest_iata} on {valid_date} ({api_cabin})")

        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                resp = await client.post(
                    gateway_url,
                    json=payload,
                    headers=headers,
                )
            reached_supplier = True
            status_code = resp.status_code

            if resp.status_code in (401, 403):
                sanitized_error = f"HTTP_{resp.status_code}_AUTH_FAILURE"
                raise RuntimeError(f"BankG authentication error: HTTP {resp.status_code}")
            elif resp.status_code == 429:
                sanitized_error = "HTTP_429_RATE_LIMIT"
                raise RuntimeError("BankG rate limit exceeded: HTTP 429")
            elif resp.status_code >= 500:
                sanitized_error = f"HTTP_{resp.status_code}_SERVER_ERROR"
                raise RuntimeError(f"BankG upstream server error: HTTP {resp.status_code}")
            elif resp.status_code not in (200, 201):
                sanitized_error = f"HTTP_{resp.status_code}"
                raise RuntimeError(f"BankG unexpected HTTP status {resp.status_code}")

            try:
                data = resp.json()
            except Exception as parse_err:
                sanitized_error = "INVALID_JSON_RESPONSE"
                raise RuntimeError(f"BankG returned invalid JSON: {parse_err}")

            response_code, description, journeys_data = _unwrap_bankg_envelope(data)
            journeys = journeys_data.get("journeys", []) if isinstance(journeys_data, dict) else []

            if response_code != "0":
                desc_lower = (description or "").lower()
                if "no flight" in desc_lower or "no result" in desc_lower or "not found" in desc_lower or "no data found" in desc_lower:
                    usable_response = True
                    logger.info(
                        f"BankG returned valid zero-flight result for {orig_iata}->{dest_iata} ({api_cabin}): {description}"
                    )
                    return []
                sanitized_error = f"PROVIDER_ERROR_{response_code}"
                raise RuntimeError(f"BankG provider error: {description or response_code}")

            usable_response = True
            logger.info(f"BankG returned {len(journeys)} flight journeys for {orig_iata}->{dest_iata}")

            offers: List[FlightOffer] = []
            include_alternate_airports = bool(kwargs.get("include_alternate_airports", False))
            excluded_route_mismatches = 0
            for i, journey in enumerate(journeys):
                flights_raw = journey.get("flights", [])
                fares_raw = journey.get("fares", [])
                if not flights_raw or not fares_raw:
                    continue

                # Flight segments: preserve exact supplier airport codes and names; never default to requested IATA
                segments: List[FlightSegment] = []
                raw_airline_names: List[Optional[str]] = []
                total_duration = 0
                segment_valid = True
                for f in flights_raw:
                    airline_dict = f.get("airline", {})
                    carrier_code = airline_dict.get("code", "")
                    raw_airline_names.append(airline_dict.get("name"))
                    airline_name = canonical_airline_name(airline_dict.get("name"), carrier_code).display
                    flight_num = f"{carrier_code} {f.get('flightNumber', '')}".strip()
                    aircraft = f.get("aircraftType")
                    dur = f.get("durationMinutes") or 0
                    total_duration += dur

                    dep_info = f.get("departure", {}) or {}
                    arr_info = f.get("arrival", {}) or {}
                    dep_code = str(dep_info.get("airportCode") or "").strip().upper()
                    arr_code = str(arr_info.get("airportCode") or "").strip().upper()
                    if not dep_code or not arr_code:
                        segment_valid = False
                        break

                    segments.append(
                        FlightSegment(
                            carrierCode=carrier_code,
                            carrierName=airline_name,
                            flightNumber=flight_num,
                            origin=dep_code,
                            destination=arr_code,
                            departureLocal=dep_info.get("dateTime") or f"{valid_date}T00:00:00",
                            arrivalLocal=arr_info.get("dateTime") or f"{valid_date}T00:00:00",
                            durationMinutes=dur,
                            aircraft=aircraft,
                        )
                    )

                if not segment_valid or not segments:
                    continue

                seg0 = segments[0]
                seg_last = segments[-1]
                exact_origin_match = (seg0.origin == orig_iata)
                exact_dest_match = (seg_last.destination == dest_iata)
                route_matched = exact_origin_match and exact_dest_match
                is_alternate_airport = False

                if not route_matched:
                    co_terminal_orig = _are_co_terminal_airports(seg0.origin, orig_iata)
                    co_terminal_dest = _are_co_terminal_airports(seg_last.destination, dest_iata)
                    if include_alternate_airports and co_terminal_orig and co_terminal_dest:
                        is_alternate_airport = True
                    else:
                        excluded_route_mismatches += 1
                        continue

                stops = max(0, len(segments) - 1)
                primary_carrier = seg0.carrierCode or "AI"
                airline_name = seg0.carrierName or "Airline"
                airline_logo = f"https://images.kiwi.com/airlines/64x64/{primary_carrier}.png"

                booking_url = AIRLINE_BOOKING_URLS.get(
                    primary_carrier,
                    f"https://www.google.com/travel/flights?q=flights+from+{seg0.origin}+to+{seg_last.destination}+on+{valid_date}",
                )

                dep_first = flights_raw[0].get("departure", {}) or {}
                arr_final = flights_raw[-1].get("arrival", {}) or {}
                dep_terminal = dep_first.get("terminal")
                arr_terminal = arr_final.get("terminal")
                orig_airport_name = dep_first.get("airportName") or seg0.origin
                orig_city_name = dep_first.get("city") or orig_airport_name
                dest_airport_name = arr_final.get("airportName") or seg_last.destination
                dest_city_name = arr_final.get("city") or dest_airport_name

                # Iterate through distinct fare options for this journey
                for j, fare in enumerate(fares_raw):
                    price_info = fare.get("price", {})
                    total_fare = float(price_info.get("totalFare", 0))
                    if total_fare <= 0:
                        continue

                    base_fare = float(price_info.get("baseFare", 0))
                    tax = float(price_info.get("tax", 0))
                    currency = price_info.get("currency", "INR")

                    baggage_info = fare.get("baggage", {})
                    check_in_str = baggage_info.get("checkIn") or ""
                    cabin_bag_str = baggage_info.get("cabin") or ""

                    raw_fare_id = fare.get("fareId")
                    fare_id = raw_fare_id or f"bankg_{i}_{j}"
                    offer_id = f"bankg_{fare_id}"

                    raw_cabin = str(
                        fare.get("cabinClass")
                        or fare.get("cabin")
                        or (flights_raw[0].get("cabinClass") if flights_raw else "")
                        or ""
                    ).lower()

                    if "first" in raw_cabin:
                        offer_cabin = "first"
                    elif "business" in raw_cabin or "biz" in raw_cabin:
                        offer_cabin = "business"
                    elif "premium" in raw_cabin:
                        offer_cabin = "premium_economy"
                    elif "economy" in raw_cabin or "eco" in raw_cabin:
                        offer_cabin = "economy"
                    else:
                        offer_cabin = api_cabin.lower() if api_cabin else "economy"

                    # Honest cabin matching: strictly respect requested cabin
                    req_clean = api_cabin.lower()
                    if req_clean in ["first", "business", "economy"]:
                        if raw_cabin and offer_cabin != req_clean:
                            continue

                    price_basis = "per_person" if adults == 1 else "party_total"
                    pax_label = f"Party total ({adults} passengers)" if adults > 1 else "Per passenger"

                    observed_details = {
                        "fareId": raw_fare_id,
                        "baseFare": base_fare,
                        "tax": tax,
                        "requestedOrigin": orig_iata,
                        "requestedDestination": dest_iata,
                        "originAirportCode": seg0.origin,
                        "originAirportName": orig_airport_name,
                        "originCity": orig_city_name,
                        "destinationAirportCode": seg_last.destination,
                        "destinationAirportName": dest_airport_name,
                        "destinationCity": dest_city_name,
                        "routeMatched": route_matched,
                        "isAlternateAirport": is_alternate_airport,
                        "rawCarrierNames": raw_airline_names,
                        "flightNumbers": [s.flightNumber for s in segments if s.flightNumber],
                        "aircraft": [s.aircraft for s in segments if s.aircraft],
                        "terminals": [t for t in [dep_terminal, arr_terminal] if t],
                        "baggagePolicies": [
                            f"Check-in: {check_in_str}" if check_in_str else "Check-in: Standard allowance",
                            f"Cabin: {cabin_bag_str}" if cabin_bag_str else "Cabin: 1 piece 7 kg",
                        ],
                        "fareBreakdown": f"Base: ₹{base_fare:.0f} + Tax: ₹{tax:.0f} ({pax_label})",
                    }

                    offers.append(
                        FlightOffer(
                            offerId=offer_id,
                            provider="BankG",
                            airlineLogo=airline_logo,
                            price=Money(amount=total_fare, currency=currency, verified=True),
                            outbound=FlightLeg(
                                airline=airline_name,
                                airlineLogo=airline_logo,
                                flightNumber=seg0.flightNumber,
                                origin=seg0.origin,
                                destination=seg_last.destination,
                                departureLocal=seg0.departureLocal,
                                arrivalLocal=seg_last.arrivalLocal,
                                durationMinutes=total_duration,
                                stops=stops,
                                segments=segments,
                            ),
                            cabinClass=offer_cabin,
                            bags=FlightBags(
                                carryOn=1 if cabin_bag_str else 0,
                                checked=1 if check_in_str else 0,
                            ),
                            isCheapest=False,
                            sourceUrl=booking_url,
                            observedDetails=observed_details,
                            priceBasis=price_basis,
                        )
                    )

            if excluded_route_mismatches:
                logger.info(
                    f"BankG excluded {excluded_route_mismatches} non-matching airport journey(s) for exact route {orig_iata}->{dest_iata}"
                )

            offers.sort(key=lambda o: (bool(o.observedDetails.get("isAlternateAirport")), o.price.amount))
            exact_offers = [o for o in offers if o.observedDetails.get("routeMatched", True)]
            if exact_offers:
                exact_offers[0].isCheapest = True
            elif offers:
                offers[0].isCheapest = True

            return offers

        except httpx.TimeoutException as te:
            sanitized_error = type(te).__name__
            raise TimeoutError(f"BankG flight search timed out after {self.timeout}s: {sanitized_error}")
        except (BaseException, asyncio.CancelledError) as be:
            if not sanitized_error:
                sanitized_error = type(be).__name__
            raise
        finally:
            elapsed_ms = int((time.monotonic() - start_mono) * 1000)
            logger.info(
                f"[Supplier Diagnostic] provider=BankG endpoint={gateway_url} "
                f"route={orig_iata}->{dest_iata} date={valid_date} cabin={api_cabin} adults={adults} "
                f"duration_ms={elapsed_ms} reached={reached_supplier} status={status_code} "
                f"error={sanitized_error} usable={usable_response}"
            )

    @property
    def base_url(self) -> str:
        return getattr(settings, "BANKG_BASE_URL", "https://bankg-stagging-api.testuat.com/travel/v1/api").rstrip("/")

    async def review_fare(
        self,
        fare_id: str,
        search_offer: Optional[FlightOffer] = None,
        search_price: Optional[float] = None,
        search_currency: Optional[str] = None,
        search_base_fare: Optional[float] = None,
        search_tax: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Gateway v3.2 Section 4: POST /travel-master/flightReview
        Verifies seat inventory and locks pricing prior to passenger collection (15-minute lock window).
        Requires fare ID returned from airSearch (journeys[].fares[].fareId).

        Data-Integrity Rules (Priority 1):
        - Preserves original search price and reviewed price separately.
        - Detects material price difference (>= 0.01) and marks status as PRICE_CHANGED.
        - Never claims original price is confirmed when review returns a different price, missing price,
          currency mismatch, or mismatched fare identifier.
        - Never silently overwrites, averages, or invents values.
        """
        clean_fare_id = _clean_supplier_id(fare_id, "fareId")

        if search_offer is not None:
            if search_price is None and search_offer.price and search_offer.price.amount > 0:
                search_price = float(search_offer.price.amount)
            if search_currency is None and search_offer.price and search_offer.price.currency:
                search_currency = str(search_offer.price.currency).strip().upper()
            obs = search_offer.observedDetails or {}
            if search_base_fare is None and obs.get("baseFare") is not None:
                search_base_fare = float(obs["baseFare"])
            if search_tax is None and obs.get("tax") is not None:
                search_tax = float(obs["tax"])

        url = f"{self.base_url}/travel-master/flightReview"
        payload = {"fareId": [clean_fare_id]}
        headers = self._build_headers(payload)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0, read=self.timeout)) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"BankG flightReview HTTP error: {resp.status_code}")
            data = resp.json()

        res_code, desc, rev_data = _unwrap_bankg_envelope(data)
        if res_code != "0":
            raise RuntimeError(f"BankG flightReview failed: {desc or 'Unknown review error'}")

        booking_id = rev_data.get("bookingId") or ""
        req_id = rev_data.get("reqId") or ""
        flights_list = rev_data.get("flights", []) if isinstance(rev_data.get("flights"), list) else []

        # Live BankG nests pricing, cancellationRules, dateChangeRules, and noShowRules inside flights[0],
        # and nests mealOptions/baggageOptions inside flights[].segments[]. Support both nested and top-level.
        pricing = list(rev_data.get("pricing", []))
        cancellation_rules = list(rev_data.get("cancellationRules", []))
        date_change_rules = list(rev_data.get("dateChangeRules", []))
        no_show_rules = list(rev_data.get("noShowRules", []))
        meal_options = []
        baggage_options = []

        for flight in flights_list:
            if not isinstance(flight, dict):
                continue
            pricing.extend(flight.get("pricing", []) or [])
            cancellation_rules.extend(flight.get("cancellationRules", []) or [])
            date_change_rules.extend(flight.get("dateChangeRules", []) or [])
            no_show_rules.extend(flight.get("noShowRules", []) or [])
            meal_options.extend(flight.get("mealOptions", []) or [])
            baggage_options.extend(flight.get("baggageOptions", []) or [])
            for seg in flight.get("segments", []) or []:
                if isinstance(seg, dict):
                    meal_options.extend(seg.get("mealOptions", []) or [])
                    baggage_options.extend(seg.get("baggageOptions", []) or [])

        raw_total = None
        base_fare = None
        total_tax = None
        tax_breakup = {}
        price_id = ""
        reviewed_currency = "INR"
        if pricing and isinstance(pricing[0], dict):
            p0 = pricing[0]
            if p0.get("totalFare") is not None:
                try:
                    raw_total = float(p0.get("totalFare"))
                except (TypeError, ValueError):
                    raw_total = None
            if p0.get("baseFare") is not None:
                try:
                    base_fare = float(p0.get("baseFare"))
                except (TypeError, ValueError):
                    base_fare = None
            if p0.get("totalTax") is not None:
                try:
                    total_tax = float(p0.get("totalTax"))
                except (TypeError, ValueError):
                    total_tax = None
            tax_breakup = p0.get("taxBreakup") if isinstance(p0.get("taxBreakup"), dict) else {}
            price_id = str(p0.get("priceId") or "")
            reviewed_currency = str(p0.get("currency") or rev_data.get("currency") or "INR").strip().upper()

        price_verified = raw_total is not None and raw_total > 0
        reviewed_price = raw_total if price_verified else None
        fare_id_matched = _verify_fare_id_correlation(clean_fare_id, price_id, req_id) if price_id else False

        norm_search_curr = (search_currency or "INR").strip().upper() if search_price is not None else None
        currency_mismatch = bool(norm_search_curr and reviewed_currency and norm_search_curr != reviewed_currency)

        price_delta: Optional[float] = None
        tax_delta: Optional[float] = None
        base_fare_delta: Optional[float] = None
        price_changed = False
        original_price_confirmed = False
        discrepancy_notice: Optional[str] = None

        if not price_verified:
            status = "PRICE_MISSING"
            discrepancy_notice = "Supplier flightReview response did not include a valid positive totalFare."
        elif not fare_id_matched:
            status = "FARE_ID_MISMATCH"
            discrepancy_notice = (
                f"Supplier returned priceId '{price_id}' which does not match requested fareId '{clean_fare_id}'."
            )
        elif currency_mismatch:
            status = "CURRENCY_MISMATCH"
            price_changed = True
            discrepancy_notice = (
                f"Currency mismatch between airSearch ({norm_search_curr} {search_price}) and "
                f"flightReview ({reviewed_currency} {reviewed_price})."
            )
        elif search_price is not None:
            price_delta = round(reviewed_price - float(search_price), 2)
            if search_tax is not None and total_tax is not None:
                tax_delta = round(total_tax - float(search_tax), 2)
            if search_base_fare is not None and base_fare is not None:
                base_fare_delta = round(base_fare - float(search_base_fare), 2)
            if abs(price_delta) >= 0.01:
                price_changed = True
                status = "PRICE_CHANGED"
                original_price_confirmed = False
                direction = "increased" if price_delta > 0 else "decreased"
                sign = "+" if price_delta > 0 else ""
                discrepancy_notice = (
                    f"Fare {direction} during supplier review from ₹{float(search_price):.2f} (airSearch) "
                    f"to ₹{reviewed_price:.2f} (flightReview) ({sign}₹{price_delta:.2f})."
                )
            else:
                price_changed = False
                status = "LOCKED"
                original_price_confirmed = True
        else:
            status = "LOCKED"

        return {
            "requestedFareId": clean_fare_id,
            "fareIdMatched": fare_id_matched,
            "bookingId": booking_id,
            "reqId": req_id,
            "priceId": price_id,
            "fareLockMinutes": 15,
            "status": status,
            "priceVerified": price_verified,
            "searchPrice": float(search_price) if search_price is not None else None,
            "searchBaseFare": float(search_base_fare) if search_base_fare is not None else None,
            "searchTax": float(search_tax) if search_tax is not None else None,
            "searchCurrency": norm_search_curr,
            "reviewedPrice": reviewed_price,
            "reviewedCurrency": reviewed_currency,
            "totalFare": reviewed_price,
            "baseFare": base_fare,
            "totalTax": total_tax,
            "taxBreakup": tax_breakup,
            "priceChanged": price_changed,
            "priceDelta": price_delta,
            "baseFareDelta": base_fare_delta,
            "taxDelta": tax_delta,
            "currencyMismatch": currency_mismatch,
            "originalPriceConfirmed": original_price_confirmed,
            "priceDiscrepancyNotice": discrepancy_notice,
            "cancellationRules": cancellation_rules,
            "dateChangeRules": date_change_rules,
            "noShowRules": no_show_rules,
            "mealOptions": meal_options,
            "baggageOptions": baggage_options,
            "raw": rev_data,
        }

    async def get_seat_map(self, booking_id: str) -> Dict[str, Any]:
        """
        Gateway v3.2 Section 5: POST /travel-master/flightSeatMap
        Retrieves the aircraft seat layout and seat availability / pricing for each flight segment of a reviewed booking.
        Precondition: Requires a valid bookingId returned from flightReview.
        """
        if not booking_id or not str(booking_id).strip():
            raise ValueError("bookingId is required for flightSeatMap (must be obtained from a successful flightReview)")

        url = f"{self.base_url}/travel-master/flightSeatMap"
        payload = {"bookingId": str(booking_id).strip()}
        headers = self._build_headers(payload)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0, read=self.timeout)) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"BankG flightSeatMap HTTP error: {resp.status_code}")
            data = resp.json()

        res_code, desc, map_data = _unwrap_bankg_envelope(data)
        if res_code != "0":
            raise RuntimeError(f"BankG flightSeatMap failed: {desc or 'Unknown seat map error'}")

        return {
            "bookingId": map_data.get("bookingId", str(booking_id).strip()),
            "seatMaps": map_data.get("seatMaps", []),
        }

    async def get_fare_rules(self, price_id: str) -> Dict[str, Any]:
        """
        Gateway v3.2 Section 6: POST /travel-master/fareRule
        Returns cancellation charges, date-change charges, no-show rules by time window before departure.
        """
        clean_price_id = _clean_supplier_id(price_id, "priceId")

        url = f"{self.base_url}/travel-master/fareRule"
        payload = {"priceId": clean_price_id}
        headers = self._build_headers(payload)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0, read=self.timeout)) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"BankG fareRule HTTP error: {resp.status_code}")
            data = resp.json()

        res_code, desc, rule_data = _unwrap_bankg_envelope(data)
        if res_code != "0":
            raise RuntimeError(f"BankG fareRule failed: {desc or 'Unknown fare rule error'}")

        return {
            "fareRules": rule_data.get("fareRules", []),
        }

    async def get_booking_details(self, booking_id: str, authenticated_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Gateway v3.2 Section 9: POST /travel-master/bookingDetails
        Read-only lookup for authenticated, authorized existing bookings.
        Reports PNR or ticket numbers ONLY if retrieved response actually contains it.
        Requires authenticated identity and server-side ownership verification.
        """
        if not authenticated_user_id:
            raise PermissionError(
                "Unauthorized: Access to booking records (PNR, ticket, and traveller manifest) "
                "requires verified caller authentication and booking ownership."
            )

        if not booking_id or not str(booking_id).strip():
            raise ValueError("bookingId is required for bookingDetails")

        url = f"{self.base_url}/travel-master/bookingDetails"
        payload = {"bookingId": str(booking_id).strip()}
        headers = self._build_headers(payload)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0, read=self.timeout)) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"BankG bookingDetails HTTP error: {resp.status_code}")
            data = resp.json()

        res_code, desc, b_data = _unwrap_bankg_envelope(data)
        if res_code != "0":
            raise RuntimeError(f"BankG bookingDetails failed: {desc or 'Unknown booking lookup error'}")

        booking_info = b_data.get("booking", {})
        travellers = b_data.get("travellers", [])

        # Extract verified PNRs and ticket numbers without fabrication
        pnr_details = {}
        ticket_numbers = {}
        for t in travellers:
            if isinstance(t, dict):
                pnr_details.update(t.get("pnrDetails", {}))
                ticket_numbers.update(t.get("ticketNumberDetails", {}))

        return {
            "bookingId": booking_info.get("bookingId", str(booking_id).strip()),
            "status": booking_info.get("status", "UNKNOWN"),
            "amount": booking_info.get("amount"),
            "createdOn": booking_info.get("createdOn"),
            "trips": b_data.get("trips", []),
            "pnrs": pnr_details,
            "tickets": ticket_numbers,
            "travellersCount": len(travellers),
        }

    async def get_amendment_details(self, amendment_id: str, authenticated_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Gateway v3.2 Section 11: POST /travel-master/amendmentDetails
        Read-only lookup for authenticated, authorized existing amendments.
        Requires authenticated identity and server-side ownership verification.
        """
        if not authenticated_user_id:
            raise PermissionError(
                "Unauthorized: Access to amendment records "
                "requires verified caller authentication and ownership."
            )

        if not amendment_id or not str(amendment_id).strip():
            raise ValueError("amendmentId is required for amendmentDetails")

        url = f"{self.base_url}/travel-master/amendmentDetails"
        payload = {"amendmentId": str(amendment_id).strip()}
        headers = self._build_headers(payload)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0, read=self.timeout)) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"BankG amendmentDetails HTTP error: {resp.status_code}")
            data = resp.json()

        res_code, desc, raw_amend = _unwrap_bankg_envelope(data)
        if res_code != "0":
            raise RuntimeError(f"BankG amendmentDetails failed: {desc or 'Unknown amendment query error'}")

        a_data = raw_amend.get("amendment", {})
        return {
            "amendmentId": a_data.get("amendmentId", str(amendment_id).strip()),
            "bookingId": a_data.get("bookingId"),
            "status": a_data.get("status", "UNKNOWN"),
            "totalAmendmentCharges": a_data.get("totalAmendmentCharges", 0),
            "totalRefundableAmount": a_data.get("totalRefundableAmount", 0),
        }

    # =========================================================================
    # CRITICAL SERVER-SIDE SAFETY GATES & OPERATION GUARDS (NON-NEGOTIABLE)
    # =========================================================================

    async def book_flight(self, *args, **kwargs) -> Any:
        """
        Gateway v3.2 Section 7: POST /travel-master/flightBook
        PERMANENTLY DISABLED IN THIS RELEASE.
        """
        raise OperationDisabledError(
            "CRITICAL SAFETY GATE VIOLATION: flightBook is permanently disabled in this release. "
            "Ticket booking, passenger manifest submission, and financial debit operations are prohibited."
        )

    async def submit_amendment(self, *args, **kwargs) -> Any:
        """
        Gateway v3.2 Section 10: POST /travel-master/submitAmendment
        PERMANENTLY DISABLED IN THIS RELEASE.
        """
        raise OperationDisabledError(
            "CRITICAL SAFETY GATE VIOLATION: submitAmendment is permanently disabled in this release. "
            "Direct cancellation, reschedule, and ticket mutation operations are prohibited."
        )

    async def validate_fare(self, *args, **kwargs) -> Any:
        """
        Gateway v3.2 Section 8: POST /travel-master/fareValidate
        DISABLED BEHIND FEATURE FLAG.
        """
        if not getattr(settings, "ENABLE_FARE_VALIDATE", False):
            raise OperationDisabledError(
                "fareValidate is disabled behind feature flag ENABLE_FARE_VALIDATE pending provider clarification."
            )
        raise NotImplementedError("fareValidate sandbox integration pending.")

    async def check_balance(self, *args, **kwargs) -> Any:
        """
        Gateway v3.2 Section 2: POST /travel-master/checkBalance
        RESTRICTED TO INTERNAL OPERATORS ONLY.
        """
        raise OperationDisabledError(
            "checkBalance is restricted to internal operator access and is not an ordinary conversational tool."
        )

