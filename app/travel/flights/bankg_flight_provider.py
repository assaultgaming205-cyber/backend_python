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

logger = logging.getLogger("bankg_flight_provider")

GATEWAY_URL = "https://bankg-stagging-api.testuat.com/travel/v1/api/travel-master-gateway/airSearch"

AIRLINE_BOOKING_URLS = {
    "AI": "https://www.airindia.com",
    "6E": "https://www.goindigo.in",
    "SG": "https://www.spicejet.com",
    "IX": "https://www.airindiaexpress.com",
    "QP": "https://www.akasaair.com",
    "UK": "https://www.airindia.com",
}


def _resolve_iata(city_or_code: str) -> str:
    """Resolve city name or code to 3-letter IATA code."""
    if not city_or_code:
        return "DEL"
    clean = city_or_code.strip()
    if len(clean) == 3 and clean.isalpha():
        return clean.upper()
    resolved = LocationResolver.resolve_city(clean)
    if resolved and resolved.code:
        return resolved.code.upper()
    return clean[:3].upper()


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


class BankGFlightProvider:
    def __init__(self, timeout: float = 25.0):
        self.timeout = timeout

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
        headers = {
            "Content-Type": "application/json",
            "xchecksum": getattr(settings, "BANKG_CHECKSUM", "DUMMY_ENCRYPTED_CHECKSUM_SAMPLE"),
        }

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

            response_code = str(
                data.get("responseCode")
                if "responseCode" in data
                else data.get("options", {}).get("responseCode", "")
            )
            description = data.get("description") or data.get("options", {}).get("description", "")
            journeys_data = data.get("data") if "data" in data else data.get("options", {}).get("data", {})
            journeys = journeys_data.get("journeys", []) if isinstance(journeys_data, dict) else []

            if response_code != "0":
                sanitized_error = f"PROVIDER_ERROR_{response_code}"
                raise RuntimeError(f"BankG provider error: {description or response_code}")

            usable_response = True
            logger.info(f"BankG returned {len(journeys)} flight journeys for {orig_iata}->{dest_iata}")

            offers: List[FlightOffer] = []
            for i, journey in enumerate(journeys):
                flights_raw = journey.get("flights", [])
                fares_raw = journey.get("fares", [])
                if not flights_raw or not fares_raw:
                    continue

                # Flight segments
                segments: List[FlightSegment] = []
                raw_airline_names: List[Optional[str]] = []
                total_duration = 0
                for f in flights_raw:
                    airline_dict = f.get("airline", {})
                    carrier_code = airline_dict.get("code", "")
                    raw_airline_names.append(airline_dict.get("name"))
                    airline_name = canonical_airline_name(airline_dict.get("name"), carrier_code).display
                    flight_num = f"{carrier_code} {f.get('flightNumber', '')}".strip()
                    aircraft = f.get("aircraftType")
                    dur = f.get("durationMinutes") or 0
                    total_duration += dur

                    dep_info = f.get("departure", {})
                    arr_info = f.get("arrival", {})

                    segments.append(
                        FlightSegment(
                            carrierCode=carrier_code,
                            carrierName=airline_name,
                            flightNumber=flight_num,
                            origin=dep_info.get("airportCode", orig_iata),
                            destination=arr_info.get("airportCode", dest_iata),
                            departureLocal=dep_info.get("dateTime") or f"{valid_date}T00:00:00",
                            arrivalLocal=arr_info.get("dateTime") or f"{valid_date}T00:00:00",
                            durationMinutes=dur,
                            aircraft=aircraft,
                        )
                    )

                if not segments:
                    continue

                seg0 = segments[0]
                seg_last = segments[-1]
                stops = max(0, len(segments) - 1)
                primary_carrier = seg0.carrierCode or "AI"
                airline_name = seg0.carrierName or "Airline"
                airline_logo = f"https://images.kiwi.com/airlines/64x64/{primary_carrier}.png"

                booking_url = AIRLINE_BOOKING_URLS.get(
                    primary_carrier,
                    f"https://www.google.com/travel/flights?q=flights+from+{orig_iata}+to+{dest_iata}+on+{valid_date}",
                )

                dep_terminal = flights_raw[0].get("departure", {}).get("terminal")
                arr_terminal = flights_raw[-1].get("arrival", {}).get("terminal")

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

                    fare_id = fare.get("fareId") or f"bankg_{i}_{j}"
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
                                origin=orig_iata,
                                destination=dest_iata,
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

            offers.sort(key=lambda o: o.price.amount)
            if offers:
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

