import asyncio
import time
import logging
from typing import Callable, Dict, Any, List, Optional
from ..config import settings
from ..schemas.chat import ChatResults, ToolCallSummary
from ..travel.flights.bankg_flight_provider import BankGFlightProvider
from ..travel.trains.makemytrip_rail import MakeMyTripRailProvider
from ..travel.hotels.makemytrip_hotel import MakeMyTripHotelProvider
from .search_preflight import today_ist, check_future_iso, check_hotel_dates

logger = logging.getLogger("tool_executor")

MAX_CONCURRENT_SEARCHES = 3


ALLOWLISTED_TOOLS = {
    "search_flights",
    "review_fare",
    "get_seat_map",
    "get_fare_rules",
    "get_booking_details",
    "get_amendment_details",
    "search_trains",
    "search_hotels",
}

PROHIBITED_TOOLS = {
    "book_flight",
    "flight_book",
    "flightBook",
    "instant_ticketing",
    "submit_amendment",
    "submitAmendment",
    "cancel_flight",
    "cancel_booking",
    "fare_validate",
    "fareValidate",
    "check_balance",
    "checkBalance",
}


class ToolExecutor:
    def __init__(
        self,
        flight_provider: Optional[BankGFlightProvider] = None,
        rail_provider: Optional[MakeMyTripRailProvider] = None,
        hotel_provider: Optional[MakeMyTripHotelProvider] = None,
        today_fn: Optional[Callable[[], Any]] = None,
    ):
        self.flight_provider = flight_provider or BankGFlightProvider()
        self.rail_provider = rail_provider or MakeMyTripRailProvider()
        self.hotel_provider = hotel_provider or MakeMyTripHotelProvider()
        self._today_fn = today_fn  # injectable clock for tests; returns an India-local date
        self._sem = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    def _today(self):
        return self._today_fn() if self._today_fn else today_ist(tz=getattr(settings, "APP_TIMEZONE", "Asia/Kolkata"))

    def _gate(
        self,
        name: str,
        arguments: Dict[str, Any],
        trusted_auth: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Reason to block the supplier call, or None. Runs before any provider is invoked."""
        norm_name = name.strip()
        is_details = "detail" in norm_name.lower() or "seat" in norm_name.lower()
        if norm_name in PROHIBITED_TOOLS or (not is_details and ("book" in norm_name.lower() or "amend" in norm_name.lower())):
            return f"prohibited_operation:{norm_name} is permanently blocked by server-side safety policy."

        if norm_name not in ALLOWLISTED_TOOLS:
            return f"unauthorized_tool:{norm_name} is not in the allowlist."

        today = self._today()
        if norm_name == "search_flights":
            d = arguments.get("departureDate") or arguments.get("departure_date")
            why = check_future_iso(d, today)
            return f"flight_date_{why}" if why else None
        if norm_name == "review_fare":
            fid = arguments.get("fareId") or arguments.get("fare_id")
            if not fid:
                return "missing_fare_id"
            return None
        if norm_name == "get_seat_map":
            bid = arguments.get("bookingId") or arguments.get("booking_id")
            if not bid:
                return "missing_booking_id"
            return None
        if norm_name == "get_fare_rules":
            pid = arguments.get("priceId") or arguments.get("price_id") or arguments.get("fareId") or arguments.get("fare_id")
            if not pid:
                return "missing_price_id"
            return None
        if norm_name == "get_booking_details":
            bid = arguments.get("bookingId") or arguments.get("booking_id")
            if not bid:
                return "missing_booking_id"
            # Server-side security gate: User/model arguments can NEVER establish ownership.
            # Only trusted_auth established server-side is accepted.
            auth_user = (trusted_auth or {}).get("user_id")
            verified = (trusted_auth or {}).get("verified_ownership", False)
            if not auth_user or not verified:
                return "unauthorized_booking_lookup: Access to booking records with sensitive PNR and passenger data is blocked until authenticated user identity and booking ownership are established."
            return None
        if norm_name == "get_amendment_details":
            aid = arguments.get("amendmentId") or arguments.get("amendment_id")
            if not aid:
                return "missing_amendment_id"
            auth_user = (trusted_auth or {}).get("user_id")
            verified = (trusted_auth or {}).get("verified_ownership", False)
            if not auth_user or not verified:
                return "unauthorized_amendment_lookup: Access to amendment records with refund details is blocked until authenticated user identity and ownership are established."
            return None
        if norm_name == "search_trains":
            why = check_future_iso(arguments.get("date"), today)
            return f"train_date_{why}" if why else None
        if norm_name == "search_hotels":
            return check_hotel_dates(
                arguments.get("checkIn") or arguments.get("check_in"),
                arguments.get("checkOut") or arguments.get("check_out"), today)
        return None

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        results: ChatResults,
        trusted_auth: Optional[Dict[str, Any]] = None,
    ) -> ToolCallSummary:
        start_time = time.time()
        success = True
        error_msg: Optional[str] = None

        try:
            blocked = self._gate(name, arguments, trusted_auth=trusted_auth)
            if blocked:
                logger.warning(f"Supplier call {name} blocked by preflight: {blocked}")
                return ToolCallSummary(name=name, arguments=arguments, durationMs=0,
                                       success=False, error=f"blocked:{blocked}")
            if name == "search_flights":
                timeout = max(5.0, getattr(settings, "FLIGHT_PROVIDER_TIMEOUT_MS", 25000) / 1000)
            else:
                timeout = max(1.0, getattr(settings, "PROVIDER_TIMEOUT_MS", 30000) / 1000)
            async with self._sem:
                await asyncio.wait_for(self._dispatch(name, arguments, results, trusted_auth=trusted_auth), timeout)

        except Exception as e:
            err_name = type(e).__name__
            err_msg = str(e) if str(e) else err_name
            logger.error(f"Error executing tool {name}: {err_msg}")
            success = False
            error_msg = err_msg

        duration_ms = int((time.time() - start_time) * 1000)
        return ToolCallSummary(
            name=name,
            arguments=arguments,
            durationMs=duration_ms,
            success=success,
            error=error_msg,
        )

    async def _dispatch(
        self,
        name: str,
        arguments: Dict[str, Any],
        results: ChatResults,
        trusted_auth: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name == "search_flights":
            extra_kw: Dict[str, Any] = {}
            if "includeAlternateAirports" in arguments or "include_alternate_airports" in arguments:
                extra_kw["include_alternate_airports"] = bool(
                    arguments.get("includeAlternateAirports") or arguments.get("include_alternate_airports")
                )
            if int(arguments.get("child") or arguments.get("children") or 0) > 0:
                extra_kw["child"] = int(arguments.get("child") or arguments.get("children"))
            if int(arguments.get("infant") or arguments.get("infants") or 0) > 0:
                extra_kw["infant"] = int(arguments.get("infant") or arguments.get("infants"))
            flights = await self.flight_provider.search_flights(
                origin=arguments.get("origin", ""),
                destination=arguments.get("destination", ""),
                departure_date=arguments.get("departureDate") or arguments.get("departure_date", ""),
                adults=int(arguments.get("adults", 1)),
                cabin_class=arguments.get("cabin") or arguments.get("cabin_class", "economy"),
                **extra_kw,
            )
            results.flights.extend(flights)

        elif name == "search_trains":
            trains = await self.rail_provider.search_trains(
                origin=arguments.get("origin", ""),
                destination=arguments.get("destination", ""),
                date=arguments.get("date", ""),
            )
            results.trains.extend(trains)

        elif name == "search_hotels":
            hotels = await self.hotel_provider.search_hotels(
                destination=arguments.get("destination", ""),
                check_in=arguments.get("checkIn") or arguments.get("check_in", ""),
                check_out=arguments.get("checkOut") or arguments.get("check_out", ""),
                adults=int(arguments.get("adults", 2)),
                rooms=int(arguments.get("rooms", 1)),
            )
            results.hotels.extend(hotels)

        elif name == "review_fare":
            fid = str(arguments.get("fareId") or arguments.get("fare_id") or "")
            clean_target = fid[len("bankg_"):] if fid.startswith("bankg_") else fid
            matched_offer = None
            for f_offer in results.flights:
                obs_fid = (f_offer.observedDetails or {}).get("fareId")
                if f_offer.offerId == fid or f_offer.offerId == f"bankg_{clean_target}" or obs_fid == clean_target:
                    matched_offer = f_offer
                    break
            rev_kw: Dict[str, Any] = {}
            if matched_offer is not None:
                rev_kw["search_offer"] = matched_offer
            if arguments.get("searchPrice") is not None or arguments.get("search_price") is not None:
                rev_kw["search_price"] = float(arguments.get("searchPrice") or arguments.get("search_price"))
            if arguments.get("searchCurrency") or arguments.get("search_currency"):
                rev_kw["search_currency"] = str(arguments.get("searchCurrency") or arguments.get("search_currency"))
            rev = await self.flight_provider.review_fare(fid, **rev_kw)
            if matched_offer is not None:
                rev["matchedOfferId"] = matched_offer.offerId
                rev["flightNumber"] = matched_offer.outbound.flightNumber
                rev["origin"] = matched_offer.outbound.origin
                rev["destination"] = matched_offer.outbound.destination
                matched_offer.observedDetails["fareReview"] = {
                    k: v for k, v in rev.items() if k != "raw"
                }
            results.fareReview = rev

        elif name == "get_seat_map":
            bid = str(arguments.get("bookingId") or arguments.get("booking_id") or "")
            smap = await self.flight_provider.get_seat_map(bid)
            results.seatMap = smap

        elif name == "get_fare_rules":
            pid = str(arguments.get("priceId") or arguments.get("price_id") or arguments.get("fareId") or arguments.get("fare_id") or "")
            frules = await self.flight_provider.get_fare_rules(pid)
            results.fareRules = frules

        elif name == "get_booking_details":
            bid = str(arguments.get("bookingId") or arguments.get("booking_id") or "")
            trusted_user = (trusted_auth or {}).get("user_id")
            bdetails = await self.flight_provider.get_booking_details(bid, authenticated_user_id=trusted_user)
            results.bookingLookup = bdetails

        elif name == "get_amendment_details":
            aid = str(arguments.get("amendmentId") or arguments.get("amendment_id") or "")
            trusted_user = (trusted_auth or {}).get("user_id")
            adetails = await self.flight_provider.get_amendment_details(aid, authenticated_user_id=trusted_user)
            results.amendmentLookup = adetails

        elif name in ("book_flight", "flight_book", "flightBook", "instant_ticketing"):
            await self.flight_provider.book_flight(**arguments)

        elif name in ("submit_amendment", "submitAmendment", "cancel_flight", "cancel_booking"):
            await self.flight_provider.submit_amendment(**arguments)

        else:
            logger.warning(f"Unknown tool name: {name}")
