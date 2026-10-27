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

    def _gate(self, name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Reason to block the supplier call, or None. Runs before any provider is invoked."""
        today = self._today()
        if name == "search_flights":
            d = arguments.get("departureDate") or arguments.get("departure_date")
            why = check_future_iso(d, today)
            return f"flight_date_{why}" if why else None
        if name == "search_trains":
            why = check_future_iso(arguments.get("date"), today)
            return f"train_date_{why}" if why else None
        if name == "search_hotels":
            return check_hotel_dates(
                arguments.get("checkIn") or arguments.get("check_in"),
                arguments.get("checkOut") or arguments.get("check_out"), today)
        return None

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        results: ChatResults,
    ) -> ToolCallSummary:
        start_time = time.time()
        success = True
        error_msg: Optional[str] = None

        try:
            blocked = self._gate(name, arguments)
            if blocked:
                logger.warning(f"Supplier call {name} blocked by preflight: {blocked}")
                return ToolCallSummary(name=name, arguments=arguments, durationMs=0,
                                       success=False, error=f"blocked:{blocked}")
            if name == "search_flights":
                timeout = max(5.0, getattr(settings, "FLIGHT_PROVIDER_TIMEOUT_MS", 25000) / 1000)
            else:
                timeout = max(1.0, getattr(settings, "PROVIDER_TIMEOUT_MS", 30000) / 1000)
            async with self._sem:
                await asyncio.wait_for(self._dispatch(name, arguments, results), timeout)

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

    async def _dispatch(self, name: str, arguments: Dict[str, Any], results: ChatResults) -> None:
        if name == "search_flights":
            flights = await self.flight_provider.search_flights(
                origin=arguments.get("origin", ""),
                destination=arguments.get("destination", ""),
                departure_date=arguments.get("departureDate") or arguments.get("departure_date", ""),
                adults=int(arguments.get("adults", 1)),
                cabin_class=arguments.get("cabin") or arguments.get("cabin_class", "economy"),
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

        else:
            logger.warning(f"Unknown tool name: {name}")
