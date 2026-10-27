import logging
from typing import List, Optional
from .bankg_flight_provider import BankGFlightProvider
from ...schemas.flight import FlightOffer

logger = logging.getLogger("agoda_provider")


class AgodaFlightProvider:
    """
    Replaced legacy Playwright Agoda scraper with fast BankG Flight Gateway.
    Keeps class name for seamless backwards compatibility across all callers.
    """
    def __init__(self, session_service=None):
        self._provider = BankGFlightProvider()

    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1,
        cabin_class: str = "economy",
        **kwargs,
    ) -> List[FlightOffer]:
        return await self._provider.search_flights(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            adults=adults,
            cabin_class=cabin_class,
            **kwargs,
        )
