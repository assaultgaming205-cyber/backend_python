from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from ..travel.flights.bankg_flight_provider import BankGFlightProvider

router = APIRouter(tags=["flights"])
flight_provider = BankGFlightProvider()


class FlightSearchRequest(BaseModel):
    origin: str
    destination: str
    departureDate: Optional[str] = None
    returnDate: Optional[str] = None
    adults: int = 1
    cabin: str = "economy"


@router.get("/flights/search")
@router.get("/api/flights/search")
@router.get("/travel/flights/search")
async def search_flights_get(
    origin: str = Query(...),
    destination: str = Query(...),
    departureDate: Optional[str] = Query(None),
    returnDate: Optional[str] = Query(None),
    adults: int = Query(1),
    cabin: str = Query("economy"),
):
    flights = await flight_provider.search_flights(
        origin=origin,
        destination=destination,
        departure_date=departureDate or "",
        adults=adults,
        cabin_class=cabin,
    )
    return {
        "success": True,
        "results": flights,
        "count": len(flights),
        "source": "bankg",
    }


@router.post("/flights/search")
@router.post("/api/flights/search")
@router.post("/travel/flights/search")
async def search_flights_post(body: FlightSearchRequest):
    flights = await flight_provider.search_flights(
        origin=body.origin,
        destination=body.destination,
        departure_date=body.departureDate or "",
        adults=body.adults,
        cabin_class=body.cabin,
    )
    return {
        "success": True,
        "results": flights,
        "count": len(flights),
        "source": "bankg",
    }
