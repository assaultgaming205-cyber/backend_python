from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

CabinClass = Literal['economy', 'premium_economy', 'business', 'first']
FlightSort = Literal['cheapest', 'fastest', 'fewest_stops', 'earliest']


class Money(BaseModel):
    amount: float
    currency: str = "INR"
    verified: Optional[bool] = None


class FlightSegment(BaseModel):
    carrierCode: Optional[str] = None
    flightNumber: Optional[str] = None
    carrierName: Optional[str] = None
    origin: str
    destination: str
    departureLocal: str
    arrivalLocal: str
    durationMinutes: Optional[int] = None
    aircraft: Optional[str] = None
    departureTimezone: Optional[str] = None
    arrivalTimezone: Optional[str] = None
    departureUtc: Optional[str] = None
    arrivalUtc: Optional[str] = None


class FlightLeg(BaseModel):
    airline: Optional[str] = None
    airlineLogo: Optional[str] = None
    origin: str
    destination: str
    departureLocal: str
    arrivalLocal: str
    durationMinutes: Optional[int] = None
    stops: Optional[int] = 0
    segments: List[FlightSegment] = Field(default_factory=list)


class FlightBags(BaseModel):
    carryOn: Optional[int] = None
    checked: Optional[int] = None


class FlightOffer(BaseModel):
    offerId: str
    provider: str
    airlineLogo: Optional[str] = None
    price: Money
    outbound: FlightLeg
    inbound: Optional[FlightLeg] = None
    cabinClass: Optional[str] = "economy"
    bags: Optional[FlightBags] = None
    requiresSelfTransfer: Optional[bool] = False
    isCheapest: Optional[bool] = False
    isBest: Optional[bool] = False
    observedAt: Optional[str] = None
    priceBasis: Optional[str] = "per_person"
    searchContextId: Optional[str] = None
    sourceUrl: Optional[str] = None
    calendarFare: Optional[bool] = False
    extractionStatus: Optional[str] = "summary_only"
    observedDetails: Optional[Dict[str, Any]] = None


class FlightBookingLinks(BaseModel):
    offerId: str
    airline: Optional[str] = None
    bookingUrl: str
    provider: str
