from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from .flight import Money

AirportPickupStatus = Literal[
    'included',
    'available_for_fee',
    'available_cost_unknown',
    'unavailable',
    'unknown',
]


class AirportPickupBenefit(BaseModel):
    status: AirportPickupStatus = "unknown"
    conditions: Optional[str] = None
    hours: Optional[str] = None
    advanceNoticeHours: Optional[int] = None
    feeAmount: Optional[float] = None
    feeCurrency: Optional[str] = None
    confirmed: Optional[bool] = False


class HotelRating(BaseModel):
    score: float
    scale: Optional[float] = 10.0
    reviewCount: Optional[int] = 0
    label: Optional[str] = None


class HotelPrice(Money):
    display: Optional[str] = None
    beforeDiscount: Optional[str] = None


class HotelOffer(BaseModel):
    hotelId: str
    provider: str = "makemytrip"
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    distance: Optional[str] = None
    stars: Optional[int] = None
    rating: Optional[HotelRating] = None
    imageUrl: Optional[str] = None
    roomName: Optional[str] = None
    price: Optional[HotelPrice] = None
    priceBasis: Optional[str] = "per_night"
    freeCancellation: Optional[bool] = False
    noPrepayment: Optional[bool] = False
    soldOut: Optional[bool] = False
    airportPickup: Optional[AirportPickupBenefit] = None
    amenities: Optional[List[str]] = Field(default_factory=list)
    fetchedAt: Optional[str] = None
