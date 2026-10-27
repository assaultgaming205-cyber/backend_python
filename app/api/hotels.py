from typing import Optional, List
from fastapi import APIRouter
from pydantic import BaseModel
from ..travel.hotels.makemytrip_hotel import MakeMyTripHotelProvider

router = APIRouter(tags=["hotels"])
hotel_provider = MakeMyTripHotelProvider()


class HotelSearchRequest(BaseModel):
    destination: str
    checkIn: str
    checkOut: str
    adults: int = 2
    children: int = 0
    childrenAges: Optional[List[int]] = None
    rooms: int = 1


@router.post("/travel/hotels/search")
@router.post("/api/hotels/search")
async def search_hotels(body: HotelSearchRequest):
    hotels = await hotel_provider.search_hotels(
        destination=body.destination,
        check_in=body.checkIn,
        check_out=body.checkOut,
        adults=body.adults,
        children=body.children,
        children_ages=body.childrenAges,
        rooms=body.rooms,
    )
    return {
        "success": True,
        "results": hotels,
        "count": len(hotels),
        "source": "makemytrip_hotel",
    }


@router.get("/travel/hotels/{hotel_id}/photo")
@router.get("/api/travel/hotels/{hotel_id}/photo")
async def get_hotel_photo(hotel_id: str):
    return {
        "imageUrl": None,
        "source": "makemytrip",
        "attributions": []
    }
