from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from ..travel.trains.makemytrip_rail import MakeMyTripRailProvider

router = APIRouter(tags=["trains"])
rail_provider = MakeMyTripRailProvider()


class TrainSearchRequest(BaseModel):
    origin: str
    destination: str
    date: str


@router.post("/travel/trains/search")
@router.post("/api/trains/search")
async def search_trains(body: TrainSearchRequest):
    trains = await rail_provider.search_trains(
        origin=body.origin,
        destination=body.destination,
        date=body.date,
    )
    return {
        "success": True,
        "results": trains,
        "count": len(trains),
        "source": "makemytrip_rail",
    }
