from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

TrainSort = Literal['departure', 'duration']


class Station(BaseModel):
    code: str
    name: str
    city: Optional[str] = None


class TrainClassFare(BaseModel):
    classCode: str
    className: str
    fare: float
    currency: Optional[str] = "INR"
    status: Optional[str] = None


RAIL_CLASS_DISPLAY_NAMES = {
    '1A': '1st AC',
    '2A': '2nd AC',
    '3A': '3rd AC',
    '3E': '3 AC Economy',
    'SL': 'Sleeper',
    'CC': 'Chair Car',
    'EC': 'Exec Chair Car',
    'EA': 'Anubhuti',
    '2S': '2nd Sitting',
    'FC': 'First Class',
    'EV': 'Vistadome AC',
    'VS': 'Vistadome Non-AC',
}


class TrainLiveStatus(BaseModel):
    trainNumber: str
    currentStation: Optional[str] = None
    delayMinutes: Optional[int] = 0
    status: Optional[str] = None


class DayAvailability(BaseModel):
    date: str
    status: str
    fare: Optional[float] = None


class TrainAvailability(BaseModel):
    trainNumber: str
    classCode: str
    days: List[DayAvailability] = Field(default_factory=list)


class RailTicketSegment(BaseModel):
    origin: str
    destination: str
    classCode: str
    quota: Optional[str] = None
    fare: Optional[float] = None
    currency: Optional[str] = "INR"
    availability: Optional[str] = None
    departureDate: Optional[str] = None
    arrivalDate: Optional[str] = None
    departureTime: Optional[str] = None
    arrivalTime: Optional[str] = None


class RailAlternateOption(BaseModel):
    classCode: str
    type: str
    boarding: Optional[Station] = None
    alighting: Optional[Station] = None
    quotedTotal: Optional[float] = None
    currency: Optional[str] = "INR"
    segments: List[RailTicketSegment] = Field(default_factory=list)
    availability: List[str] = Field(default_factory=list)
    updatedLabels: List[str] = Field(default_factory=list)
    requiresMultipleTickets: bool = False
    fareNeedsReview: bool = False


class TrainOption(BaseModel):
    provider: str = "makemytrip"
    trainNumber: str
    trainName: str
    type: Optional[str] = None
    category: Optional[str] = None
    runDays: Optional[List[str]] = None
    origin: Station
    destination: Station
    departureTime: Optional[str] = None
    arrivalTime: Optional[str] = None
    arrivalDayOffset: Optional[int] = 0
    durationMinutes: Optional[int] = None
    distanceKm: Optional[int] = None
    haltsBetween: Optional[int] = None
    classes: Optional[List[str]] = None
    classFares: Optional[List[TrainClassFare]] = None
    departureDate: Optional[str] = None
    journeyKind: Optional[str] = "primary"
    alternateOptions: Optional[List[RailAlternateOption]] = None
