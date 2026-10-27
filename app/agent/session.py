import time
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

TravelMode = Literal['unknown', 'flight', 'train', 'any']
TripTask = Literal['transport', 'hotel', 'discovery', 'full_trip']
UserLanguage = Literal['hinglish', 'hindi', 'english']
WorkflowState = Literal[
    'IDLE',
    'GENERAL_CONVERSATION',
    'COLLECTING_SEARCH_DETAILS',
    'SEARCH_READY',
    'SEARCHING',
    'RESULTS_READY',
    'FLIGHT_SELECTED',
    'FARE_REVIEW_REQUESTED',
    'FARE_REVIEWED',
    'TRAVEL_PREPARATION_ONLY',
    'EXISTING_BOOKING_LOOKUP',
    'AMENDMENT_STATUS_LOOKUP',
    'FLOW_PAUSED',
    'FLOW_ABANDONED',
]
SlotName = Literal[
    'origin',
    'destination',
    'mode',
    'date',
    'passengers',
    'class',
    'hotel_dates',
    'hotel_guests',
    'hotel_child_ages',
    'none'
]


class ResolvedLocation(BaseModel):
    code: str
    name: str
    station: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    isExplicit: bool = True
    rawInput: Optional[str] = None
    hasAirport: bool = True
    nearestAirportCode: Optional[str] = None
    nearestAirportName: Optional[str] = None


class DateState(BaseModel):
    departureDate: Optional[str] = None
    returnDate: Optional[str] = None
    nights: Optional[int] = None
    isExplicit: bool = False
    ambiguous: Optional[bool] = False
    isDateFlexible: Optional[bool] = False
    flexibleWindowDays: Optional[int] = None
    dateRangeStart: Optional[str] = None
    dateRangeEnd: Optional[str] = None
    dateRangeLabel: Optional[str] = None


class PassengerState(BaseModel):
    adults: int = 1
    children: int = 0
    infantsInSeat: int = 0
    infantsOnLap: int = 0
    isExplicit: bool = False
    compositionConfirmed: bool = False


class HotelState(BaseModel):
    destination: Optional[str] = None
    checkIn: Optional[str] = None
    checkOut: Optional[str] = None
    nights: Optional[int] = None
    adults: int = 2
    children: int = 0
    childrenAges: Optional[List[int]] = None
    rooms: int = 1
    maxBudget: Optional[float] = None
    budgetScope: Optional[Literal['per_night', 'total']] = None
    guestsExplicit: Optional[bool] = False
    datesExplicit: Optional[bool] = False


class SearchConstraints(BaseModel):
    airline: Optional[str] = None
    timeOfDay: Optional[Literal['morning', 'afternoon', 'evening', 'night']] = None
    maxBudget: Optional[float] = None
    budgetScope: Optional[Literal['per_person', 'total']] = None
    stops: Optional[int] = None
    luxury: Optional[bool] = False


class TravelSession(BaseModel):
    sessionId: str
    turnId: int = 0
    stateVersion: int = 1
    createdAt: int = Field(default_factory=lambda: int(time.time() * 1000))
    updatedAt: int = Field(default_factory=lambda: int(time.time() * 1000))
    userLanguage: UserLanguage = "hinglish"
    currentTask: TripTask = "transport"
    lastAskedSlot: SlotName = "none"
    origin: Optional[ResolvedLocation] = None
    destination: Optional[ResolvedLocation] = None
    destinationRegion: Optional[str] = None
    pendingClarification: Optional[str] = None
    mode: TravelMode = "unknown"
    modePolicy: Literal['flight_first', 'train_first', 'both'] = "flight_first"
    tripType: Literal['oneway', 'roundtrip'] = "oneway"
    dates: DateState = Field(default_factory=DateState)
    passengers: PassengerState = Field(default_factory=PassengerState)
    cabinClass: Literal['economy', 'premium_economy', 'business', 'first', 'all'] = "economy"
    cabinClassExplicit: bool = False
    compareAllCabins: bool = False
    railClass: Optional[str] = None
    railClassExplicit: bool = False
    hotel: HotelState = Field(default_factory=HotelState)
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    rankingPreference: Literal['cheapest', 'fastest', 'best'] = "cheapest"
    selectedOfferIds: List[str] = Field(default_factory=list)
    searchStatuses: Dict[str, str] = Field(default_factory=lambda: {"flights": "idle", "trains": "idle", "hotels": "idle"})
    lastSearchSummary: Optional[str] = None
    lastAskedQuestion: Optional[str] = None
    lastUserIntent: Optional[str] = None
    interest: Optional[str] = None
    wantsAllClasses: Optional[bool] = False
    activeSearchContextId: Optional[str] = None
    historySlots: Dict[str, int] = Field(default_factory=dict)
    workflowState: WorkflowState = "IDLE"
    selectedFareId: Optional[str] = None
    bookingSessionId: Optional[str] = None
    reviewedFareData: Optional[Dict[str, Any]] = None
    existingBookingRef: Optional[str] = None
    existingAmendmentRef: Optional[str] = None
    distractionPendingSlot: Optional[str] = None
    abandoned: bool = False
    lastOffers: List[Any] = Field(default_factory=list)
    lastSeatMap: Optional[Dict[str, Any]] = None
    lastFareRules: Optional[Dict[str, Any]] = None
    nonstopOnly: Optional[bool] = None
    processedMessages: List[str] = Field(default_factory=list)

    def invalidate_search_state(self) -> None:
        """Clear downstream selection, fare review, seat map, fare rules, and cached offers when route/date/pax/cabin change."""
        self.selectedOfferIds = []
        self.selectedFareId = None
        self.bookingSessionId = None
        self.reviewedFareData = None
        self.lastOffers = []
        self.lastSeatMap = None
        self.lastFareRules = None
        self.activeSearchContextId = None
        if self.workflowState in ("RESULTS_READY", "FLIGHT_SELECTED", "FARE_REVIEW_REQUESTED", "FARE_REVIEWED"):
            self.workflowState = "COLLECTING_SEARCH_DETAILS"


def create_initial_session(session_id: Optional[str] = None) -> TravelSession:
    import uuid
    sid = session_id or str(uuid.uuid4())
    return TravelSession(sessionId=sid)
