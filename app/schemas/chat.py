from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from .flight import FlightOffer, FlightBookingLinks
from .train import TrainOption, TrainLiveStatus, TrainAvailability
from .hotel import HotelOffer


class TripQuestionnaire(BaseModel):
    origin: Optional[str] = None
    originCode: Optional[str] = None
    destination: str
    destinationCode: str


class ChatResults(BaseModel):
    flights: List[FlightOffer] = Field(default_factory=list)
    hotels: List[HotelOffer] = Field(default_factory=list)
    trains: List[TrainOption] = Field(default_factory=list)
    bookingLinks: List[FlightBookingLinks] = Field(default_factory=list)
    trainStatus: List[TrainLiveStatus] = Field(default_factory=list)
    trainAvailability: List[TrainAvailability] = Field(default_factory=list)
    cabinComparisons: Optional[List[Dict[str, Any]]] = None
    cabinNotice: Optional[str] = None
    questionnaire: Optional[TripQuestionnaire] = None
    searchStatuses: Optional[Dict[str, str]] = None
    searchContextId: Optional[str] = None
    dateStatuses: Optional[Dict[str, str]] = None
    fareReview: Optional[Dict[str, Any]] = None
    seatMap: Optional[Dict[str, Any]] = None
    fareRules: Optional[Dict[str, Any]] = None
    bookingLookup: Optional[Dict[str, Any]] = None
    amendmentLookup: Optional[Dict[str, Any]] = None


def empty_results() -> ChatResults:
    return ChatResults()


class ToolCallSummary(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    success: bool
    error: Optional[str] = None
    durationMs: int


class NarrationSection(BaseModel):
    text: str
    flightOfferIds: List[str] = Field(default_factory=list)
    hotelIds: List[str] = Field(default_factory=list)
    trainNumbers: List[str] = Field(default_factory=list)


class MessageRole(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str
    selectedOfferId: Optional[str] = None


ChatMessage = MessageRole


class ChatRequest(BaseModel):
    messages: List[MessageRole]
    sessionId: Optional[str] = None
    modelProvider: Optional[str] = None
    character: Optional[Literal['meera', 'kabir']] = "meera"
    interactiveMode: Optional[bool] = False


class ChatMetadata(BaseModel):
    requestId: str
    model: str
    durationMs: int
    sessionId: Optional[str] = None


class ChatResponse(BaseModel):
    narration: Optional[List[NarrationSection]] = None
    message: MessageRole
    toolCalls: List[ToolCallSummary] = Field(default_factory=list)
    results: ChatResults = Field(default_factory=ChatResults)
    metadata: ChatMetadata
