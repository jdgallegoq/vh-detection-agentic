from typing import Optional
from enum import Enum
from langgraph.graph import MessagesState

class VehicleTypeEnum(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    OTHER = "other"

class AgentState(MessagesState):
    b64_image: Optional[str] = None  # base64 encoded image
    is_vehicle: bool = False
    vehicle_type: Optional[VehicleTypeEnum] = None
    review: Optional[str] = None
