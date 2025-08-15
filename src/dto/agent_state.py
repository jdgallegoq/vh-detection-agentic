from typing import Optional, Enum
from langgraph.graph import MessagesState

class VehicleType(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    OTHER = "other"

class AgentState(MessagesState):
    b64_image: Optional[str] = None  # base64 encoded image
    is_vehicle: bool = False
    vehicle_type: Optional[VehicleType] = None
    review: Optional[str] = None
