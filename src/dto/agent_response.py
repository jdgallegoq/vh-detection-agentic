from pydantic import BaseModel

class VerifyImageResponse(BaseModel):
    is_vehicle: bool

class VehicleTypeResponse(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    OTHER = "other"

class ReviewImageResponse(BaseModel):
    review: str
