from pydantic import BaseModel
from enum import Enum

class VerifyImageResponse(BaseModel):
    is_vehicle: bool

class VehicleTypeEnum(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    OTHER = "other"

class VehicleTypeResponse(BaseModel):
    vehicle_type: VehicleTypeEnum

class ReviewImageResponse(BaseModel):
    review: str
