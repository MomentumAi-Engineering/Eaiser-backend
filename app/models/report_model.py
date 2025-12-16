from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class Location(BaseModel):
    latitude: float
    longitude: float
    streetAddress: Optional[str] = None
    zipCode: Optional[str] = None

class FlagFlowDetails(BaseModel):
    userNotifiedOfFlag: bool = False
    sentToTeam: bool = False
    reason: Optional[str] = None
    finalUserNotification: Optional[str] = None

class RealFlowDetails(BaseModel):
    category: Optional[str] = None
    modelUsed: Optional[str] = None
    kmProcessed: bool = False
    reportContent: Optional[Dict[str, Any]] = None

class ReportModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    imageUrl: str
    originalName: Optional[str] = None
    status: str = "pending"  # pending, flagged, declined, waiting_review, submitted, approved
    location: Location
    checkResult: Optional[Dict[str, Any]] = None
    flagFlowDetails: Optional[FlagFlowDetails] = None
    realFlowDetails: Optional[RealFlowDetails] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}
