from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")

class AdminBase(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    role: str = "admin"  # admin, super_admin, viewer

class AdminCreate(AdminBase):
    password: str

class AdminInDB(AdminBase):
    id: Optional[str] = Field(alias="_id")
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    assigned_issues: List[str] = []  # List of issue IDs assigned to this admin
    permissions: dict = {
        "can_approve": True,
        "can_decline": True,
        "can_assign": False,  # Only super_admin can assign
        "can_manage_team": False  # Only super_admin
    }
    last_login: Optional[datetime] = None
    
    # Security fields
    require_password_change: bool = False
    password_last_changed: Optional[datetime] = None
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    two_factor_method: Optional[str] = None # 'email' or 'totp'
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Token(BaseModel):
    access_token: str
    token_type: str
    admin: dict
