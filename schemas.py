from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr


class Box(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float

    # Populated only when /detect/faces is called with ?debug=1. Omitted
    # entirely (not just null) for regular requests via response_model_exclude_none,
    # so existing clients that ignore unknown fields are unaffected either way.
    matched_face_id: Optional[int] = None
    matched_face_name: Optional[str] = None
    match_score: Optional[float] = None
    match_threshold: Optional[float] = None
    is_unknown: Optional[bool] = None
    reason: Optional[str] = None
    raw_similarity: Optional[float] = None


class DetectionResponse(BaseModel):
    boxes: List[Box]


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    created_at: datetime


class AuthRegister(BaseModel):
    name: str
    email: EmailStr
    password: str


class AuthLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


class FaceItem(BaseModel):
    id: int
    name: str
    image_path: Optional[str]
    created_at: datetime


class ObjectItem(BaseModel):
    id: int
    name: str
    category: Optional[str]
    image_path: Optional[str]
    created_at: datetime


class LogItem(BaseModel):
    id: int
    kind: str
    label: str
    confidence: Optional[float]
    created_at: datetime
