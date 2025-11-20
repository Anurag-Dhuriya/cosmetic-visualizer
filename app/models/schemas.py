# app/models/schemas.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class UploadResponse(BaseModel):
    """Response after uploading an image"""
    success: bool
    message: str
    image_id: str
    image_url: str

class TransformationRequest(BaseModel):
    """Request to apply a transformation"""
    image_id: str = Field(..., description="ID of the uploaded image")
    category: str = Field(..., description="Category: face, lips, nose, eyebrows")
    treatment: str = Field(..., description="Specific treatment name")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Intensity slider (0-100%)")
    position_x: Optional[float] = Field(None, ge=-1.0, le=1.0, description="X position adjustment")
    position_y: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Y position adjustment")
    preview: bool = Field(True, description="If true, apply transform on low-res preview for speed")

class TransformationResponse(BaseModel):
    """Response after transformation"""
    success: bool
    message: str
    preview_url: str
    transformation_id: str

class SaveRequest(BaseModel):
    """Request to save final result"""
    image_id: str
    transformations: List[Dict] = Field(..., description="List of all applied transformations")

class SaveResponse(BaseModel):
    """Response after saving"""
    success: bool
    message: str
    output_url: str
    metadata: Dict
    