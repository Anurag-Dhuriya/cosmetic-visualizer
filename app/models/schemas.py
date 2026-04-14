# app/models/schemas.py

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class FaceQualityWarning(BaseModel):
    """
    Describes a single face quality issue found during upload.

    FIX (Problem 7 — Face Quality Check):
    When a photo fails a quality check, instead of just returning a plain
    error string we return a structured warning object. This gives the
    frontend enough information to show the user:
      - what exactly went wrong (code)
      - a human-readable explanation (message)
      - a specific suggestion on how to fix it (suggestion)

    Codes used:
      - 'face_too_small'   : face covers less than 8% of image area
      - 'face_cropped'     : face bounding box touches the image edge
      - 'no_face_detected' : MediaPipe found no face in the image
    """
    code: str                # machine-readable identifier e.g. 'face_too_small'
    message: str             # human-readable description of the problem
    suggestion: str          # specific advice on how to fix the photo


class UploadResponse(BaseModel):
    """
    Response after uploading an image.

    FIX (Problem 7 — Face Quality Check):
    Two new optional fields have been added:

    face_quality_passed (bool):
        True  = face passed all quality checks, image was saved, image_id is valid
        False = face failed one or more checks, image was NOT saved, image_id is empty

    face_quality_warnings (list):
        Empty when face_quality_passed is True.
        Contains one FaceQualityWarning per failed check when False.
        The frontend should display these warnings to the user.

    The existing fields (success, message, image_id, image_url) are unchanged
    so any existing code that reads them continues to work.
    """
    success: bool
    message: str
    image_id: str
    image_url: str

    # New fields for face quality feedback
    face_quality_passed: bool = True
    face_quality_warnings: List[FaceQualityWarning] = []


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