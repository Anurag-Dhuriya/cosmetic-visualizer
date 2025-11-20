# app/routes/analysis.py

from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.config import UPLOAD_DIR, TEMP_DIR
from app.models.schemas import UploadResponse
from app.services.face_detector import FaceDetector
from pydantic import BaseModel

# Create router
router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

# Initialize face detector (reuse same instance)
face_detector = FaceDetector()

class AnalysisResponse(BaseModel):
    """Response from face analysis"""
    success: bool
    message: str
    face_detected: bool
    landmarks_count: int
    facial_regions: dict
    image_dimensions: dict

@router.post("/detect-face", response_model=AnalysisResponse)
async def analyze_face(image_id: str):
    """
    Analyze uploaded image and detect facial landmarks
    
    This endpoint:
    1. Finds the uploaded image by ID
    2. Detects face and landmarks
    3. Returns facial regions for all treatment categories
    
    Args:
        image_id: The unique ID from upload response
    """
    
    try:
        # Step 1: Find uploaded image
        image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
        
        if not image_files:
            raise HTTPException(
                status_code=404,
                detail=f"Image with ID {image_id} not found"
            )
        
        image_path = image_files[0]
        
        # Step 2: Detect face and get landmarks
        face_data = face_detector.detect_face(str(image_path))
        
        if face_data is None:
            return AnalysisResponse(
                success=False,
                message="No face detected in the image",
                face_detected=False,
                landmarks_count=0,
                facial_regions={},
                image_dimensions={}
            )
        
        # Step 3: Return analysis results
        return AnalysisResponse(
            success=True,
            message="Face detected successfully",
            face_detected=True,
            landmarks_count=len(face_data['landmarks']),
            facial_regions=face_data['facial_regions'],
            image_dimensions={
                'width': face_data['image_width'],
                'height': face_data['image_height']
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Face detection failed: {str(e)}"
        )

@router.post("/visualize-landmarks")
async def visualize_face_landmarks(image_id: str):
    """
    Create a visualization of detected landmarks (for debugging)
    
    This draws all 468 landmarks on the face so you can see
    what MediaPipe detected
    """
    
    try:
        # Find uploaded image
        image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
        
        if not image_files:
            raise HTTPException(
                status_code=404,
                detail=f"Image with ID {image_id} not found"
            )
        
        image_path = image_files[0]
        
        # Create output path for visualization
        output_path = TEMP_DIR / f"{image_id}_landmarks{image_path.suffix}"
        
        # Generate visualization
        success = face_detector.visualize_landmarks(
            str(image_path),
            str(output_path)
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to visualize landmarks"
            )
        
        return {
            "success": True,
            "message": "Landmarks visualized successfully",
            "visualization_url": f"/temp/{output_path.name}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Visualization failed: {str(e)}"
        )