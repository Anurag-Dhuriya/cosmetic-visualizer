# app/routes/transform.py

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pathlib import Path
import uuid

from app.config import UPLOAD_DIR, TEMP_DIR, TREATMENT_CATEGORIES
from app.services.face_detector import FaceDetector
from app.services.transformations import FacialTransformations
from app.models.schemas import TransformationRequest, TransformationResponse

# Create router
router = APIRouter(prefix="/api/transform", tags=["Transform"])

# Initialize services
face_detector = FaceDetector()
facial_transformations = FacialTransformations()

# Map treatment names to transformation functions
TRANSFORMATION_MAP = {
    # Lips
    'plumper': facial_transformations.apply_lip_plumper,
    'cupids_bow': facial_transformations.apply_cupids_bow,
    'upper_lip_fillers': facial_transformations.apply_upper_lip_fillers,
    'lower_lip_fillers': facial_transformations.apply_lower_lip_fillers,
    'corner_lip_lift_fillers': facial_transformations.apply_corner_lip_lift,
    
    # Nose
    'bridge_fillers': facial_transformations.apply_nose_bridge_fillers,
    'tip_lift_fillers': facial_transformations.apply_nose_tip_lift,
    'slimming_fillers': facial_transformations.apply_nose_slimming,
    
    # Eyebrows
    'brow_lift': facial_transformations.apply_brow_lift,
    
    # Face
    'cheek_fillers': facial_transformations.apply_cheek_fillers,
    'chin_fillers': facial_transformations.apply_chin_fillers,
    'jawline_contouring': facial_transformations.apply_jawline_contouring,
    'forehead_lines': facial_transformations.apply_forehead_lines_reduction,
    'nasolabial_folds': facial_transformations.apply_nasolabial_folds_reduction,

        # New mappings (add to existing TRANSFORMATION_MAP)
    'temples_fillers': facial_transformations.apply_temples_fillers,
    'glabellar_lines': facial_transformations.apply_glabellar_lines_reduction,
    'marionette_folds': facial_transformations.apply_marionette_folds_reduction,
    'root_fillers': facial_transformations.apply_nose_root_fillers,
    'contouring': facial_transformations.apply_nose_contouring,

}

@router.post("/apply", response_model=TransformationResponse)
async def apply_transformation(request: TransformationRequest):
    """
    Apply a single transformation to an image. Supports fast preview mode (downscaled image)
    when request.preview is True, and full-resolution when False.
    """
    try:
        # Validate category and treatment as you already do
        if request.category not in TREATMENT_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {list(TREATMENT_CATEGORIES.keys())}")
        if request.treatment not in TREATMENT_CATEGORIES[request.category]:
            raise HTTPException(status_code=400, detail=f"Invalid treatment for category {request.category}")

        # Find uploaded image file
        image_files = list(UPLOAD_DIR.glob(f"{request.image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail=f"Image with ID {request.image_id} not found")
        image_path = image_files[0]

        # Use cached landmarks (detect if not cached)
        face_data = face_detector.get_landmarks_cached(str(image_path), request.image_id)
        if face_data is None or not face_data.get('face_detected'):
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Get transformation function
        transformation_func = TRANSFORMATION_MAP.get(request.treatment)
        if transformation_func is None:
            raise HTTPException(status_code=501, detail=f"Transformation for {request.treatment} not yet implemented")

        # Decide preview vs full resolution
        preview_mode = bool(request.preview)

        if preview_mode:
            # ------------------ PREVIEW MODE ------------------
            MAX_PREVIEW_DIM = 512  # max width/height for preview
            orig_image = cv2.imread(str(image_path))
            if orig_image is None:
                raise HTTPException(status_code=400, detail="Failed to load image")

            h, w = orig_image.shape[:2]
            scale = 1.0
            # If image is larger than MAX_PREVIEW_DIM, scale it down preserving aspect ratio
            if max(h, w) > MAX_PREVIEW_DIM:
                scale = MAX_PREVIEW_DIM / max(h, w)
                preview_image = cv2.resize(orig_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                preview_image = orig_image.copy()

            # Scale facial regions (landmarks) to preview coordinates
            scaled_facial_regions = {}
            for cat, regions in face_data['facial_regions'].items():
                scaled_facial_regions[cat] = {}
                for region_name, pts in regions.items():
                    scaled_pts = []
                    for p in pts:
                        scaled_pts.append({
                            'x': int(p['x'] * scale),
                            'y': int(p['y'] * scale),
                            'z': p.get('z', 0)
                        })
                    scaled_facial_regions[cat][region_name] = scaled_pts

            # Convert position_x / position_y into pixels for preview image
            position_adjustment = (
                (request.position_x or 0) * preview_image.shape[1] * 0.1,
                (request.position_y or 0) * preview_image.shape[0] * 0.1
            )

            # Apply transformation on the preview image using scaled landmarks
            transformed_preview = transformation_func(
                preview_image,
                scaled_facial_regions,
                request.intensity,
                position_adjustment
            )

            # Save preview to temp folder
            transformation_id = str(uuid.uuid4())
            preview_filename = f"{request.image_id}_{transformation_id}_preview{image_path.suffix}"
            preview_path = TEMP_DIR / preview_filename
            cv2.imwrite(str(preview_path), transformed_preview)

            return TransformationResponse(
                success=True,
                message=f"Preview transformation '{request.treatment}' applied successfully",
                preview_url=f"/temp/{preview_filename}",
                transformation_id=transformation_id
            )

        else:
            # ------------------ FULL-RES MODE ------------------
            image = cv2.imread(str(image_path))
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to load image")

            position_adjustment = (
                (request.position_x or 0) * face_data['image_width'] * 0.1,
                (request.position_y or 0) * face_data['image_height'] * 0.1
            )

            transformed_image = transformation_func(
                image,
                face_data['facial_regions'],
                request.intensity,
                position_adjustment
            )

            transformation_id = str(uuid.uuid4())
            preview_filename = f"{request.image_id}_{transformation_id}{image_path.suffix}"
            preview_path = TEMP_DIR / preview_filename
            cv2.imwrite(str(preview_path), transformed_image)

            return TransformationResponse(
                success=True,
                message=f"Transformation '{request.treatment}' applied successfully",
                preview_url=f"/temp/{preview_filename}",
                transformation_id=transformation_id
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")




@router.post("/apply-multiple")
async def apply_multiple_transformations(
    image_id: str,
    transformations: list[TransformationRequest]
):
    """
    Apply multiple transformations at once
    
    This allows doctors to combine treatments:
    - Chin fillers + Jawline contouring
    - Upper lip + Lower lip + Cupid's bow
    - Cheek fillers + Nasolabial fold reduction
    
    The transformations are applied sequentially
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
        
        # Step 2: Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Step 3: Detect face once
        face_data = face_detector.detect_face(str(image_path))
        
        if face_data is None or not face_data['face_detected']:
            raise HTTPException(
                status_code=400,
                detail="No face detected in image"
            )
        
        # Step 4: Apply each transformation sequentially
        result_image = image.copy()
        applied_transformations = []
        
        for trans_request in transformations:
            # Validate
            if trans_request.category not in TREATMENT_CATEGORIES:
                continue
            
            if trans_request.treatment not in TREATMENT_CATEGORIES[trans_request.category]:
                continue
            
            # Get transformation function
            transformation_func = TRANSFORMATION_MAP.get(trans_request.treatment)
            
            if transformation_func is None:
                continue
            
            # Prepare position adjustment
            position_adjustment = (
                trans_request.position_x * face_data['image_width'] * 0.1 if trans_request.position_x else 0,
                trans_request.position_y * face_data['image_height'] * 0.1 if trans_request.position_y else 0
            )
            
            # Apply transformation to current result
            result_image = transformation_func(
                result_image,
                face_data['facial_regions'],
                trans_request.intensity,
                position_adjustment
            )
            
            # Track what was applied
            applied_transformations.append({
                'category': trans_request.category,
                'treatment': trans_request.treatment,
                'intensity': trans_request.intensity,
                'position_x': trans_request.position_x,
                'position_y': trans_request.position_y
            })
        
        # Step 5: Generate unique ID for this combination
        transformation_id = str(uuid.uuid4())
        
        # Step 6: Save preview
        preview_filename = f"{image_id}_{transformation_id}_multi{image_path.suffix}"
        preview_path = TEMP_DIR / preview_filename
        
        cv2.imwrite(str(preview_path), result_image)
        
        # Step 7: Return response
        return {
            "success": True,
            "message": f"Applied {len(applied_transformations)} transformations",
            "preview_url": f"/temp/{preview_filename}",
            "transformation_id": transformation_id,
            "applied_transformations": applied_transformations
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multiple transformations failed: {str(e)}"
        )


@router.get("/categories")
async def get_treatment_categories():
    """
    Get all available treatment categories and their treatments
    
    Returns the complete list of what treatments are available
    """
    return {
        "success": True,
        "categories": TREATMENT_CATEGORIES
    }


@router.get("/category/{category_name}")
async def get_treatments_by_category(category_name: str):
    """
    Get all treatments for a specific category
    
    Args:
        category_name: face, lips, nose, or eyebrows
    """
    if category_name not in TREATMENT_CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category_name}' not found"
        )
    
    return {
        "success": True,
        "category": category_name,
        "treatments": TREATMENT_CATEGORIES[category_name]
    }