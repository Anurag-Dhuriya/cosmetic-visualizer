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
    'temples_fillers': facial_transformations.apply_temples_fillers,
    'glabellar_lines': facial_transformations.apply_glabellar_lines_reduction,
    'marionette_folds': facial_transformations.apply_marionette_folds_reduction,
    'root_fillers': facial_transformations.apply_nose_root_fillers,
    'contouring': facial_transformations.apply_nose_contouring,
}

BASE_URL = "http://127.0.0.1:8000"


@router.post("/apply", response_model=TransformationResponse)
async def apply_transformation(request: TransformationRequest):
    """
    Apply a single transformation to an image. Supports fast preview mode
    (downscaled image) when request.preview is True, and full-resolution
    when False.
    """
    try:
        # Validate category and treatment
        if request.category not in TREATMENT_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {list(TREATMENT_CATEGORIES.keys())}"
            )
        if request.treatment not in TREATMENT_CATEGORIES[request.category]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid treatment for category {request.category}"
            )

        # Find uploaded image file
        image_files = list(UPLOAD_DIR.glob(f"{request.image_id}.*"))
        if not image_files:
            raise HTTPException(
                status_code=404,
                detail=f"Image with ID {request.image_id} not found"
            )
        image_path = image_files[0]

        # Use cached landmarks (detect if not cached)
        face_data = face_detector.get_landmarks_cached(str(image_path), request.image_id)
        if face_data is None or not face_data.get('face_detected'):
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Get transformation function
        transformation_func = TRANSFORMATION_MAP.get(request.treatment)
        if transformation_func is None:
            raise HTTPException(
                status_code=501,
                detail=f"Transformation for {request.treatment} not yet implemented"
            )

        preview_mode = bool(request.preview)

        if preview_mode:
            # ------------------ PREVIEW MODE ------------------
            MAX_PREVIEW_DIM = 512

            orig_image = cv2.imread(str(image_path))
            if orig_image is None:
                raise HTTPException(status_code=400, detail="Failed to load image")

            # Original dimensions
            h, w = orig_image.shape[:2]

            # FIX (Problem 3): Calculate new dimensions maintaining aspect ratio
            # checking width and height independently for portrait vs landscape
            if max(h, w) > MAX_PREVIEW_DIM:
                if w >= h:
                    # Landscape or square — width is the longer side
                    new_w = MAX_PREVIEW_DIM
                    new_h = int(h * (MAX_PREVIEW_DIM / w))
                else:
                    # Portrait — height is the longer side
                    new_h = MAX_PREVIEW_DIM
                    new_w = int(w * (MAX_PREVIEW_DIM / h))
                preview_image = cv2.resize(
                    orig_image,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )
            else:
                new_w = w
                new_h = h
                preview_image = orig_image.copy()

            # FIX (Problem 3): Separate scale factors for x and y axes
            scale_x = new_w / w
            scale_y = new_h / h

            # Scale all facial region landmarks using correct axis scale factors
            scaled_facial_regions = {}
            for cat, regions in face_data['facial_regions'].items():
                scaled_facial_regions[cat] = {}
                for region_name, pts in regions.items():
                    scaled_pts = []
                    for p in pts:
                        scaled_pts.append({
                            'x': int(p['x'] * scale_x),
                            'y': int(p['y'] * scale_y),
                            'z': p.get('z', 0)
                        })
                    scaled_facial_regions[cat][region_name] = scaled_pts

            # FIX (Problem 2): Also scale face_boundary landmarks using the
            # same separate scale_x / scale_y so the boundary mask is accurate
            # on the preview image. Without this, the boundary mask would be
            # built at original-image coordinates and clip the warp wrongly.
            raw_boundary = face_data.get('face_boundary', [])
            scaled_face_boundary = []
            for p in raw_boundary:
                scaled_face_boundary.append({
                    'x': int(p['x'] * scale_x),
                    'y': int(p['y'] * scale_y),
                    'z': p.get('z', 0)
                })

            # Build the combined landmarks dict that transformation functions receive.
            # It contains facial_regions AND face_boundary so transformations.py
            # can access both with landmarks.get('face_boundary', None).
            scaled_landmarks = dict(scaled_facial_regions)
            scaled_landmarks['face_boundary'] = scaled_face_boundary

            # Position adjustment uses correct scaled dimensions
            position_adjustment = (
                (request.position_x or 0) * new_w * 0.1,
                (request.position_y or 0) * new_h * 0.1
            )

            transformed_preview = transformation_func(
                preview_image,
                scaled_landmarks,
                request.intensity,
                position_adjustment
            )

            transformation_id = str(uuid.uuid4())
            preview_filename = f"{request.image_id}_{transformation_id}_preview{image_path.suffix}"
            preview_path = TEMP_DIR / preview_filename
            cv2.imwrite(str(preview_path), transformed_preview)

            return TransformationResponse(
                success=True,
                message=f"Preview transformation '{request.treatment}' applied successfully",
                preview_url=f"{BASE_URL}/temp/{preview_filename}",
                transformation_id=transformation_id
            )

        else:
            # ------------------ FULL-RES MODE ------------------
            image = cv2.imread(str(image_path))
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to load image")

            # Build the full-res landmarks dict.
            # face_data['facial_regions'] contains the treatment regions.
            # We also attach face_boundary so transformations can use it.
            full_res_landmarks = dict(face_data['facial_regions'])
            full_res_landmarks['face_boundary'] = face_data.get('face_boundary', [])

            position_adjustment = (
                (request.position_x or 0) * face_data['image_width'] * 0.1,
                (request.position_y or 0) * face_data['image_height'] * 0.1
            )

            transformed_image = transformation_func(
                image,
                full_res_landmarks,
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
                preview_url=f"{BASE_URL}/temp/{preview_filename}",
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
    Apply multiple transformations at once.

    Allows combining treatments sequentially:
    - Chin fillers + Jawline contouring
    - Upper lip + Lower lip + Cupid's bow
    - Cheek fillers + Nasolabial fold reduction
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

        # Step 3: Use cached landmarks
        face_data = face_detector.get_landmarks_cached(str(image_path), image_id)
        if face_data is None or not face_data.get('face_detected'):
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Build landmarks dict including face_boundary
        full_res_landmarks = dict(face_data['facial_regions'])
        full_res_landmarks['face_boundary'] = face_data.get('face_boundary', [])

        # Step 4: Apply each transformation sequentially
        result_image = image.copy()
        applied_transformations = []

        for trans_request in transformations:
            if trans_request.category not in TREATMENT_CATEGORIES:
                continue
            if trans_request.treatment not in TREATMENT_CATEGORIES[trans_request.category]:
                continue

            transformation_func = TRANSFORMATION_MAP.get(trans_request.treatment)
            if transformation_func is None:
                continue

            position_adjustment = (
                trans_request.position_x * face_data['image_width'] * 0.1 if trans_request.position_x else 0,
                trans_request.position_y * face_data['image_height'] * 0.1 if trans_request.position_y else 0
            )

            result_image = transformation_func(
                result_image,
                full_res_landmarks,
                trans_request.intensity,
                position_adjustment
            )

            applied_transformations.append({
                'category': trans_request.category,
                'treatment': trans_request.treatment,
                'intensity': trans_request.intensity,
                'position_x': trans_request.position_x,
                'position_y': trans_request.position_y
            })

        # Step 5: Generate unique ID and save
        transformation_id = str(uuid.uuid4())
        preview_filename = f"{image_id}_{transformation_id}_multi{image_path.suffix}"
        preview_path = TEMP_DIR / preview_filename
        cv2.imwrite(str(preview_path), result_image)

        return {
            "success": True,
            "message": f"Applied {len(applied_transformations)} transformations",
            "preview_url": f"{BASE_URL}/temp/{preview_filename}",
            "transformation_id": transformation_id,
            "applied_transformations": applied_transformations
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multiple transformations failed: {str(e)}")


@router.get("/categories")
async def get_treatment_categories():
    """Get all available treatment categories and their treatments"""
    return {
        "success": True,
        "categories": TREATMENT_CATEGORIES
    }


@router.get("/category/{category_name}")
async def get_treatments_by_category(category_name: str):
    """Get all treatments for a specific category"""
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