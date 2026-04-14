# app/routes/upload.py

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from PIL import Image
import io

from app.config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS, MAX_IMAGE_DIMENSION
from app.models.schemas import UploadResponse, FaceQualityWarning
from app.services.face_detector import FaceDetector

# Create API router
router = APIRouter(prefix="/api/upload", tags=["Upload"])

# Single shared FaceDetector instance
# We reuse the same instance so MediaPipe is only loaded once
face_detector = FaceDetector()


@router.post("/image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload patient photo.

    This endpoint:
    1.  Validates file type and size
    2.  Opens and converts the image with PIL
    3.  Resizes image if too large
    4.  Saves image temporarily with a unique ID
    5.  Runs face quality checks (Problem 7 fix):
          - Is a face present?
          - Is the face large enough to work with?
          - Is the face fully within the frame (not cropped)?
    6a. If quality checks PASS  → keep the saved file, return success
    6b. If quality checks FAIL  → delete the saved file, return warnings
        so the frontend can show the user exactly what is wrong and how
        to fix their photo. The image_id returned will be an empty string
        and face_quality_passed will be False.

    FIX (Problem 7 — Face Quality Check):
    Previously the image was saved and returned to the user with no face
    validation. The user would only discover problems later when clicking
    "Preview", at which point the error was confusing and generic.

    Now the check happens at upload time, before the image_id is returned
    to the frontend. The frontend receives structured warnings with clear
    messages and suggestions, and no image_id is issued for bad photos.
    """

    # ── Step 1: Validate file extension ───────────────────────────────────
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # ── Step 2: Read file content ──────────────────────────────────────────
    content = await file.read()

    # ── Step 3: Check file size ────────────────────────────────────────────
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE // (1024 * 1024)} MB."
        )

    # ── Step 4: Open image with PIL ────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not open it.")

    # ── Step 5: Convert to RGB (removes alpha channel if present) ──────────
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # ── Step 6: Resize if image is too large ──────────────────────────────
    width, height = image.size
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        if width > height:
            new_width = MAX_IMAGE_DIMENSION
            new_height = int(height * (MAX_IMAGE_DIMENSION / width))
        else:
            new_height = MAX_IMAGE_DIMENSION
            new_width = int(width * (MAX_IMAGE_DIMENSION / height))

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # ── Step 7: Save image temporarily so face detector can read it ────────
    # We save first because MediaPipe needs a file path, not PIL image bytes.
    # If quality checks fail we delete this file before returning.
    image_id = str(uuid.uuid4())
    image_filename = f"{image_id}{file_extension}"
    image_path = UPLOAD_DIR / image_filename

    try:
        image.save(image_path, quality=95)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save image. Please try again.")

    # ── Step 8: Run face quality checks ───────────────────────────────────
    # FIX (Problem 7): This is the new step. We call check_face_quality()
    # which runs three checks:
    #   1. Was a face detected at all?
    #   2. Is the face large enough (>= 8% of image area)?
    #   3. Is the face fully within the frame (not within 3% of any edge)?
    #
    # If any check fails, we delete the saved file (so uploads/ stays clean)
    # and return a structured response with face_quality_passed=False and
    # a list of warnings explaining what is wrong and how to fix it.
    #
    # If all checks pass, we keep the saved file and return success normally.
    try:
        quality_result = face_detector.check_face_quality(str(image_path))
    except Exception:
        # If the quality check itself errors, clean up and fail safely
        if image_path.exists():
            image_path.unlink()
        raise HTTPException(
            status_code=500,
            detail="Face quality check failed unexpectedly. Please try again."
        )

    if not quality_result['passed']:
        # Quality check failed — delete the saved file so we don't accumulate
        # bad uploads in the uploads/ folder
        if image_path.exists():
            image_path.unlink()

        # Convert raw warning dicts to FaceQualityWarning schema objects
        warnings = [
            FaceQualityWarning(
                code=w['code'],
                message=w['message'],
                suggestion=w['suggestion']
            )
            for w in quality_result['warnings']
        ]

        # Return a non-error response (HTTP 200) with quality failure details.
        # We use HTTP 200 rather than 4xx so the frontend can read the
        # structured warnings easily without parsing an exception body.
        # The frontend should check face_quality_passed to know if the
        # upload succeeded.
        return UploadResponse(
            success=False,
            message="Photo did not pass face quality checks. See warnings for details.",
            image_id="",          # no valid image_id — nothing was saved
            image_url="",         # no URL — nothing was saved
            face_quality_passed=False,
            face_quality_warnings=warnings
        )

    # ── Step 9: Quality passed — return success ────────────────────────────
    return UploadResponse(
        success=True,
        message="Image uploaded successfully.",
        image_id=image_id,
        image_url=f"/uploads/{image_filename}",
        face_quality_passed=True,
        face_quality_warnings=[]
    )