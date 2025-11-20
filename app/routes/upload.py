# app/routes/upload.py

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from PIL import Image
import io

from app.config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS, MAX_IMAGE_DIMENSION
from app.models.schemas import UploadResponse

# Create API router
router = APIRouter(prefix="/api/upload", tags=["Upload"])

@router.post("/image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload patient photo
    
    This endpoint:
    1. Validates file type and size
    2. Resizes image if too large
    3. Saves image with unique ID
    4. Returns image URL and ID
    """
    
    try:
        # Step 1: Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Step 2: Read file content
        content = await file.read()
        
        # Step 3: Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Step 4: Open image with PIL
        try:
            image = Image.open(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Step 5: Convert to RGB if needed (removes alpha channel)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 6: Resize if image is too large
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            # Calculate new size maintaining aspect ratio
            if width > height:
                new_width = MAX_IMAGE_DIMENSION
                new_height = int(height * (MAX_IMAGE_DIMENSION / width))
            else:
                new_height = MAX_IMAGE_DIMENSION
                new_width = int(width * (MAX_IMAGE_DIMENSION / height))
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Step 7: Generate unique image ID
        image_id = str(uuid.uuid4())
        
        # Step 8: Save image
        image_filename = f"{image_id}{file_extension}"
        image_path = UPLOAD_DIR / image_filename
        image.save(image_path, quality=95)
        
        # Step 9: Return response
        return UploadResponse(
            success=True,
            message="Image uploaded successfully",
            image_id=image_id,
            image_url=f"/uploads/{image_filename}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")