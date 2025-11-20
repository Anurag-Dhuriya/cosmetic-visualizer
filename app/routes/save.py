# app/routes/save.py

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pathlib import Path
import uuid
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from app.config import UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR
from app.models.schemas import SaveRequest, SaveResponse

# Create router
router = APIRouter(prefix="/api/save", tags=["Save"])

@router.post("/result", response_model=SaveResponse)
async def save_transformation_result(request: SaveRequest):
    """
    Save the final before/after comparison image with treatment details
    
    This endpoint:
    1. Loads original (before) image
    2. Loads final transformed (after) image
    3. Creates side-by-side comparison
    4. Adds treatment details as text overlay
    5. Saves to outputs folder
    6. Saves metadata JSON file
    """
    
    try:
        # Step 1: Find original image
        original_files = list(UPLOAD_DIR.glob(f"{request.image_id}.*"))
        
        if not original_files:
            raise HTTPException(
                status_code=404,
                detail=f"Original image with ID {request.image_id} not found"
            )
        
        original_path = original_files[0]
        
        # Step 2: Load original image
        original_image = cv2.imread(str(original_path))
        if original_image is None:
            raise HTTPException(status_code=400, detail="Failed to load original image")
        
        # Step 3: Find latest transformed image (from temp folder)
        # The frontend should provide the latest transformation_id
        # For now, we'll use the most recent temp file for this image_id
        temp_files = sorted(
            TEMP_DIR.glob(f"{request.image_id}_*.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not temp_files:
            raise HTTPException(
                status_code=404,
                detail="No transformed image found. Please apply transformations first."
            )
        
        transformed_path = temp_files[0]
        
        # Step 4: Load transformed image
        transformed_image = cv2.imread(str(transformed_path))
        if transformed_image is None:
            raise HTTPException(status_code=400, detail="Failed to load transformed image")
        
        # Step 5: Create side-by-side comparison
        comparison_image = create_before_after_comparison(
            original_image,
            transformed_image,
            request.transformations
        )
        
        # Step 6: Generate output filename
        output_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"result_{timestamp}_{output_id}.jpg"
        output_path = OUTPUT_DIR / output_filename
        
        # Step 7: Save comparison image
        cv2.imwrite(str(output_path), comparison_image)
        
        # Step 8: Create and save metadata
        metadata = {
            "output_id": output_id,
            "original_image_id": request.image_id,
            "timestamp": timestamp,
            "transformations_applied": request.transformations,
            "output_filename": output_filename
        }
        
        metadata_path = OUTPUT_DIR / f"result_{timestamp}_{output_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Step 9: Return response
        return SaveResponse(
            success=True,
            message="Result saved successfully",
            output_url=f"/outputs/{output_filename}",
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save result: {str(e)}"
        )


def create_before_after_comparison(
    before_image: np.ndarray,
    after_image: np.ndarray,
    transformations: list
) -> np.ndarray:
    """
    Create a side-by-side before/after comparison image with details
    
    Layout:
    - Left half: Original image with "BEFORE" label
    - Right half: Transformed image with "AFTER" label
    - Bottom: List of applied transformations
    
    Args:
        before_image: Original image
        after_image: Transformed image
        transformations: List of applied transformations
        
    Returns:
        Combined comparison image
    """
    
    # Get dimensions
    height, width = before_image.shape[:2]
    
    # Create canvas for side-by-side comparison
    # Add extra space at bottom for text
    text_height = 200
    canvas_height = height + text_height
    canvas_width = width * 2 + 40  # 40px gap in middle
    
    # Create white canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Place BEFORE image on left
    canvas[0:height, 0:width] = before_image
    
    # Place AFTER image on right
    canvas[0:height, width + 40:canvas_width] = after_image
    
    # Add vertical divider line
    cv2.line(
        canvas,
        (width + 20, 0),
        (width + 20, height),
        (200, 200, 200),
        2
    )
    
    # Convert to PIL for text rendering (easier than OpenCV)
    canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas_pil)
    
    # Try to use a nicer font, fallback to default if not available
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    
    # Add "BEFORE" label
    draw.text(
        (width // 2 - 80, height + 20),
        "BEFORE",
        fill=(0, 0, 0),
        font=title_font
    )
    
    # Add "AFTER" label
    draw.text(
        (width + width // 2 - 60, height + 20),
        "AFTER",
        fill=(0, 0, 0),
        font=title_font
    )
    
    # Add transformation details
    y_position = height + 80
    draw.text(
        (20, y_position),
        "Applied Treatments:",
        fill=(0, 0, 0),
        font=text_font
    )
    
    y_position += 35
    
    for i, trans in enumerate(transformations[:5]):  # Show max 5 treatments
        treatment_name = trans.get('treatment', '').replace('_', ' ').title()
        category = trans.get('category', '').title()
        intensity = trans.get('intensity', 0) * 100
        
        text = f"• {category}: {treatment_name} (Intensity: {intensity:.0f}%)"
        draw.text(
            (30, y_position),
            text,
            fill=(60, 60, 60),
            font=text_font
        )
        y_position += 30
    
    if len(transformations) > 5:
        draw.text(
            (30, y_position),
            f"... and {len(transformations) - 5} more",
            fill=(100, 100, 100),
            font=text_font
        )
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text(
        (canvas_width - 250, canvas_height - 30),
        timestamp,
        fill=(150, 150, 150),
        font=text_font
    )
    
    # Convert back to OpenCV format
    result = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
    
    return result


@router.get("/history")
async def get_saved_results():
    """
    Get list of all saved results
    
    Returns:
        List of saved result metadata
    """
    try:
        # Find all JSON metadata files
        metadata_files = list(OUTPUT_DIR.glob("result_*.json"))
        
        results = []
        for metadata_file in sorted(metadata_files, reverse=True):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                results.append(metadata)
        
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.delete("/result/{output_id}")
async def delete_saved_result(output_id: str):
    """
    Delete a saved result
    
    Args:
        output_id: ID of the result to delete
    """
    try:
        # Find files with this output_id
        image_files = list(OUTPUT_DIR.glob(f"result_*_{output_id}.jpg"))
        metadata_files = list(OUTPUT_DIR.glob(f"result_*_{output_id}.json"))
        
        if not image_files and not metadata_files:
            raise HTTPException(
                status_code=404,
                detail=f"Result with ID {output_id} not found"
            )
        
        # Delete files
        for file in image_files + metadata_files:
            file.unlink()
        
        return {
            "success": True,
            "message": f"Result {output_id} deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete result: {str(e)}"
        )