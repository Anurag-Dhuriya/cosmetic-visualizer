# app/services/transformations.py

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.utils.image_utils import ImageTransformer

class FacialTransformations:
    """
    Contains all transformation functions for cosmetic treatments
    """
    
    def __init__(self):
        self.transformer = ImageTransformer()
    
    # ==================== LIPS TRANSFORMATIONS ====================
    
    def apply_lip_plumper(self, image: np.ndarray, landmarks: Dict, 
                         intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Apply lip plumper effect (increases overall lip volume)
        
        Args:
            image: Input image
            landmarks: Facial landmarks dictionary
            intensity: Effect strength (0.0 - 1.0)
            position_adjustment: (x, y) adjustment for manual positioning
            
        Returns:
            Transformed image
        """
        result = image.copy()
        
        # Get lip region landmarks
        upper_lip = landmarks['lips']['upper_lip_outer']
        lower_lip = landmarks['lips']['lower_lip_outer']
        
        # Calculate center of lips
        all_lip_points = upper_lip + lower_lip
        center_x = int(np.mean([p['x'] for p in all_lip_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in all_lip_points]) + position_adjustment[1])
        
        # Calculate radius based on lip size
        lip_width = max([p['x'] for p in all_lip_points]) - min([p['x'] for p in all_lip_points])
        radius = int(lip_width * 0.6)
        
        # Apply local warp to add volume
        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity,
            direction='expand'
        )
        
        return result
    
    def apply_cupids_bow(self, image: np.ndarray, landmarks: Dict, 
                        intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Enhance Cupid's bow (the M-shape at the top of upper lip)
        
        Args:
            image: Input image
            landmarks: Facial landmarks dictionary
            intensity: Effect strength (0.0 - 1.0)
            position_adjustment: (x, y) adjustment
            
        Returns:
            Transformed image
        """
        result = image.copy()
        
        # Get Cupid's bow landmarks
        cupids_bow_points = landmarks['lips']['cupids_bow']
        
        if len(cupids_bow_points) < 3:
            return result
        
        # Get center point (the dip in the middle)
        center_point = cupids_bow_points[len(cupids_bow_points) // 2]
        center_x = int(center_point['x'] + position_adjustment[0])
        center_y = int(center_point['y'] + position_adjustment[1])
        
        # Apply small upward warp to accentuate the bow
        radius = 15 + int(intensity * 10)
        
        # Create upward pull effect
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                x, y = center_x + i, center_y + j
                
                if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                    distance = np.sqrt(i**2 + j**2)
                    
                    if distance < radius:
                        factor = (1 - distance / radius) * intensity
                        # Pull slightly upward to enhance the peak
                        shift_y = int(factor * 3)
                        
                        if 0 <= y - shift_y < result.shape[0]:
                            result[y, x] = image[y - shift_y, x]
        
        return result
    
    def apply_upper_lip_fillers(self, image: np.ndarray, landmarks: Dict, 
                               intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to upper lip
        
        Args:
            image: Input image
            landmarks: Facial landmarks dictionary
            intensity: Effect strength (0.0 - 1.0)
            position_adjustment: (x, y) adjustment
            
        Returns:
            Transformed image
        """
        result = image.copy()
        
        # Get upper lip landmarks
        upper_lip = landmarks['lips']['upper_lip_outer']
        
        # Calculate center of upper lip
        center_x = int(np.mean([p['x'] for p in upper_lip]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in upper_lip]) + position_adjustment[1])
        
        # Calculate radius
        lip_width = max([p['x'] for p in upper_lip]) - min([p['x'] for p in upper_lip])
        radius = int(lip_width * 0.5)
        
        # Apply expansion
        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity,
            direction='expand'
        )
        
        return result
    
    def apply_lower_lip_fillers(self, image: np.ndarray, landmarks: Dict, 
                               intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to lower lip
        """
        result = image.copy()
        
        lower_lip = landmarks['lips']['lower_lip_outer']
        
        center_x = int(np.mean([p['x'] for p in lower_lip]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in lower_lip]) + position_adjustment[1])
        
        lip_width = max([p['x'] for p in lower_lip]) - min([p['x'] for p in lower_lip])
        radius = int(lip_width * 0.5)
        
        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity,
            direction='expand'
        )
        
        return result
    
    def apply_corner_lip_lift(self, image: np.ndarray, landmarks: Dict, 
                             intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Lift corners of mouth (creates subtle smile effect)
        """
        result = image.copy()
        height = image.shape[0]
        
        # Get left and right lip corners
        left_corner = landmarks['lips']['lip_corners_left'][0]
        right_corner = landmarks['lips']['lip_corners_right'][0]
        
        corners = [
            (left_corner['x'] + position_adjustment[0], left_corner['y'] + position_adjustment[1]),
            (right_corner['x'] + position_adjustment[0], right_corner['y'] + position_adjustment[1])
        ]
        
        # Lift each corner upward
        for corner_x, corner_y in corners:
            center_x = int(corner_x)
            center_y = int(corner_y)
            radius = 20
            
            # Calculate upward shift
            lift_amount = int(intensity * 8)
            
            # Apply upward warp
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    x, y = center_x + i, center_y + j
                    
                    if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                        distance = np.sqrt(i**2 + j**2)
                        
                        if distance < radius:
                            factor = (1 - distance / radius) * intensity
                            shift_y = int(factor * lift_amount)
                            
                            if 0 <= y - shift_y < result.shape[0]:
                                result[y, x] = image[y - shift_y, x]
        
        return result
    
    # ==================== NOSE TRANSFORMATIONS ====================
    
    def apply_nose_bridge_fillers(self, image: np.ndarray, landmarks: Dict, 
                                  intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to nose bridge (creates straighter, more defined bridge)
        """
        result = image.copy()
        
        bridge_points = landmarks['nose']['bridge']
        
        # Calculate center of bridge
        center_x = int(np.mean([p['x'] for p in bridge_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in bridge_points]) + position_adjustment[1])
        
        radius = 25
        
        # Apply expansion to bridge area
        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 0.5,  # Subtle effect
            direction='expand'
        )
        
        return result
    
    def apply_nose_tip_lift(self, image: np.ndarray, landmarks: Dict, 
                           intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Lift the nose tip upward
        """
        result = image.copy()
        
        tip_points = landmarks['nose']['tip']
        
        center_x = int(np.mean([p['x'] for p in tip_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in tip_points]) + position_adjustment[1])
        
        radius = 20
        lift_amount = int(intensity * 10)
        
        # Apply upward warp
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                x, y = center_x + i, center_y + j
                
                if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                    distance = np.sqrt(i**2 + j**2)
                    
                    if distance < radius:
                        factor = (1 - distance / radius) * intensity
                        shift_y = int(factor * lift_amount)
                        
                        if 0 <= y - shift_y < result.shape[0]:
                            result[y, x] = image[y - shift_y, x]
        
        return result
    
    def apply_nose_slimming(self, image: np.ndarray, landmarks: Dict, 
                           intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Slim the nose (narrow the width)
        """
        result = image.copy()
        
        nose_points = landmarks['nose']['full_nose']
        
        center_x = int(np.mean([p['x'] for p in nose_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in nose_points]) + position_adjustment[1])
        
        # Calculate nose width
        nose_width = max([p['x'] for p in nose_points]) - min([p['x'] for p in nose_points])
        radius = int(nose_width * 0.6)
        
        # Apply contraction horizontally
        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 0.7,
            direction='contract'
        )
        
        return result
    
    # ==================== EYEBROW TRANSFORMATIONS ====================
    
    def apply_brow_lift(self, image: np.ndarray, landmarks: Dict, 
                       intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Lift eyebrows upward (creates more youthful, alert appearance)
        """
        result = image.copy()
        
        # Process both eyebrows
        for brow_key in ['left_eyebrow', 'right_eyebrow']:
            brow_points = landmarks['eyebrows'][brow_key]
            
            # Calculate center of eyebrow
            center_x = int(np.mean([p['x'] for p in brow_points]) + position_adjustment[0])
            center_y = int(np.mean([p['y'] for p in brow_points]) + position_adjustment[1])
            
            # Calculate brow width
            brow_width = max([p['x'] for p in brow_points]) - min([p['x'] for p in brow_points])
            radius = int(brow_width * 0.6)
            
            lift_amount = int(intensity * 12)
            
            # Apply upward warp
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    x, y = center_x + i, center_y + j
                    
                    if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                        distance = np.sqrt(i**2 + j**2)
                        
                        if distance < radius:
                            factor = (1 - distance / radius) * intensity
                            shift_y = int(factor * lift_amount)
                            
                            if 0 <= y - shift_y < result.shape[0]:
                                result[y, x] = image[y - shift_y, x]
        
        return result
    
    # ==================== FACE TRANSFORMATIONS ====================
    
    def apply_cheek_fillers(self, image: np.ndarray, landmarks: Dict, 
                           intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to cheeks (enhances cheekbones)
        """
        result = image.copy()
        
        # Process both cheeks
        for cheek_key in ['cheeks_left', 'cheeks_right']:
            cheek_points = landmarks['face'][cheek_key]
            
            center_x = int(np.mean([p['x'] for p in cheek_points]) + position_adjustment[0])
            center_y = int(np.mean([p['y'] for p in cheek_points]) + position_adjustment[1])
            
            # Calculate cheek area size
            cheek_width = max([p['x'] for p in cheek_points]) - min([p['x'] for p in cheek_points])
            radius = int(cheek_width * 0.8)
            
            # Apply expansion
            result = self.transformer.apply_local_warp(
                result,
                center=(center_x, center_y),
                radius=radius,
                strength=intensity * 0.6,
                direction='expand'
            )
        
        return result
    
    def apply_chin_fillers(self, image: np.ndarray, landmarks: Dict, 
                          intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to chin (creates more defined, projected chin)
        """
        result = image.copy()
        
        chin_points = landmarks['face']['chin']
        
        center_x = int(np.mean([p['x'] for p in chin_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in chin_points]) + position_adjustment[1])
        
        radius = 40
        
        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 0.7,
            direction='expand'
        )
        
        return result
    
    def apply_jawline_contouring(self, image: np.ndarray, landmarks: Dict, 
                                intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Define and sharpen jawline
        """
        result = image.copy()
        
        # Process both sides of jawline
        for jaw_key in ['jawline_left', 'jawline_right']:
            jaw_points = landmarks['face'][jaw_key]
            
            # Create contour along jawline
            points = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1])) 
                     for p in jaw_points]
            
            # Apply subtle sharpening along the jawline
            smooth_contour = self.transformer.create_smooth_contour(points, num_points=50)
            
            # Create mask for jawline area
            mask = self.transformer.create_mask_from_points(
                (result.shape[0], result.shape[1]),
                smooth_contour,
                feather=20
            )
            
            # Apply subtle darkening/sharpening for definition
            darkened = self.transformer.adjust_brightness_contrast(
                result,
                brightness=-int(intensity * 10),
                contrast=1.0 + intensity * 0.1
            )
            
            # Blend
            result = self.transformer.blend_images(result, darkened, mask)
        
        return result
    
    def apply_forehead_lines_reduction(self, image: np.ndarray, landmarks: Dict, 
                                      intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Smooth forehead wrinkles/lines
        """
        result = image.copy()
        
        forehead_points = landmarks['face']['forehead']
        
        # Create points for mask
        points = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1])) 
                 for p in forehead_points]
        
        # Create forehead mask
        smooth_contour = self.transformer.create_smooth_contour(points, num_points=100)
        mask = self.transformer.create_mask_from_points(
            (result.shape[0], result.shape[1]),
            smooth_contour,
            feather=30
        )
        
        # Apply skin smoothing
        result = self.transformer.smooth_skin_region(result, mask, intensity)
        
        return result
    
    def apply_nasolabial_folds_reduction(self, image: np.ndarray, landmarks: Dict, 
                                        intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Reduce appearance of smile lines (nasolabial folds)
        """
        result = image.copy()
        
        # Process both sides
        for fold_key in ['nasolabial_left', 'nasolabial_right']:
            fold_points = landmarks['face'][fold_key]
            
            points = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1])) 
                     for p in fold_points]
            
            # Create mask for fold area
            smooth_contour = self.transformer.create_smooth_contour(points, num_points=30)
            mask = self.transformer.create_mask_from_points(
                (result.shape[0], result.shape[1]),
                smooth_contour,
                feather=15
            )
            
            # Apply smoothing
            result = self.transformer.smooth_skin_region(result, mask, intensity)
        
        return result
    
        # -------------------- ADDITIONAL FACE TRANSFORMATIONS --------------------

    def apply_temples_fillers(self, image: np.ndarray, landmarks: Dict,
                              intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add subtle volume to the temple areas (left & right).
        Uses local expand warp centered on the temple regions.
        """
        result = image.copy()

        # Left & right temple landmarks (from face detector mapping)
        for key in ['temples_left', 'temples_right']:
            points = landmarks['face'].get(key, [])
            if not points:
                continue

            center_x = int(np.mean([p['x'] for p in points]) + position_adjustment[0])
            center_y = int(np.mean([p['y'] for p in points]) + position_adjustment[1])

            # Estimate size using spread of points
            temple_width = max([p['x'] for p in points]) - min([p['x'] for p in points]) if points else 40
            radius = int(max(30, temple_width * 0.8))

            # Use smaller strength so temples look natural
            result = self.transformer.apply_local_warp(
                result,
                center=(center_x, center_y),
                radius=radius,
                strength=float(intensity) * 0.5,
                direction='expand'
            )

        return result


    def apply_glabellar_lines_reduction(self, image: np.ndarray, landmarks: Dict,
                                        intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Reduce glabellar (frown) lines between the eyebrows.
        We'll create a small mask over the glabella and smooth it.
        """
        result = image.copy()

        points = landmarks['face'].get('glabella', [])
        if not points:
            return result

        # Convert landmarks to (x,y) tuples and adjust positions
        pts = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1])) for p in points]

        # If only a couple points exist, expand them into a small region
        if len(pts) < 3:
            cx = int(np.mean([p['x'] for p in landmarks['landmarks']]))
            cy = int(np.mean([p['y'] for p in landmarks['landmarks']]))
            pts = [(cx - 10, cy - 5), (cx, cy + 5), (cx + 10, cy - 5)]

        smooth_contour = self.transformer.create_smooth_contour(pts, num_points=80)
        mask = self.transformer.create_mask_from_points((result.shape[0], result.shape[1]), smooth_contour, feather=18)

        # Slightly increase radius of smoothing depending on intensity
        result = self.transformer.smooth_skin_region(result, mask, intensity=float(intensity))

        return result


    def apply_marionette_folds_reduction(self, image: np.ndarray, landmarks: Dict,
                                         intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Reduce marionette folds (lines from mouth corners down).
        We'll smooth the region and slightly lift pixels inward to reduce depth.
        """
        result = image.copy()

        for key in ['marionette_left', 'marionette_right']:
            pts = landmarks['face'].get(key, [])
            if not pts:
                continue

            points = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1])) for p in pts]
            smooth_contour = self.transformer.create_smooth_contour(points, num_points=40)
            mask = self.transformer.create_mask_from_points((result.shape[0], result.shape[1]), smooth_contour, feather=16)

            # First smooth skin texture in the area
            result = self.transformer.smooth_skin_region(result, mask, intensity=float(intensity))

            # Then apply a small inward warp to reduce apparent fold depth
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))
            radius = int(max(20, (max([p[0] for p in points]) - min([p[0] for p in points])) * 0.8))

            # Contract (pull pixels inward) slightly — smaller strength for subtlety
            result = self.transformer.apply_local_warp(
                result,
                center=(center_x, center_y),
                radius=radius,
                strength=float(intensity) * 0.35,
                direction='contract'
            )

        return result


    # -------------------- ADDITIONAL NOSE TRANSFORMATIONS --------------------

    def apply_nose_root_fillers(self, image: np.ndarray, landmarks: Dict,
                                intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add subtle volume at the nose root (between the eyes) to smooth transition.
        Uses expand warp with small radius.
        """
        result = image.copy()

        points = landmarks['nose'].get('root', [])
        if not points:
            return result

        center_x = int(np.mean([p['x'] for p in points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in points]) + position_adjustment[1])

        radius = 18 + int(intensity * 10)

        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=float(intensity) * 0.6,
            direction='expand'
        )

        return result


    def apply_nose_contouring(self, image: np.ndarray, landmarks: Dict,
                              intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Contour the nose by subtly darkening the sides (creates illusion of slimmer or more defined bridge).
        We build side masks and blend a slightly darkened version.
        """
        result = image.copy()

        # Use full_nose points to determine sides
        nose_pts = landmarks['nose'].get('full_nose', [])
        if not nose_pts:
            return result

        xs = [p['x'] for p in nose_pts]
        ys = [p['y'] for p in nose_pts]
        cx = int(np.mean(xs) + position_adjustment[0])
        cy = int(np.mean(ys) + position_adjustment[1])

        nose_width = max(xs) - min(xs) if xs else 40
        height = max(20, int((max(ys) - min(ys)) * 1.2))

        # Create left and right side masks
        left_poly = np.array([
            (min(xs) - int(nose_width * 0.2), cy - height // 2),
            (cx - int(nose_width * 0.1), cy - height // 2),
            (cx - int(nose_width * 0.1), cy + height // 2),
            (min(xs) - int(nose_width * 0.2), cy + height // 2)
        ], dtype=np.int32)

        right_poly = np.array([
            (cx + int(nose_width * 0.1), cy - height // 2),
            (max(xs) + int(nose_width * 0.2), cy - height // 2),
            (max(xs) + int(nose_width * 0.2), cy + height // 2),
            (cx + int(nose_width * 0.1), cy + height // 2)
        ], dtype=np.int32)

        h, w = result.shape[:2]
        mask_left = np.zeros((h, w), dtype=np.uint8)
        mask_right = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_left, [left_poly], 255)
        cv2.fillPoly(mask_right, [right_poly], 255)

        # Feather both masks
        mask_left = cv2.GaussianBlur(mask_left, (31, 31), 0)
        mask_right = cv2.GaussianBlur(mask_right, (31, 31), 0)

        # Darken underlying image slightly for contouring
        darkened = self.transformer.adjust_brightness_contrast(result, brightness=-10 - int(intensity * 10), contrast=1.0)
        # Blend darkened into left and right using masks (intensity controls strength)
        blend_strength = float(intensity) * 0.7
        result = self.transformer.blend_images(result, darkened, (mask_left * blend_strength).astype(np.uint8))
        result = self.transformer.blend_images(result, darkened, (mask_right * blend_strength).astype(np.uint8))

        return result