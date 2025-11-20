# app/utils/image_utils.py

import cv2
import numpy as np
from typing import List, Tuple, Dict
from scipy.interpolate import splprep, splev

class ImageTransformer:
    """
    Utility class for image transformation operations
    """
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Apply Gaussian blur to smooth an image
        
        Args:
            image: Input image
            kernel_size: Size of blur kernel (must be odd numbers)
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def create_smooth_contour(points: List[Tuple[int, int]], num_points: int = 100) -> np.ndarray:
        """
        Create a smooth contour from a set of points using spline interpolation
        
        This makes our transformations look more natural by creating smooth curves
        instead of jagged lines
        
        Args:
            points: List of (x, y) coordinate tuples
            num_points: Number of points in the smooth contour
            
        Returns:
            Array of smooth contour points
        """
        if len(points) < 3:
            return np.array(points)
        
        points = np.array(points)
        
        # Separate x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        
        # Create spline interpolation
        # k=2 means quadratic spline (smooth curves)
        tck, u = splprep([x, y], s=0, k=min(3, len(points) - 1))
        
        # Evaluate spline at num_points positions
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)
        
        return np.column_stack(smooth_points).astype(np.int32)
    
    @staticmethod
    def create_mask_from_points(image_shape: Tuple[int, int], points: np.ndarray, feather: int = 15) -> np.ndarray:
        """
        Create a mask from contour points with feathered edges
        
        Feathering makes transitions smooth and natural-looking
        
        Args:
            image_shape: Shape of the target image (height, width)
            points: Contour points
            feather: Amount of edge feathering (higher = softer edges)
            
        Returns:
            Mask with values 0-255
        """
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Fill polygon defined by points
        cv2.fillPoly(mask, [points], 255)
        
        # Apply feathering with Gaussian blur
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
        
        return mask
    
    @staticmethod
    def warp_region(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray, 
                   intensity: float = 1.0) -> np.ndarray:
        """
        Warp a region of the image from source points to destination points
        
        This is how we create the "filler" or "lift" effect
        
        Args:
            image: Input image
            src_points: Original position of points
            dst_points: Target position of points
            intensity: How much to warp (0.0 = no warp, 1.0 = full warp)
            
        Returns:
            Warped image
        """
        # Interpolate between source and destination based on intensity
        interpolated_dst = src_points + (dst_points - src_points) * intensity
        
        # Calculate affine transform
        # We need at least 3 points for affine transformation
        if len(src_points) >= 3:
            # Get the transformation matrix
            src_tri = src_points[:3].astype(np.float32)
            dst_tri = interpolated_dst[:3].astype(np.float32)
            
            matrix = cv2.getAffineTransform(src_tri, dst_tri)
            
            # Apply transformation
            warped = cv2.warpAffine(
                image, 
                matrix, 
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            return warped
        
        return image
    
    @staticmethod
    def apply_local_warp(image: np.ndarray, center: Tuple[int, int], radius: int, 
                        strength: float, direction: str = 'expand') -> np.ndarray:
        """
        Apply local warping effect (for fillers and volume)
        
        This creates the "puffing out" effect for fillers
        
        Args:
            image: Input image
            center: Center point (x, y) of the warp
            radius: Radius of effect
            strength: Intensity of warp (0.0 - 1.0)
            direction: 'expand' for fillers, 'contract' for slimming
            
        Returns:
            Warped image
        """
        height, width = image.shape[:2]
        result = image.copy()
        
        # Create coordinate grids
        y_coords, x_coords = np.indices((height, width))
        
        # Calculate distance from center
        dx = x_coords - center[0]
        dy = y_coords - center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Create smooth falloff
        factor = np.zeros_like(distance)
        mask = distance < radius
        factor[mask] = (1 - (distance[mask] / radius)) ** 2
        
        # Apply warp
        if direction == 'expand':
            # Push pixels outward (filler effect)
            scale = 1 + strength * factor * 0.3
        else:
            # Pull pixels inward (slimming effect)
            scale = 1 - strength * factor * 0.2
        
        # Calculate new positions
        new_x = center[0] + dx * scale
        new_y = center[1] + dy * scale
        
        # Clip to image boundaries
        new_x = np.clip(new_x, 0, width - 1).astype(np.float32)
        new_y = np.clip(new_y, 0, height - 1).astype(np.float32)
        
        # Remap pixels
        result = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR)
        
        return result
    
    @staticmethod
    def smooth_skin_region(image: np.ndarray, mask: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Smooth skin in a specific region (for wrinkle/line reduction)
        
        Args:
            image: Input image
            mask: Region mask (0-255)
            intensity: Smoothing intensity (0.0 - 1.0)
            
        Returns:
            Smoothed image
        """
        # Apply bilateral filter for skin smoothing
        # This preserves edges while smoothing texture
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Blend based on intensity and mask
        mask_normalized = (mask / 255.0 * intensity).astype(np.float32)
        mask_3channel = cv2.merge([mask_normalized] * 3)
        
        result = (image * (1 - mask_3channel) + smoothed * mask_3channel).astype(np.uint8)
        
        return result
    
    @staticmethod
    def blend_images(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Blend two images using a mask
        
        Args:
            base: Base image
            overlay: Overlay image
            mask: Blending mask (0-255)
            
        Returns:
            Blended image
        """
        # Normalize mask to 0-1 range
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # Convert to 3 channels
        if len(mask.shape) == 2:
            mask_normalized = cv2.merge([mask_normalized] * 3)
        
        # Blend
        blended = (base * (1 - mask_normalized) + overlay * mask_normalized).astype(np.uint8)
        
        return blended
    
    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, brightness: float = 0, 
                                   contrast: float = 1.0) -> np.ndarray:
        """
        Adjust brightness and contrast of an image
        
        Args:
            image: Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.5 to 2.0)
            
        Returns:
            Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted