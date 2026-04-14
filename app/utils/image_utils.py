# app/utils/image_utils.py

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import splprep, splev


class ImageTransformer:
    """
    Utility class for image transformation operations.
    """

    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Apply Gaussian blur to smooth an image.

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
        Create a smooth contour from a set of points using spline interpolation.

        This makes transformations look more natural by creating smooth curves
        instead of jagged lines.

        Args:
            points: List of (x, y) coordinate tuples
            num_points: Number of points in the smooth contour

        Returns:
            Array of smooth contour points
        """
        if len(points) < 3:
            return np.array(points)

        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        tck, u = splprep([x, y], s=0, k=min(3, len(points) - 1))
        u_new = np.linspace(0, 1, num_points)
        smooth_points = splev(u_new, tck)

        return np.column_stack(smooth_points).astype(np.int32)

    @staticmethod
    def create_mask_from_points(image_shape: Tuple[int, int], points: np.ndarray, feather: int = 15) -> np.ndarray:
        """
        Create a mask from contour points with feathered edges.

        Feathering makes transitions smooth and natural-looking.

        Args:
            image_shape: Shape of the target image (height, width)
            points: Contour points
            feather: Amount of edge feathering (higher = softer edges)

        Returns:
            Mask with values 0-255
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        if feather > 0:
            mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)

        return mask

    @staticmethod
    def create_face_boundary_mask(
        image_shape: Tuple[int, int],
        face_boundary: List[Dict],
        feather: int = 20
    ) -> np.ndarray:
        """
        Create a soft mask of the face shape from the face oval boundary points.

        FIX (Problem 2 — Face Boundary Mask):
        The old apply_local_warp had no awareness of where the face ends.
        It expanded pixels outward in a circle, which could bleed past the
        face edge into the background, hair, or ears.

        This method takes the 36 face oval landmark points from face_detector.py
        and draws a filled polygon in the exact shape of the face. The edges are
        feathered (blurred) so the boundary fades softly rather than cutting hard.

        This mask is then multiplied with the warp effect in apply_local_warp
        so that only pixels INSIDE the face boundary are moved. Pixels outside
        the face are never touched.

        Args:
            image_shape  : (height, width) of the image
            face_boundary: List of {x, y, z} dicts tracing the face oval
                           (36 points from face_detector._extract_face_boundary)
            feather      : How softly to fade the boundary edges (pixels).
                           Higher = softer, more natural blend at face edge.

        Returns:
            2D numpy array (height x width), values 0-255.
            255 = fully inside face, 0 = fully outside face,
            values in between = soft feathered edge zone.
        """
        # Build a numpy array of (x, y) integer points from the boundary dicts
        pts = np.array(
            [[p['x'], p['y']] for p in face_boundary],
            dtype=np.int32
        )

        # Create empty mask same size as the image
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Fill the face-shaped polygon with white (255)
        cv2.fillPoly(mask, [pts], 255)

        # Feather (blur) the edges so the boundary fades softly
        # kernel size must be odd — feather*2+1 guarantees this
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)

        return mask

    @staticmethod
    def warp_region(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray,
                    intensity: float = 1.0) -> np.ndarray:
        """
        Warp a region of the image from source points to destination points.

        Args:
            image: Input image
            src_points: Original position of points
            dst_points: Target position of points
            intensity: How much to warp (0.0 = no warp, 1.0 = full warp)

        Returns:
            Warped image
        """
        interpolated_dst = src_points + (dst_points - src_points) * intensity

        if len(src_points) >= 3:
            src_tri = src_points[:3].astype(np.float32)
            dst_tri = interpolated_dst[:3].astype(np.float32)

            matrix = cv2.getAffineTransform(src_tri, dst_tri)

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
    def apply_local_warp(
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        strength: float,
        direction: str = 'expand',
        face_boundary: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Apply local warping effect (for fillers and volume).

        FIX (Problem 2 — Face Boundary Mask):
        A new optional parameter `face_boundary` has been added.

        OLD behaviour: The warp expanded pixels outward in a perfect circle
        with no knowledge of the face shape. On cheek fillers for example,
        the warp could push pixels past the face edge into the background
        or ear area, creating visible artifacts.

        NEW behaviour: If face_boundary is provided, we call
        create_face_boundary_mask() to build a face-shaped mask. The warp
        is computed as before, but then we blend it back onto the original
        image using this face mask. Pixels outside the face boundary keep
        their original values — they are never moved. Pixels inside the
        face are warped normally. The feathered mask edge ensures the
        transition between warped and original is smooth and invisible.

        The face_boundary parameter is optional so this function remains
        fully backwards compatible — existing calls without face_boundary
        work exactly as before.

        Args:
            image        : Input image
            center       : Center point (x, y) of the warp
            radius       : Radius of effect in pixels
            strength     : Intensity of warp (0.0 - 1.0)
            direction    : 'expand' for fillers, 'contract' for slimming
            face_boundary: Optional list of {x,y,z} dicts tracing the face
                           oval (from face_detector._extract_face_boundary).
                           When provided, warp is clipped to face shape.

        Returns:
            Warped image with effect contained within face boundary
        """
        height, width = image.shape[:2]

        # ── Step 1: Compute the warp (same as original) ────────────────────
        y_coords, x_coords = np.indices((height, width))

        dx = x_coords - center[0]
        dy = y_coords - center[1]
        distance = np.sqrt(dx**2 + dy**2)

        factor = np.zeros_like(distance)
        mask_circle = distance < radius
        factor[mask_circle] = (1 - (distance[mask_circle] / radius)) ** 2

        if direction == 'expand':
            scale = 1 + strength * factor * 0.3
        else:
            scale = 1 - strength * factor * 0.2

        new_x = center[0] + dx * scale
        new_y = center[1] + dy * scale

        new_x = np.clip(new_x, 0, width - 1).astype(np.float32)
        new_y = np.clip(new_y, 0, height - 1).astype(np.float32)

        warped = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR)

        # ── Step 2: Apply face boundary mask if provided ───────────────────
        if face_boundary is not None and len(face_boundary) >= 3:
            # Build the face-shaped mask (white inside face, black outside)
            face_mask = ImageTransformer.create_face_boundary_mask(
                image_shape=(height, width),
                face_boundary=face_boundary,
                feather=20
            )

            # Normalise mask to 0.0–1.0 float for blending
            face_mask_f = face_mask.astype(np.float32) / 255.0

            # Expand to 3 channels (one per colour channel)
            face_mask_3ch = cv2.merge([face_mask_f, face_mask_f, face_mask_f])

            # Blend: inside face → use warped pixel
            #        outside face → use original pixel
            result = (
                warped.astype(np.float32) * face_mask_3ch +
                image.astype(np.float32) * (1.0 - face_mask_3ch)
            ).astype(np.uint8)

            return result

        # No face_boundary provided — return warp as-is (backwards compatible)
        return warped

    @staticmethod
    def smooth_skin_region(image: np.ndarray, mask: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Smooth skin in a specific region (for wrinkle/line reduction).

        FIX (Problem 5 — LAB Colour Smoothing):
        Converts to LAB colour space and applies bilateral filter ONLY on the
        L (lightness) channel. The A and B channels (skin colour) are left
        completely untouched, preserving skin tone at all intensity levels.

        Args:
            image    : Input BGR image
            mask     : Region mask (0-255), white = area to smooth
            intensity: Smoothing intensity (0.0 - 1.0)

        Returns:
            Image with smoothed skin in masked region, colour fully preserved
        """
        # Convert BGR → LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Smooth ONLY the L channel (texture, not colour)
        l_smoothed = cv2.bilateralFilter(l_channel, d=9, sigmaColor=75, sigmaSpace=75)

        # Merge smoothed L with original A and B
        lab_smoothed = cv2.merge([l_smoothed, a_channel, b_channel])

        # Convert LAB → BGR
        smoothed = cv2.cvtColor(lab_smoothed, cv2.COLOR_LAB2BGR)

        # Blend using the mask
        mask_normalized = (mask / 255.0 * intensity).astype(np.float32)
        mask_3channel = cv2.merge([mask_normalized, mask_normalized, mask_normalized])

        result = (
            image.astype(np.float32) * (1 - mask_3channel) +
            smoothed.astype(np.float32) * mask_3channel
        ).astype(np.uint8)

        return result

    @staticmethod
    def blend_images(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Blend two images using a mask.

        Args:
            base   : Base image
            overlay: Overlay image
            mask   : Blending mask (0-255)

        Returns:
            Blended image
        """
        mask_normalized = mask.astype(np.float32) / 255.0

        if len(mask.shape) == 2:
            mask_normalized = cv2.merge([mask_normalized] * 3)

        blended = (base * (1 - mask_normalized) + overlay * mask_normalized).astype(np.uint8)
        return blended

    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, brightness: float = 0,
                                   contrast: float = 1.0) -> np.ndarray:
        """
        Adjust brightness and contrast of an image.

        Args:
            image     : Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast  : Contrast multiplier (0.5 to 2.0)

        Returns:
            Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted