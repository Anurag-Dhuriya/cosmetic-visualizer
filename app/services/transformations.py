# app/services/transformations.py

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.utils.image_utils import ImageTransformer


class FacialTransformations:
    """
    Contains all transformation functions for cosmetic treatments.

    FIX (Problem 2 — Face Boundary Mask):
    Every function that calls self.transformer.apply_local_warp() now passes
    the face_boundary from the landmarks dict. This ensures all warp effects
    (fillers, lifts, slimming) are clipped to the face shape and never bleed
    into the background, hair, or ears.

    The face_boundary is extracted from landmarks by:
        face_boundary = landmarks.get('face_boundary', None)
    and then passed as the last argument to apply_local_warp().

    apply_local_warp accepts face_boundary as an optional argument, so if it
    is missing for any reason the function still works as before.
    """

    def __init__(self):
        self.transformer = ImageTransformer()

    # ==================== LIPS TRANSFORMATIONS ====================

    def apply_lip_plumper(self, image: np.ndarray, landmarks: Dict,
                          intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Apply lip plumper effect (increases overall lip volume).
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        upper_lip = landmarks['lips']['upper_lip_outer']
        lower_lip = landmarks['lips']['lower_lip_outer']

        all_lip_points = upper_lip + lower_lip
        center_x = int(np.mean([p['x'] for p in all_lip_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in all_lip_points]) + position_adjustment[1])

        lip_width = max([p['x'] for p in all_lip_points]) - min([p['x'] for p in all_lip_points])
        radius = int(lip_width * 0.8)

        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 1.5,
            direction='expand',
            face_boundary=face_boundary
        )

        return result

    def apply_cupids_bow(self, image: np.ndarray, landmarks: Dict,
                         intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Enhance Cupid's bow (the M-shape at the top of upper lip).
        """
        result = image.copy()

        cupids_bow_points = landmarks['lips']['cupids_bow']

        if len(cupids_bow_points) < 3:
            return result

        center_point = cupids_bow_points[len(cupids_bow_points) // 2]
        center_x = int(center_point['x'] + position_adjustment[0])
        center_y = int(center_point['y'] + position_adjustment[1])

        radius = 25 + int(intensity * 20)

        h, w = result.shape[:2]
        y0 = max(0, center_y - radius)
        y1 = min(h, center_y + radius)
        x0 = max(0, center_x - radius)
        x1 = min(w, center_x + radius)

        gy, gx = np.mgrid[y0:y1, x0:x1]
        di = gx - center_x
        dj = gy - center_y
        distance = np.sqrt(di**2 + dj**2)
        inside = distance < radius
        factor = np.where(inside, (1 - distance / radius) * intensity, 0)
        shift_y = (factor * 8).astype(int)
        src_y = np.clip(gy - shift_y, 0, h - 1)
        result[gy, gx] = image[src_y, gx]

        return result

    def apply_upper_lip_fillers(self, image: np.ndarray, landmarks: Dict,
                                intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to upper lip.
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        upper_lip = landmarks['lips']['upper_lip_outer']

        center_x = int(np.mean([p['x'] for p in upper_lip]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in upper_lip]) + position_adjustment[1])

        lip_width = max([p['x'] for p in upper_lip]) - min([p['x'] for p in upper_lip])
        radius = int(lip_width * 0.7)

        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 1.5,
            direction='expand',
            face_boundary=face_boundary
        )

        return result

    def apply_lower_lip_fillers(self, image: np.ndarray, landmarks: Dict,
                                intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to lower lip.
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        lower_lip = landmarks['lips']['lower_lip_outer']

        center_x = int(np.mean([p['x'] for p in lower_lip]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in lower_lip]) + position_adjustment[1])

        lip_width = max([p['x'] for p in lower_lip]) - min([p['x'] for p in lower_lip])
        radius = int(lip_width * 0.7)

        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 1.5,
            direction='expand',
            face_boundary=face_boundary
        )

        return result

    def apply_corner_lip_lift(self, image: np.ndarray, landmarks: Dict,
                              intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Lift corners of mouth (creates subtle smile effect).
        """
        result = image.copy()

        left_corner = landmarks['lips']['lip_corners_left'][0]
        right_corner = landmarks['lips']['lip_corners_right'][0]

        corners = [
            (left_corner['x'] + position_adjustment[0], left_corner['y'] + position_adjustment[1]),
            (right_corner['x'] + position_adjustment[0], right_corner['y'] + position_adjustment[1])
        ]

        for corner_x, corner_y in corners:
            center_x = int(corner_x)
            center_y = int(corner_y)
            radius = 35
            lift_amount = int(intensity * 18)

            h, w = result.shape[:2]
            y0 = max(0, center_y - radius)
            y1 = min(h, center_y + radius)
            x0 = max(0, center_x - radius)
            x1 = min(w, center_x + radius)

            gy, gx = np.mgrid[y0:y1, x0:x1]
            di = gx - center_x
            dj = gy - center_y
            distance = np.sqrt(di**2 + dj**2)
            inside = distance < radius
            factor = np.where(inside, (1 - distance / radius) * intensity, 0)
            shift_y = (factor * lift_amount).astype(int)
            src_y = np.clip(gy - shift_y, 0, h - 1)
            result[gy, gx] = result[src_y, gx]

        return result

    # ==================== NOSE TRANSFORMATIONS ====================

    def apply_nose_bridge_fillers(self, image: np.ndarray, landmarks: Dict,
                                  intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to nose bridge (creates straighter, more defined bridge).
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        bridge_points = landmarks['nose']['bridge']

        center_x = int(np.mean([p['x'] for p in bridge_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in bridge_points]) + position_adjustment[1])

        radius = 40

        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 0.9,
            direction='expand',
            face_boundary=face_boundary
        )

        return result

    def apply_nose_tip_lift(self, image: np.ndarray, landmarks: Dict,
                            intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Lift the nose tip upward.
        """
        result = image.copy()

        tip_points = landmarks['nose']['tip']

        center_x = int(np.mean([p['x'] for p in tip_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in tip_points]) + position_adjustment[1])

        radius = 35
        lift_amount = int(intensity * 22)

        h, w = result.shape[:2]
        y0 = max(0, center_y - radius)
        y1 = min(h, center_y + radius)
        x0 = max(0, center_x - radius)
        x1 = min(w, center_x + radius)

        gy, gx = np.mgrid[y0:y1, x0:x1]
        distance = np.sqrt((gx - center_x)**2 + (gy - center_y)**2)
        factor = np.where(distance < radius, (1 - distance / radius) * intensity, 0)
        shift_y = (factor * lift_amount).astype(int)
        src_y = np.clip(gy - shift_y, 0, h - 1)
        result[gy, gx] = image[src_y, gx]

        return result

    def apply_nose_slimming(self, image: np.ndarray, landmarks: Dict,
                            intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Slim the nose (narrow the width).
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        nose_points = landmarks['nose']['full_nose']

        center_x = int(np.mean([p['x'] for p in nose_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in nose_points]) + position_adjustment[1])

        nose_width = max([p['x'] for p in nose_points]) - min([p['x'] for p in nose_points])
        radius = int(nose_width * 0.8)

        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 1.0,
            direction='contract',
            face_boundary=face_boundary
        )

        return result

    # ==================== EYEBROW TRANSFORMATIONS ====================

    def apply_brow_lift(self, image: np.ndarray, landmarks: Dict,
                        intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Lift eyebrows upward (creates more youthful, alert appearance).
        """
        result = image.copy()

        for brow_key in ['left_eyebrow', 'right_eyebrow']:
            brow_points = landmarks['eyebrows'][brow_key]

            center_x = int(np.mean([p['x'] for p in brow_points]) + position_adjustment[0])
            center_y = int(np.mean([p['y'] for p in brow_points]) + position_adjustment[1])

            brow_width = max([p['x'] for p in brow_points]) - min([p['x'] for p in brow_points])
            radius = int(brow_width * 0.9)
            lift_amount = int(intensity * 30)

            h, w = result.shape[:2]
            y0 = max(0, center_y - radius)
            y1 = min(h, center_y + radius)
            x0 = max(0, center_x - radius)
            x1 = min(w, center_x + radius)

            gy, gx = np.mgrid[y0:y1, x0:x1]
            distance = np.sqrt((gx - center_x)**2 + (gy - center_y)**2)
            factor = np.where(distance < radius, (1 - distance / radius) * intensity, 0)
            shift_y = (factor * lift_amount).astype(int)
            src_y = np.clip(gy - shift_y, 0, h - 1)
            result[gy, gx] = image[src_y, gx]

        return result

    # ==================== FACE TRANSFORMATIONS ====================

    def apply_cheek_fillers(self, image: np.ndarray, landmarks: Dict,
                            intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to cheeks (enhances cheekbones).
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        for cheek_key in ['cheeks_left', 'cheeks_right']:
            cheek_points = landmarks['face'][cheek_key]

            center_x = int(np.mean([p['x'] for p in cheek_points]) + position_adjustment[0])
            center_y = int(np.mean([p['y'] for p in cheek_points]) + position_adjustment[1])

            cheek_width = max([p['x'] for p in cheek_points]) - min([p['x'] for p in cheek_points])
            radius = int(cheek_width * 1.2)

            result = self.transformer.apply_local_warp(
                result,
                center=(center_x, center_y),
                radius=radius,
                strength=intensity * 0.9,
                direction='expand',
                face_boundary=face_boundary
            )

        return result

    def apply_chin_fillers(self, image: np.ndarray, landmarks: Dict,
                           intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to chin (creates more defined, projected chin).
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        chin_points = landmarks['face']['chin']

        center_x = int(np.mean([p['x'] for p in chin_points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in chin_points]) + position_adjustment[1])

        radius = 65

        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=intensity * 1.0,
            direction='expand',
            face_boundary=face_boundary
        )

        return result

    def apply_jawline_contouring(self, image: np.ndarray, landmarks: Dict,
                                 intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Define and sharpen jawline.
        """
        result = image.copy()

        for jaw_key in ['jawline_left', 'jawline_right']:
            jaw_points = landmarks['face'][jaw_key]

            points = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1]))
                      for p in jaw_points]

            smooth_contour = self.transformer.create_smooth_contour(points, num_points=50)

            mask = self.transformer.create_mask_from_points(
                (result.shape[0], result.shape[1]),
                smooth_contour,
                feather=30
            )

            darkened = self.transformer.adjust_brightness_contrast(
                result,
                brightness=-int(intensity * 40),
                contrast=1.0 + intensity * 0.3
            )

            result = self.transformer.blend_images(result, darkened, mask)

        return result

    def apply_forehead_lines_reduction(self, image: np.ndarray, landmarks: Dict,
                                       intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Smooth forehead wrinkles/lines.
        """
        result = image.copy()

        forehead_points = landmarks['face']['forehead']

        points = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1]))
                  for p in forehead_points]

        smooth_contour = self.transformer.create_smooth_contour(points, num_points=100)
        mask = self.transformer.create_mask_from_points(
            (result.shape[0], result.shape[1]),
            smooth_contour,
            feather=30
        )

        result = self.transformer.smooth_skin_region(result, mask, intensity)

        return result

    def apply_nasolabial_folds_reduction(self, image: np.ndarray, landmarks: Dict,
                                         intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Reduce appearance of smile lines (nasolabial folds).
        """
        result = image.copy()

        for fold_key in ['nasolabial_left', 'nasolabial_right']:
            fold_points = landmarks['face'][fold_key]

            points = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1]))
                      for p in fold_points]

            smooth_contour = self.transformer.create_smooth_contour(points, num_points=30)
            mask = self.transformer.create_mask_from_points(
                (result.shape[0], result.shape[1]),
                smooth_contour,
                feather=15
            )

            result = self.transformer.smooth_skin_region(result, mask, intensity)

        return result

    # ==================== ADDITIONAL FACE TRANSFORMATIONS ====================

    def apply_temples_fillers(self, image: np.ndarray, landmarks: Dict,
                              intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume to the temple areas (left & right).
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        for key in ['temples_left', 'temples_right']:
            points = landmarks['face'].get(key, [])
            if not points:
                continue

            center_x = int(np.mean([p['x'] for p in points]) + position_adjustment[0])
            center_y = int(np.mean([p['y'] for p in points]) + position_adjustment[1])

            temple_width = max([p['x'] for p in points]) - min([p['x'] for p in points]) if points else 40
            radius = int(max(40, temple_width * 1.2))

            result = self.transformer.apply_local_warp(
                result,
                center=(center_x, center_y),
                radius=radius,
                strength=float(intensity) * 0.8,
                direction='expand',
                face_boundary=face_boundary
            )

        return result

    def apply_glabellar_lines_reduction(self, image: np.ndarray, landmarks: Dict,
                                        intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Reduce glabellar (frown) lines between the eyebrows.
        """
        result = image.copy()

        points = landmarks['face'].get('glabella', [])
        if not points:
            return result

        pts = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1])) for p in points]

        if len(pts) < 3:
            if pts:
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
            else:
                cx = image.shape[1] // 2
                cy = image.shape[0] // 2
            pts = [(cx - 20, cy - 10), (cx, cy + 10), (cx + 20, cy - 10)]

        smooth_contour = self.transformer.create_smooth_contour(pts, num_points=80)
        mask = self.transformer.create_mask_from_points(
            (result.shape[0], result.shape[1]),
            smooth_contour,
            feather=25
        )

        result = self.transformer.smooth_skin_region(result, mask, intensity=float(intensity))

        return result

    def apply_marionette_folds_reduction(self, image: np.ndarray, landmarks: Dict,
                                         intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Reduce marionette folds (lines from mouth corners down).
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        for key in ['marionette_left', 'marionette_right']:
            pts = landmarks['face'].get(key, [])
            if not pts:
                continue

            points = [(int(p['x'] + position_adjustment[0]), int(p['y'] + position_adjustment[1])) for p in pts]
            smooth_contour = self.transformer.create_smooth_contour(points, num_points=40)
            mask = self.transformer.create_mask_from_points(
                (result.shape[0], result.shape[1]),
                smooth_contour,
                feather=22
            )

            result = self.transformer.smooth_skin_region(result, mask, intensity=float(intensity))

            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))
            radius = int(max(30, (max([p[0] for p in points]) - min([p[0] for p in points])) * 1.0))

            result = self.transformer.apply_local_warp(
                result,
                center=(center_x, center_y),
                radius=radius,
                strength=float(intensity) * 0.6,
                direction='contract',
                face_boundary=face_boundary
            )

        return result

    # ==================== ADDITIONAL NOSE TRANSFORMATIONS ====================

    def apply_nose_root_fillers(self, image: np.ndarray, landmarks: Dict,
                                intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Add volume at the nose root (between the eyes).
        """
        result = image.copy()
        face_boundary = landmarks.get('face_boundary', None)

        points = landmarks['nose'].get('root', [])
        if not points:
            return result

        center_x = int(np.mean([p['x'] for p in points]) + position_adjustment[0])
        center_y = int(np.mean([p['y'] for p in points]) + position_adjustment[1])

        radius = 28 + int(intensity * 18)

        result = self.transformer.apply_local_warp(
            result,
            center=(center_x, center_y),
            radius=radius,
            strength=float(intensity) * 0.9,
            direction='expand',
            face_boundary=face_boundary
        )

        return result

    def apply_nose_contouring(self, image: np.ndarray, landmarks: Dict,
                              intensity: float, position_adjustment: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Contour the nose by darkening the sides (creates illusion of slimmer bridge).
        """
        result = image.copy()

        nose_pts = landmarks['nose'].get('full_nose', [])
        if not nose_pts:
            return result

        xs = [p['x'] for p in nose_pts]
        ys = [p['y'] for p in nose_pts]
        cx = int(np.mean(xs) + position_adjustment[0])
        cy = int(np.mean(ys) + position_adjustment[1])

        nose_width = max(xs) - min(xs) if xs else 40
        height = max(20, int((max(ys) - min(ys)) * 1.2))

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

        mask_left = cv2.GaussianBlur(mask_left, (31, 31), 0)
        mask_right = cv2.GaussianBlur(mask_right, (31, 31), 0)

        darkened = self.transformer.adjust_brightness_contrast(
            result,
            brightness=-20 - int(intensity * 25),
            contrast=1.0
        )

        blend_strength = float(intensity) * 1.0
        result = self.transformer.blend_images(result, darkened, (mask_left * blend_strength).astype(np.uint8))
        result = self.transformer.blend_images(result, darkened, (mask_right * blend_strength).astype(np.uint8))

        return result