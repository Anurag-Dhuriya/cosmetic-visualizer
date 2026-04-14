# app/services/face_detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import time


class FaceDetector:
    def __init__(self, cache_ttl: int = 300):
        """
        Initialize MediaPipe Face Mesh and a simple in-memory cache.

        cache_ttl: number of seconds a cached result is considered valid
                   (default 300s = 5 minutes)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # _cache stores { image_id: { 'timestamp': float, 'data': face_data_dict } }
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl

    # ==================== CACHE METHODS ====================

    def _is_cache_valid(self, image_id: str) -> bool:
        """Return True if cache entry exists and is not expired."""
        entry = self._cache.get(image_id)
        if not entry:
            return False
        return (time.time() - entry['timestamp']) < self.cache_ttl

    def get_landmarks_cached(self, image_path: str, image_id: str) -> Optional[Dict]:
        """
        Return cached face_data for image_id if valid, otherwise run
        detection, cache the result, and return it.
        """
        if self._is_cache_valid(image_id):
            return self._cache[image_id]['data']

        face_data = self.detect_face(image_path)
        if face_data:
            self._cache[image_id] = {'timestamp': time.time(), 'data': face_data}
        return face_data

    def invalidate_cache(self, image_id: str):
        """Remove cached entry for image_id (call this if image changed)."""
        if image_id in self._cache:
            del self._cache[image_id]

    # ==================== FACE QUALITY CHECK ====================

    def check_face_quality(self, image_path: str) -> Dict:
        """
        Run face quality checks on an uploaded image BEFORE saving it.

        FIX (Problem 7 — Face Quality Check):
        Previously, upload.py saved every image immediately without checking
        if the face was usable. Errors only appeared later at transform time,
        with confusing messages.

        This method runs three checks right at upload time:

        CHECK 1 — Face detected:
            MediaPipe must find exactly one face. If it finds zero, we return
            a 'no_face_detected' warning immediately.

        CHECK 2 — Face large enough:
            The face bounding box area must cover at least MIN_FACE_AREA_RATIO
            (8%) of the total image area. If the face is smaller than this,
            landmark positions are unreliable — they cover too few pixels for
            accurate region targeting.

            Example: A 1920x1080 image has area = 2,073,600 px.
            8% = 165,888 px. A face bounding box smaller than roughly
            407x407 px would fail this check.

        CHECK 3 — Face not cropped at edge:
            The face bounding box must not come within MIN_EDGE_MARGIN_RATIO
            (3%) of any image edge. If it does, part of the face is outside
            the frame. Treatments near the cropped area will have missing
            landmarks and warp incorrectly.

            Example: In a 1920px wide image, the margin is 1920 * 0.03 = 57px.
            If the face bounding box starts at x=20 it fails this check.

        Args:
            image_path: Path to the image file to check

        Returns:
            Dict with keys:
              'passed'   (bool) : True if all checks passed
              'warnings' (list) : List of dicts with 'code', 'message',
                                  'suggestion' — one per failed check
        """

        # Thresholds — tuned to reject genuinely bad photos without being
        # too strict for normal selfies and clinic photos
        MIN_FACE_AREA_RATIO   = 0.08  # face must be at least 8% of image area
        MIN_EDGE_MARGIN_RATIO = 0.03  # face must not be within 3% of any edge

        warnings = []

        # ── Step 1: Read image ─────────────────────────────────────────────
        image = cv2.imread(str(image_path))
        if image is None:
            warnings.append({
                'code': 'no_face_detected',
                'message': 'Could not read the image file.',
                'suggestion': 'Please upload a valid JPG or PNG photo.'
            })
            return {'passed': False, 'warnings': warnings}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        image_area = img_h * img_w

        # ── Step 2: Run MediaPipe detection ────────────────────────────────
        results = self.face_mesh.process(image_rgb)

        # CHECK 1: Was a face detected at all?
        if not results.multi_face_landmarks:
            warnings.append({
                'code': 'no_face_detected',
                'message': 'No face was detected in this photo.',
                'suggestion': (
                    'Please upload a clear, well-lit photo with your face '
                    'centred and fully visible. Avoid sunglasses, masks, or '
                    'heavy shadows across the face.'
                )
            })
            return {'passed': False, 'warnings': warnings}

        # Get the detected face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Convert normalised landmarks to pixel coordinates
        landmarks_px = []
        for lm in face_landmarks.landmark:
            landmarks_px.append({
                'x': lm.x * img_w,
                'y': lm.y * img_h
            })

        # Calculate the face bounding box from all landmark positions
        x_coords = [p['x'] for p in landmarks_px]
        y_coords = [p['y'] for p in landmarks_px]

        face_x_min = min(x_coords)
        face_x_max = max(x_coords)
        face_y_min = min(y_coords)
        face_y_max = max(y_coords)

        face_w = face_x_max - face_x_min
        face_h = face_y_max - face_y_min
        face_area = face_w * face_h

        # CHECK 2: Is the face large enough?
        face_area_ratio = face_area / image_area
        if face_area_ratio < MIN_FACE_AREA_RATIO:
            face_percent = round(face_area_ratio * 100, 1)
            warnings.append({
                'code': 'face_too_small',
                'message': (
                    f'The face in this photo is too small '
                    f'(covers only {face_percent}% of the image). '
                    f'Accurate treatment placement requires the face to fill '
                    f'at least 8% of the photo.'
                ),
                'suggestion': (
                    'Please take a closer photo. The face should fill most of '
                    'the frame — similar to a passport photo or a selfie taken '
                    'at arm\'s length.'
                )
            })

        # CHECK 3: Is the face cut off at any edge?
        edge_margin_x = img_w * MIN_EDGE_MARGIN_RATIO
        edge_margin_y = img_h * MIN_EDGE_MARGIN_RATIO

        cropped_sides = []
        if face_x_min < edge_margin_x:
            cropped_sides.append('left')
        if face_x_max > img_w - edge_margin_x:
            cropped_sides.append('right')
        if face_y_min < edge_margin_y:
            cropped_sides.append('top')
        if face_y_max > img_h - edge_margin_y:
            cropped_sides.append('bottom')

        if cropped_sides:
            sides_str = ' and '.join(cropped_sides)
            warnings.append({
                'code': 'face_cropped',
                'message': (
                    f'The face appears to be cropped or very close to the '
                    f'{sides_str} edge of the photo. This can cause treatments '
                    f'near that area to apply incorrectly.'
                ),
                'suggestion': (
                    'Please retake the photo with your face fully centred '
                    'in the frame, with clear space on all sides. '
                    'Avoid cropping the forehead, chin, or sides of the face.'
                )
            })

        # ── Return result ──────────────────────────────────────────────────
        passed = len(warnings) == 0
        return {
            'passed': passed,
            'warnings': warnings
        }

    # ==================== FACE DETECTION ====================

    def detect_face(self, image_path: str) -> Optional[Dict]:
        """
        Detect face and extract landmarks from an image.

        Returns:
            Dictionary containing landmarks, facial_regions, face_boundary,
            and image dimensions, or None if no face detected.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            landmarks.append({'x': x, 'y': y, 'z': z})

        facial_regions = self._extract_facial_regions(landmarks, width, height)

        # FIX (Problem 2): Extract face boundary for warp clipping
        face_boundary = self._extract_face_boundary(landmarks)

        return {
            'landmarks': landmarks,
            'facial_regions': facial_regions,
            'face_boundary': face_boundary,
            'image_width': width,
            'image_height': height,
            'face_detected': True
        }

    # ==================== FACE BOUNDARY ====================

    def _extract_face_boundary(self, landmarks: List[Dict]) -> List[Dict]:
        """
        Extract the outer boundary of the face (face oval / silhouette).

        These 36 MediaPipe landmark indices trace a complete loop around the
        face outline. Used by image_utils to create a face-shaped mask so
        warp effects are clipped strictly inside the face.
        """
        FACE_OVAL_INDICES = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

        boundary = []
        for idx in FACE_OVAL_INDICES:
            if idx < len(landmarks):
                boundary.append({
                    'x': landmarks[idx]['x'],
                    'y': landmarks[idx]['y'],
                    'z': landmarks[idx].get('z', 0)
                })

        return boundary

    # ==================== FACIAL REGIONS ====================

    def _extract_facial_regions(self, landmarks: List[Dict], width: int, height: int) -> Dict:
        """
        Extract specific regions for each treatment category.
        Maps MediaPipe landmark indices to treatment areas.
        """
        regions = {
            'face': {
                'temples_left': self._get_landmarks_subset(landmarks, [21, 54, 103, 67]),
                'temples_right': self._get_landmarks_subset(landmarks, [251, 284, 332, 297]),
                'cheeks_left': self._get_landmarks_subset(landmarks, [116, 123, 147, 213, 192]),
                'cheeks_right': self._get_landmarks_subset(landmarks, [345, 352, 376, 433, 416]),
                'chin': self._get_landmarks_subset(landmarks, [199, 175, 152, 148, 176, 152]),
                'jawline_left': self._get_landmarks_subset(landmarks, [172, 136, 150, 176, 152]),
                'jawline_right': self._get_landmarks_subset(landmarks, [397, 365, 379, 400, 377]),
                'forehead': self._get_landmarks_subset(landmarks, [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]),
                'glabella': self._get_landmarks_subset(landmarks, [9, 8, 168]),
                'nasolabial_left': self._get_landmarks_subset(landmarks, [216, 206, 36, 205]),
                'nasolabial_right': self._get_landmarks_subset(landmarks, [436, 426, 266, 425]),
                'marionette_left': self._get_landmarks_subset(landmarks, [57, 186, 92]),
                'marionette_right': self._get_landmarks_subset(landmarks, [287, 410, 322])
            },
            'lips': {
                'upper_lip_outer': self._get_landmarks_subset(landmarks, [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]),
                'lower_lip_outer': self._get_landmarks_subset(landmarks, [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]),
                'upper_lip_inner': self._get_landmarks_subset(landmarks, [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]),
                'lower_lip_inner': self._get_landmarks_subset(landmarks, [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]),
                'cupids_bow': self._get_landmarks_subset(landmarks, [37, 0, 267]),
                'lip_corners_left': self._get_landmarks_subset(landmarks, [61, 146]),
                'lip_corners_right': self._get_landmarks_subset(landmarks, [291, 375]),
                'full_lips': self._get_landmarks_subset(landmarks, list(range(61, 67)) + list(range(291, 297)) + list(range(146, 152)) + list(range(375, 381)))
            },
            'nose': {
                'bridge': self._get_landmarks_subset(landmarks, [6, 197, 195, 5]),
                'tip': self._get_landmarks_subset(landmarks, [4, 19, 94, 2]),
                'root': self._get_landmarks_subset(landmarks, [168, 6]),
                'nostrils_left': self._get_landmarks_subset(landmarks, [98, 97, 2, 326]),
                'nostrils_right': self._get_landmarks_subset(landmarks, [327, 326, 2, 97]),
                'full_nose': self._get_landmarks_subset(landmarks, [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 327, 326])
            },
            'eyebrows': {
                'left_eyebrow': self._get_landmarks_subset(landmarks, [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]),
                'right_eyebrow': self._get_landmarks_subset(landmarks, [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]),
                'left_eyebrow_inner': self._get_landmarks_subset(landmarks, [70, 63]),
                'left_eyebrow_arch': self._get_landmarks_subset(landmarks, [105, 66, 107]),
                'left_eyebrow_outer': self._get_landmarks_subset(landmarks, [46, 53]),
                'right_eyebrow_inner': self._get_landmarks_subset(landmarks, [300, 293]),
                'right_eyebrow_arch': self._get_landmarks_subset(landmarks, [334, 296, 336]),
                'right_eyebrow_outer': self._get_landmarks_subset(landmarks, [276, 283])
            }
        }

        return regions

    def _get_landmarks_subset(self, landmarks: List[Dict], indices: List[int]) -> List[Dict]:
        """Extract a subset of landmarks by their indices."""
        return [landmarks[i] for i in indices if i < len(landmarks)]

    # ==================== VISUALISATION ====================

    def visualize_landmarks(self, image_path: str, output_path: str) -> bool:
        """
        Draw landmarks on image for visualization (helpful for debugging).
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return False

        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        cv2.imwrite(str(output_path), image)
        return True

    def get_face_bounding_box(self, landmarks: List[Dict]) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box around the face.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return (x_min, y_min, x_max, y_max)

    def close(self):
        """Clean up resources"""
        self.face_mesh.close()