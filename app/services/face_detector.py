# app/services/face_detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import time
from typing import Any, Dict

class FaceDetector:
    def __init__(self, cache_ttl: int = 300):
        """Initialize MediaPipe Face Mesh and a simple in-memory cache.

        cache_ttl: number of seconds a cached result is considered valid (default 300s = 5 minutes)
        """
        # existing MediaPipe setup (keep your current code)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # ------------------ new cache initialization ------------------
        # _cache stores { image_id: { 'timestamp': float, 'data': face_data_dict } }
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl

    def _is_cache_valid(self, image_id: str) -> bool:
        """Return True if cache entry exists and is not expired."""
        entry = self._cache.get(image_id)
        if not entry:
            return False
        # time.time() returns current time in seconds (float)
        return (time.time() - entry['timestamp']) < self.cache_ttl

    def get_landmarks_cached(self, image_path: str, image_id: str) -> Optional[Dict]:
        """
        Return cached face_data for image_id if valid, otherwise run detection,
        cache the result, and return it.
        """
        # 1) return cached if valid
        if self._is_cache_valid(image_id):
            return self._cache[image_id]['data']

        # 2) otherwise, run detection (this calls your existing detect_face)
        face_data = self.detect_face(image_path)
        if face_data:
            self._cache[image_id] = {'timestamp': time.time(), 'data': face_data}
        return face_data

    def invalidate_cache(self, image_id: str):
        """Remove cached entry for image_id (call this if image changed)."""
        if image_id in self._cache:
            del self._cache[image_id]

    
    def detect_face(self, image_path: str) -> Optional[Dict]:
        """
        Detect face and extract landmarks from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing landmarks and face data, or None if no face detected
        """
        
        # Step 1: Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Step 2: Convert BGR to RGB (MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 3: Get image dimensions
        height, width, _ = image.shape
        
        # Step 4: Process image with MediaPipe
        results = self.face_mesh.process(image_rgb)
        
        # Step 5: Check if face was detected
        if not results.multi_face_landmarks:
            return None
        
        # Step 6: Get the first (and only) face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Step 7: Convert normalized landmarks to pixel coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # Depth information
            landmarks.append({'x': x, 'y': y, 'z': z})
        
        # Step 8: Extract specific facial regions
        facial_regions = self._extract_facial_regions(landmarks, width, height)
        
        # Step 9: Return all data
        return {
            'landmarks': landmarks,
            'facial_regions': facial_regions,
            'image_width': width,
            'image_height': height,
            'face_detected': True
        }
    
    def _extract_facial_regions(self, landmarks: List[Dict], width: int, height: int) -> Dict:
        """
        Extract specific regions for each treatment category
        
        This maps MediaPipe landmark indices to our treatment areas
        """
        
        regions = {
            # FACE REGIONS
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
            
            # LIPS REGIONS
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
            
            # NOSE REGIONS
            'nose': {
                'bridge': self._get_landmarks_subset(landmarks, [6, 197, 195, 5]),
                'tip': self._get_landmarks_subset(landmarks, [4, 19, 94, 2]),
                'root': self._get_landmarks_subset(landmarks, [168, 6]),
                'nostrils_left': self._get_landmarks_subset(landmarks, [98, 97, 2, 326]),
                'nostrils_right': self._get_landmarks_subset(landmarks, [327, 326, 2, 97]),
                'full_nose': self._get_landmarks_subset(landmarks, [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 327, 326])
            },
            
            # EYEBROWS REGIONS
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
        """
        Extract a subset of landmarks by their indices
        
        Args:
            landmarks: List of all landmarks
            indices: List of landmark indices to extract
            
        Returns:
            List of selected landmarks
        """
        return [landmarks[i] for i in indices if i < len(landmarks)]
    
    def visualize_landmarks(self, image_path: str, output_path: str) -> bool:
        """
        Draw landmarks on image for visualization (helpful for debugging)
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            
        Returns:
            True if successful, False otherwise
        """
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return False
        
        # Draw landmarks
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        # Save image
        cv2.imwrite(str(output_path), image)
        return True
    
    def get_face_bounding_box(self, landmarks: List[Dict]) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box around the face
        
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