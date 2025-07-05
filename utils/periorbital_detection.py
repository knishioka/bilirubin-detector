"""
Periorbital region detection for dark circle analysis
Detects eye regions and extracts infraorbital and cheek areas
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class PerioribitalDetector:
    """Detects periorbital (eye surrounding) regions for analysis"""
    
    def __init__(self):
        # Initialize face and eye cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Region extraction parameters
        self.infraorbital_ratio = 0.3  # Height ratio below eye
        self.cheek_ratio = 0.2  # Additional height for cheek reference
        self.lateral_extension = 0.2  # Lateral extension ratio
    
    def detect_periorbital_regions(self, image: np.ndarray) -> Dict:
        """
        Detect periorbital regions including infraorbital and cheek areas
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing detected regions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {'success': False, 'error': 'No face detected'}
        
        # Use the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = face
        
        # Extract face region for eye detection
        face_roi = gray[fy:fy+fh, fx:fx+fw]
        face_roi_color = image[fy:fy+fh, fx:fx+fw]
        
        # Detect eyes within face
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5)
        
        if len(eyes) < 2:
            return {'success': False, 'error': 'Could not detect both eyes'}
        
        # Sort eyes by x-coordinate (left to right)
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Take the two most prominent eyes
        if len(eyes) > 2:
            # Filter by size and position
            eyes = self._select_best_eye_pair(eyes)
        
        left_eye = eyes[0]  # Left in image (person's right)
        right_eye = eyes[1]  # Right in image (person's left)
        
        # Extract regions for each eye
        left_regions = self._extract_eye_regions(
            face_roi_color, left_eye, 'left'
        )
        right_regions = self._extract_eye_regions(
            face_roi_color, right_eye, 'right'
        )
        
        # Add face coordinates for absolute positioning
        for region in [left_regions, right_regions]:
            region['face_offset'] = (fx, fy)
        
        return {
            'success': True,
            'face_bbox': (fx, fy, fw, fh),
            'left_eye': left_regions['eye'],
            'left_infraorbital': left_regions['infraorbital'],
            'left_cheek': left_regions['cheek'],
            'left_eye_bbox': left_regions['eye_bbox'],
            'right_eye': right_regions['eye'],
            'right_infraorbital': right_regions['infraorbital'],
            'right_cheek': right_regions['cheek'],
            'right_eye_bbox': right_regions['eye_bbox']
        }
    
    def _select_best_eye_pair(self, eyes: list) -> list:
        """Select the best pair of eyes from multiple detections"""
        # Sort by area (larger eyes preferred)
        eyes_with_area = [(e, e[2] * e[3]) for e in eyes]
        eyes_with_area.sort(key=lambda x: x[1], reverse=True)
        
        # Take top candidates
        candidates = [e[0] for e in eyes_with_area[:4]]
        
        # Find best horizontal pair
        best_pair = None
        min_y_diff = float('inf')
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                y_diff = abs(candidates[i][1] - candidates[j][1])
                x_diff = abs(candidates[i][0] - candidates[j][0])
                
                # Eyes should be roughly horizontal and separated
                if y_diff < min_y_diff and x_diff > candidates[i][2]:
                    min_y_diff = y_diff
                    best_pair = sorted([candidates[i], candidates[j]], key=lambda e: e[0])
        
        return best_pair if best_pair else candidates[:2]
    
    def _extract_eye_regions(self, face_roi: np.ndarray, 
                           eye_bbox: Tuple[int, int, int, int],
                           side: str) -> Dict:
        """Extract eye, infraorbital, and cheek regions"""
        ex, ey, ew, eh = eye_bbox
        
        # Extend eye region slightly for better coverage
        padding = int(0.1 * ew)
        ex_start = max(0, ex - padding)
        ex_end = min(face_roi.shape[1], ex + ew + padding)
        ey_start = max(0, ey - padding)
        ey_end = min(face_roi.shape[0], ey + eh + padding)
        
        # Extract eye region
        eye_region = face_roi[ey_start:ey_end, ex_start:ex_end]
        
        # Calculate infraorbital region (below eye)
        infraorbital_height = int(eh * self.infraorbital_ratio)
        infraorbital_y_start = ey + eh
        infraorbital_y_end = min(face_roi.shape[0], infraorbital_y_start + infraorbital_height)
        
        # Add lateral extension
        lateral_ext = int(ew * self.lateral_extension)
        infraorbital_x_start = max(0, ex - lateral_ext)
        infraorbital_x_end = min(face_roi.shape[1], ex + ew + lateral_ext)
        
        infraorbital_region = face_roi[
            infraorbital_y_start:infraorbital_y_end,
            infraorbital_x_start:infraorbital_x_end
        ]
        
        # Calculate cheek reference region (further below)
        cheek_height = int(eh * self.cheek_ratio)
        cheek_y_start = infraorbital_y_end
        cheek_y_end = min(face_roi.shape[0], cheek_y_start + cheek_height)
        
        cheek_region = face_roi[
            cheek_y_start:cheek_y_end,
            infraorbital_x_start:infraorbital_x_end
        ]
        
        # Ensure regions are not empty
        if eye_region.size == 0:
            eye_region = np.zeros((10, 10, 3), dtype=np.uint8)
        if infraorbital_region.size == 0:
            infraorbital_region = np.zeros((10, 10, 3), dtype=np.uint8)
        if cheek_region.size == 0:
            cheek_region = np.zeros((10, 10, 3), dtype=np.uint8)
        
        return {
            'eye': eye_region,
            'infraorbital': infraorbital_region,
            'cheek': cheek_region,
            'eye_bbox': (ex, ey, ew, eh),
            'side': side
        }
    
    def draw_regions(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """Draw detected regions on image for visualization"""
        if not detection_result['success']:
            return image
        
        output = image.copy()
        
        # Draw face bounding box
        fx, fy, fw, fh = detection_result['face_bbox']
        cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        cv2.putText(output, "Face", (fx, fy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw eye regions
        for side in ['left', 'right']:
            eye_bbox = detection_result[f'{side}_eye_bbox']
            ex, ey, ew, eh = eye_bbox
            
            # Adjust coordinates to absolute image coordinates
            ex += fx
            ey += fy
            
            # Eye region
            cv2.rectangle(output, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(output, f"{side} eye", (ex, ey - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Infraorbital region
            infra_y = ey + eh
            infra_h = int(eh * self.infraorbital_ratio)
            lateral_ext = int(ew * self.lateral_extension)
            infra_x = ex - lateral_ext
            infra_w = ew + 2 * lateral_ext
            
            cv2.rectangle(output, (infra_x, infra_y), 
                         (infra_x + infra_w, infra_y + infra_h), 
                         (0, 0, 255), 2)
            
            # Cheek region
            cheek_y = infra_y + infra_h
            cheek_h = int(eh * self.cheek_ratio)
            
            cv2.rectangle(output, (infra_x, cheek_y), 
                         (infra_x + infra_w, cheek_y + cheek_h), 
                         (255, 255, 0), 2)
        
        return output