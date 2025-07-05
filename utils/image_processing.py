"""
Image processing utilities for eye detection and preprocessing
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class EyeDetector:
    """Detects and extracts eye/conjunctiva regions from images"""
    
    def __init__(self):
        # Initialize Haar Cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
    def detect_conjunctiva(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect and extract conjunctiva (white part of eye) region
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (conjunctiva_region, confidence_score)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces first
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            # Try detecting eyes directly without face
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
            if len(eyes) > 0:
                return self._extract_conjunctiva_from_eye(image, eyes[0]), 0.6
            return None, 0.0
        
        # For each face, detect eyes
        best_eye_region = None
        best_confidence = 0.0
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                # Extract eye region
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                
                # Analyze eye quality
                confidence = self._assess_eye_quality(eye_img)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_eye_region = eye_img
        
        if best_eye_region is not None:
            return self._extract_conjunctiva_from_eye(best_eye_region, None), best_confidence
            
        return None, 0.0
    
    def _extract_conjunctiva_from_eye(self, eye_image: np.ndarray, 
                                     eye_coords: Optional[Tuple] = None) -> np.ndarray:
        """
        Extract the conjunctiva (sclera) region from an eye image
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(eye_image, cv2.COLOR_BGR2HSV)
        
        # Define range for white/light colors (sclera)
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for sclera
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to original image
        conjunctiva = cv2.bitwise_and(eye_image, eye_image, mask=mask)
        
        # If extraction failed, return the whole eye region
        if np.sum(mask) < 100:  # Too few white pixels
            return eye_image
            
        return conjunctiva
    
    def _assess_eye_quality(self, eye_image: np.ndarray) -> float:
        """
        Assess the quality of detected eye region
        Returns confidence score between 0 and 1
        """
        # Check image size
        if eye_image.shape[0] < 20 or eye_image.shape[1] < 20:
            return 0.1
        
        # Check brightness
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 220:
            return 0.3
        
        # Check contrast
        contrast = np.std(gray)
        if contrast < 20:
            return 0.4
        
        # Check for presence of white pixels (sclera)
        hsv = cv2.cvtColor(eye_image, cv2.COLOR_BGR2HSV)
        white_pixels = np.sum((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 100))
        white_ratio = white_pixels / (eye_image.shape[0] * eye_image.shape[1])
        
        if white_ratio < 0.05:
            return 0.5
        
        # Calculate final confidence
        confidence = min(1.0, 0.3 + white_ratio * 2 + contrast / 100)
        return confidence


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better detection
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Resize if too large
    max_dim = max(image.shape[:2])
    if max_dim > 1024:
        scale = 1024 / max_dim
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return denoised


def draw_eye_regions(image: np.ndarray, eye_regions: list) -> np.ndarray:
    """
    Draw bounding boxes around detected eye regions
    
    Args:
        image: Input image
        eye_regions: List of (x, y, w, h) tuples
        
    Returns:
        Image with drawn boxes
    """
    output = image.copy()
    
    for (x, y, w, h) in eye_regions:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return output