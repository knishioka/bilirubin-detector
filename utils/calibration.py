"""
Color calibration utilities for improving measurement accuracy
Supports color calibration cards for consistent results across lighting conditions
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class ColorCalibrator:
    """Handles color calibration using reference color cards"""
    
    def __init__(self):
        self.is_calibrated = False
        self.color_matrix = None
        self.reference_colors = self._get_standard_reference_colors()
        
    def _get_standard_reference_colors(self) -> np.ndarray:
        """
        Get standard reference colors for calibration card
        Based on X-Rite ColorChecker or similar standards
        """
        # Simplified 6-patch reference (in sRGB)
        # [White, Gray, Black, Red, Green, Blue]
        reference = np.array([
            [255, 255, 255],  # White
            [128, 128, 128],  # Gray
            [0, 0, 0],        # Black
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255]       # Blue
        ], dtype=np.float32)
        
        # Convert to BGR for OpenCV
        reference_bgr = reference[:, [2, 1, 0]]
        
        return reference_bgr
    
    def calibrate_from_card(self, image: np.ndarray, 
                           card_coords: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Calibrate colors using a calibration card in the image
        
        Args:
            image: Image containing calibration card
            card_coords: Optional (x, y, width, height) of card region
            
        Returns:
            True if calibration successful
        """
        # Detect calibration card if coordinates not provided
        if card_coords is None:
            card_region, coords = self._detect_calibration_card(image)
            if card_region is None:
                return False
        else:
            x, y, w, h = card_coords
            card_region = image[y:y+h, x:x+w]
        
        # Extract color patches from card
        measured_colors = self._extract_color_patches(card_region)
        
        if measured_colors is None or len(measured_colors) < 4:
            return False
        
        # Calculate color correction matrix
        self.color_matrix = self._calculate_color_matrix(
            measured_colors[:len(self.reference_colors)], 
            self.reference_colors
        )
        
        self.is_calibrated = True
        return True
    
    def correct_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction to image
        
        Args:
            image: Input image
            
        Returns:
            Color-corrected image
        """
        if not self.is_calibrated or self.color_matrix is None:
            return image
        
        # Reshape image for matrix multiplication
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Apply color correction
        # Add homogeneous coordinate
        pixels_homo = np.column_stack([pixels, np.ones(len(pixels))])
        corrected = np.dot(pixels_homo, self.color_matrix.T)
        
        # Clip values and reshape
        corrected = np.clip(corrected, 0, 255)
        corrected_image = corrected.reshape(h, w, 3).astype(np.uint8)
        
        return corrected_image
    
    def _detect_calibration_card(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], 
                                                                   Optional[Tuple]]:
        """
        Detect calibration card in image
        Simple detection based on rectangular shape with color patches
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours
        for contour in contours:
            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if rectangle
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio (calibration cards are typically rectangular)
                aspect_ratio = w / h
                if 1.2 < aspect_ratio < 2.5 and w > 100 and h > 50:
                    card_region = image[y:y+h, x:x+w]
                    
                    # Verify it contains distinct color patches
                    if self._verify_calibration_card(card_region):
                        return card_region, (x, y, w, h)
        
        return None, None
    
    def _verify_calibration_card(self, region: np.ndarray) -> bool:
        """
        Verify if region contains a calibration card
        Check for distinct color patches
        """
        # Convert to HSV for better color distinction
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Check color variance
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        # Calibration card should have high variance in all channels
        return h_std > 30 and s_std > 30 and v_std > 50
    
    def _extract_color_patches(self, card_region: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract color values from calibration card patches
        """
        h, w = card_region.shape[:2]
        
        # Simple grid-based extraction (assumes 2x3 or 3x2 layout)
        if w > h:  # Horizontal layout
            rows, cols = 2, 3
        else:  # Vertical layout
            rows, cols = 3, 2
        
        patch_h = h // rows
        patch_w = w // cols
        
        colors = []
        
        for i in range(rows):
            for j in range(cols):
                # Extract center region of each patch
                y1 = i * patch_h + patch_h // 4
                y2 = (i + 1) * patch_h - patch_h // 4
                x1 = j * patch_w + patch_w // 4
                x2 = (j + 1) * patch_w - patch_w // 4
                
                patch = card_region[y1:y2, x1:x2]
                
                if patch.size > 0:
                    # Calculate mean color
                    mean_color = np.mean(patch.reshape(-1, 3), axis=0)
                    colors.append(mean_color)
        
        return np.array(colors) if len(colors) >= 4 else None
    
    def _calculate_color_matrix(self, measured: np.ndarray, 
                               reference: np.ndarray) -> np.ndarray:
        """
        Calculate color correction matrix using least squares
        
        Args:
            measured: Measured color values from card
            reference: Reference color values
            
        Returns:
            3x4 color correction matrix
        """
        # Add homogeneous coordinate to measured colors
        n = min(len(measured), len(reference))
        measured_homo = np.column_stack([measured[:n], np.ones(n)])
        
        # Solve for transformation matrix using least squares
        # reference = measured_homo @ matrix.T
        matrix, _, _, _ = np.linalg.lstsq(measured_homo, reference[:n], rcond=None)
        
        return matrix.T
    
    def save_calibration(self, filepath: str):
        """Save calibration data to file"""
        if self.is_calibrated and self.color_matrix is not None:
            np.savez(filepath, 
                    color_matrix=self.color_matrix,
                    is_calibrated=self.is_calibrated)
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration data from file"""
        try:
            data = np.load(filepath)
            self.color_matrix = data['color_matrix']
            self.is_calibrated = bool(data['is_calibrated'])
            return True
        except:
            return False


def create_calibration_card_reference() -> np.ndarray:
    """
    Create a visual reference calibration card for printing
    
    Returns:
        Image of calibration card
    """
    # Create 6-patch calibration card (2x3 layout)
    patch_size = 100
    border = 20
    
    card_width = 3 * patch_size + 4 * border
    card_height = 2 * patch_size + 3 * border
    
    # Create white background
    card = np.ones((card_height, card_width, 3), dtype=np.uint8) * 255
    
    # Define colors (BGR format)
    colors = [
        [255, 255, 255],  # White
        [128, 128, 128],  # Gray
        [0, 0, 0],        # Black
        [0, 0, 255],      # Red
        [0, 255, 0],      # Green
        [255, 0, 0]       # Blue
    ]
    
    # Draw patches
    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(colors):
                y1 = i * (patch_size + border) + border
                y2 = y1 + patch_size
                x1 = j * (patch_size + border) + border
                x2 = x1 + patch_size
                
                card[y1:y2, x1:x2] = colors[idx]
    
    # Add black border
    cv2.rectangle(card, (0, 0), (card_width-1, card_height-1), (0, 0, 0), 2)
    
    return card