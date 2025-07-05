"""
Dark circle segmentation utilities
Segments dark circle regions based on color difference (ΔE) thresholds
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from skimage import morphology, filters


class DarkCircleSegmenter:
    """Segments dark circle regions from periorbital areas"""
    
    def __init__(self):
        # Segmentation thresholds
        self.delta_e_threshold = 3.0  # Minimum ΔE to consider as dark circle
        self.min_area_ratio = 0.05    # Minimum area ratio to eye size
        self.max_area_ratio = 0.7     # Maximum area ratio to eye size
        
        # Morphological operation parameters
        self.kernel_size = 3
        self.closing_iterations = 2
        self.opening_iterations = 1
        
    def segment_dark_circle(self, eye_region: np.ndarray,
                          infraorbital_region: np.ndarray,
                          delta_e: float) -> np.ndarray:
        """
        Segment dark circle region based on color analysis
        
        Args:
            eye_region: Eye region image
            infraorbital_region: Infraorbital region image
            delta_e: Overall ΔE value for the region
            
        Returns:
            Binary mask of dark circle region
        """
        if delta_e < self.delta_e_threshold:
            # No significant dark circle
            return np.zeros(infraorbital_region.shape[:2], dtype=np.uint8)
        
        # Create adaptive threshold based on ΔE magnitude
        adaptive_threshold = self._calculate_adaptive_threshold(delta_e)
        
        # Convert to LAB for better color segmentation
        lab = cv2.cvtColor(infraorbital_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply adaptive thresholding
        mask = self._adaptive_segmentation(l_channel, adaptive_threshold)
        
        # Refine mask using morphological operations
        mask = self._refine_mask(mask)
        
        # Validate mask size and connectivity
        mask = self._validate_mask(mask, eye_region.shape[:2])
        
        return mask
    
    def _calculate_adaptive_threshold(self, delta_e: float) -> float:
        """
        Calculate adaptive threshold based on ΔE value
        
        Args:
            delta_e: Color difference value
            
        Returns:
            Threshold percentage (0-1)
        """
        # Linear mapping: ΔE 3->0.3, ΔE 8->0.7
        if delta_e <= 3:
            return 0.3
        elif delta_e >= 8:
            return 0.7
        else:
            return 0.3 + (delta_e - 3) * 0.08
    
    def _adaptive_segmentation(self, l_channel: np.ndarray,
                              threshold_ratio: float) -> np.ndarray:
        """
        Perform adaptive segmentation on L channel
        
        Args:
            l_channel: Lightness channel from LAB
            threshold_ratio: Threshold percentage
            
        Returns:
            Binary mask
        """
        # Calculate statistics
        mean_l = np.mean(l_channel)
        std_l = np.std(l_channel)
        
        # Adaptive threshold: pixels darker than mean - (std * ratio)
        threshold = mean_l - (std_l * threshold_ratio)
        
        # Create initial mask
        mask = (l_channel < threshold).astype(np.uint8) * 255
        
        # Apply Gaussian blur to smooth boundaries
        mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine mask using morphological operations
        
        Args:
            mask: Initial binary mask
            
        Returns:
            Refined mask
        """
        # Define morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.kernel_size, self.kernel_size)
        )
        
        # Closing to fill gaps
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel,
            iterations=self.closing_iterations
        )
        
        # Opening to remove noise
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel,
            iterations=self.opening_iterations
        )
        
        # Fill holes
        mask_filled = ndimage.binary_fill_holes(mask // 255).astype(np.uint8) * 255
        
        return mask_filled
    
    def _validate_mask(self, mask: np.ndarray,
                      eye_size: Tuple[int, int]) -> np.ndarray:
        """
        Validate mask size and connectivity
        
        Args:
            mask: Binary mask
            eye_size: Size of eye region (height, width)
            
        Returns:
            Validated mask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        if num_labels <= 1:
            # No components found
            return mask
        
        # Calculate eye area for reference
        eye_area = eye_size[0] * eye_size[1]
        min_area = int(eye_area * self.min_area_ratio)
        max_area = int(eye_area * self.max_area_ratio)
        
        # Create validated mask
        validated_mask = np.zeros_like(mask)
        
        # Keep only components within size range
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if min_area <= area <= max_area:
                validated_mask[labels == i] = 255
        
        # If no valid components, return largest component
        if np.sum(validated_mask) == 0 and num_labels > 1:
            # Find largest component (excluding background)
            largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            validated_mask[labels == largest_idx] = 255
        
        return validated_mask
    
    def segment_with_delta_e_map(self, infraorbital: np.ndarray,
                                delta_e_map: np.ndarray,
                                threshold: float = 3.0) -> np.ndarray:
        """
        Segment using pixel-wise ΔE map
        
        Args:
            infraorbital: Infraorbital region
            delta_e_map: Pixel-wise ΔE values
            threshold: ΔE threshold for segmentation
            
        Returns:
            Binary mask
        """
        # Create mask from ΔE threshold
        mask = (delta_e_map > threshold).astype(np.uint8) * 255
        
        # Apply smoothing
        mask = cv2.medianBlur(mask, 5)
        
        # Refine with morphological operations
        mask = self._refine_mask(mask)
        
        return mask
    
    def create_severity_map(self, delta_e_map: np.ndarray) -> np.ndarray:
        """
        Create severity map with multiple levels
        
        Args:
            delta_e_map: Pixel-wise ΔE values
            
        Returns:
            Severity map (0-3 levels)
        """
        severity_map = np.zeros_like(delta_e_map, dtype=np.uint8)
        
        # Define severity levels
        severity_map[delta_e_map >= 3.0] = 1  # Mild
        severity_map[delta_e_map >= 5.0] = 2  # Moderate
        severity_map[delta_e_map >= 8.0] = 3  # Severe
        
        return severity_map
    
    def extract_dark_circle_features(self, mask: np.ndarray,
                                   infraorbital: np.ndarray) -> dict:
        """
        Extract features from segmented dark circle region
        
        Args:
            mask: Binary mask of dark circle
            infraorbital: Infraorbital region image
            
        Returns:
            Dictionary of features
        """
        if np.sum(mask) == 0:
            return {
                'area': 0,
                'mean_intensity': 0,
                'eccentricity': 0,
                'solidity': 0,
                'extent': 0
            }
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return {
                'area': 0,
                'mean_intensity': 0,
                'eccentricity': 0,
                'solidity': 0,
                'extent': 0
            }
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(largest_contour)
        
        # Fit ellipse if possible
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (center, (width, height), angle) = ellipse
            eccentricity = np.sqrt(1 - (min(width, height) / max(width, height)) ** 2)
        else:
            eccentricity = 0
        
        # Calculate solidity (convexity measure)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate extent (rectangularity)
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Calculate mean intensity in masked region
        gray = cv2.cvtColor(infraorbital, cv2.COLOR_BGR2GRAY)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        
        return {
            'area': int(area),
            'mean_intensity': float(mean_intensity),
            'eccentricity': float(eccentricity),
            'solidity': float(solidity),
            'extent': float(extent)
        }
    
    def visualize_segmentation(self, infraorbital: np.ndarray,
                             mask: np.ndarray,
                             delta_e_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create visualization of segmentation results
        
        Args:
            infraorbital: Infraorbital region
            mask: Binary mask
            delta_e_map: Optional ΔE map for overlay
            
        Returns:
            Visualization image
        """
        h, w = infraorbital.shape[:2]
        
        # Create 2x2 grid visualization
        vis = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Top-left: Original
        vis[:h, :w] = infraorbital
        cv2.putText(vis, "Original", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Top-right: Mask overlay
        overlay = infraorbital.copy()
        overlay[mask > 0] = [0, 0, 255]  # Red overlay
        vis[:h, w:] = cv2.addWeighted(infraorbital, 0.7, overlay, 0.3, 0)
        cv2.putText(vis, "Mask", (w + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bottom-left: Binary mask
        vis[h:, :w] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, "Binary", (5, h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bottom-right: ΔE map or severity
        if delta_e_map is not None:
            # Normalize ΔE map to 0-255
            delta_e_norm = np.clip(delta_e_map * 25, 0, 255).astype(np.uint8)
            delta_e_color = cv2.applyColorMap(delta_e_norm, cv2.COLORMAP_JET)
            vis[h:, w:] = delta_e_color
            cv2.putText(vis, "Delta E", (w + 5, h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Show severity map
            severity = self.create_severity_map(np.ones((h, w)) * 5)  # Dummy
            severity_color = cv2.applyColorMap(severity * 85, cv2.COLORMAP_JET)
            vis[h:, w:] = severity_color
            cv2.putText(vis, "Severity", (w + 5, h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add grid lines
        cv2.line(vis, (w, 0), (w, h * 2), (128, 128, 128), 1)
        cv2.line(vis, (0, h), (w * 2, h), (128, 128, 128), 1)
        
        return vis