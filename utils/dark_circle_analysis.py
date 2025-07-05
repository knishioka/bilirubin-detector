"""
Dark circle color analysis utilities
Implements CIELAB color space analysis and ΔE calculations
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from skimage import color as skcolor
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color


class DarkCircleAnalyzer:
    """Analyzes color characteristics for dark circle detection"""
    
    def __init__(self):
        # Standard illuminant for color calculations
        self.illuminant = 'D65'
        
    def calculate_delta_e(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """
        Calculate CIE2000 ΔE between two regions
        
        Args:
            region1: First region (e.g., infraorbital)
            region2: Second region (e.g., cheek)
            
        Returns:
            ΔE value (color difference)
        """
        # Get mean LAB values for each region
        lab1 = self.get_mean_lab(region1)
        lab2 = self.get_mean_lab(region2)
        
        # Create LabColor objects
        color1 = LabColor(lab1[0], lab1[1], lab1[2])
        color2 = LabColor(lab2[0], lab2[1], lab2[2])
        
        # Calculate CIE2000 ΔE
        delta_e = delta_e_cie2000(color1, color2)
        
        return float(delta_e)
    
    def get_mean_lab(self, region: np.ndarray) -> np.ndarray:
        """
        Get mean LAB values for a region
        
        Args:
            region: Input region in BGR format
            
        Returns:
            Mean LAB values [L, a, b]
        """
        # Convert BGR to RGB
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        region_normalized = region_rgb.astype(np.float32) / 255.0
        
        # Convert to LAB
        region_lab = skcolor.rgb2lab(region_normalized)
        
        # Calculate mean values
        mean_lab = np.mean(region_lab.reshape(-1, 3), axis=0)
        
        return mean_lab
    
    def calculate_ita(self, lab_values: np.ndarray) -> float:
        """
        Calculate Individual Typology Angle (ITA)
        ITA = arctan((L* - 50) / b*) × 180 / π
        
        Args:
            lab_values: LAB color values [L, a, b]
            
        Returns:
            ITA value in degrees
        """
        L, a, b = lab_values
        
        # ITA formula
        ita = np.arctan2(L - 50, b) * 180 / np.pi
        
        return ita
    
    def classify_skin_type(self, ita: float) -> str:
        """
        Classify skin type based on ITA value
        
        Args:
            ita: Individual Typology Angle
            
        Returns:
            Skin type classification
        """
        if ita > 55:
            return "very_light"
        elif ita > 41:
            return "light"
        elif ita > 28:
            return "intermediate"
        elif ita > 10:
            return "tan"
        elif ita > -30:
            return "brown"
        else:
            return "dark"
    
    def calculate_darkness_ratio(self, l1: float, l2: float) -> float:
        """
        Calculate darkness ratio between two regions
        
        Args:
            l1: L* value of darker region (infraorbital)
            l2: L* value of lighter region (cheek)
            
        Returns:
            Darkness ratio (0-1, higher = darker)
        """
        # Ensure l2 is not zero
        if l2 == 0:
            return 1.0
        
        # Calculate ratio (inverted so higher = darker)
        ratio = 1.0 - (l1 / l2)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, ratio))
    
    def calculate_redness_index(self, region: np.ndarray) -> float:
        """
        Calculate redness index (positive a* values in LAB)
        Useful for detecting vascular-type dark circles
        
        Args:
            region: Input region in BGR format
            
        Returns:
            Redness index
        """
        lab = self.get_mean_lab(region)
        
        # a* positive values indicate redness
        # Normalize to typical range
        redness = max(0, lab[1]) / 20.0  # Typical a* range is -20 to +20
        
        return min(1.0, redness)
    
    def calculate_blueness_index(self, region: np.ndarray) -> float:
        """
        Calculate blueness index (negative b* values in LAB)
        Useful for detecting venous pooling
        
        Args:
            region: Input region in BGR format
            
        Returns:
            Blueness index
        """
        lab = self.get_mean_lab(region)
        
        # Negative b* values indicate blueness
        # For skin, we look at how much less yellow it is
        reference_b = 15.0  # Typical b* for healthy skin
        blueness = max(0, reference_b - lab[2]) / reference_b
        
        return min(1.0, blueness)
    
    def analyze_color_distribution(self, region: np.ndarray) -> Dict:
        """
        Analyze color distribution statistics
        
        Args:
            region: Input region in BGR format
            
        Returns:
            Dictionary with distribution statistics
        """
        # Convert to LAB
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        region_normalized = region_rgb.astype(np.float32) / 255.0
        region_lab = skcolor.rgb2lab(region_normalized)
        
        # Flatten for statistics
        l_channel = region_lab[:, :, 0].flatten()
        a_channel = region_lab[:, :, 1].flatten()
        b_channel = region_lab[:, :, 2].flatten()
        
        return {
            'l_mean': float(np.mean(l_channel)),
            'l_std': float(np.std(l_channel)),
            'l_min': float(np.min(l_channel)),
            'l_max': float(np.max(l_channel)),
            'a_mean': float(np.mean(a_channel)),
            'a_std': float(np.std(a_channel)),
            'b_mean': float(np.mean(b_channel)),
            'b_std': float(np.std(b_channel))
        }
    
    def calculate_color_uniformity(self, region: np.ndarray) -> float:
        """
        Calculate color uniformity score
        Lower values indicate more uniform color (less texture/vessels)
        
        Args:
            region: Input region in BGR format
            
        Returns:
            Uniformity score (0-1, higher = more uniform)
        """
        stats = self.analyze_color_distribution(region)
        
        # Use coefficient of variation of L* channel
        cv = stats['l_std'] / (stats['l_mean'] + 1e-6)
        
        # Invert and normalize (lower CV = more uniform)
        uniformity = 1.0 - min(cv / 0.2, 1.0)  # 0.2 is typical max CV
        
        return uniformity
    
    def get_delta_e_map(self, infraorbital: np.ndarray, 
                       cheek_reference: np.ndarray) -> np.ndarray:
        """
        Create a pixel-wise ΔE map comparing infraorbital region to cheek reference
        
        Args:
            infraorbital: Infraorbital region
            cheek_reference: Cheek reference region
            
        Returns:
            ΔE map (same size as infraorbital)
        """
        # Get reference LAB color from cheek
        cheek_lab = self.get_mean_lab(cheek_reference)
        cheek_color = LabColor(cheek_lab[0], cheek_lab[1], cheek_lab[2])
        
        # Convert infraorbital to LAB
        infra_rgb = cv2.cvtColor(infraorbital, cv2.COLOR_BGR2RGB)
        infra_normalized = infra_rgb.astype(np.float32) / 255.0
        infra_lab = skcolor.rgb2lab(infra_normalized)
        
        # Calculate ΔE for each pixel
        h, w = infraorbital.shape[:2]
        delta_e_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                pixel_lab = infra_lab[i, j]
                pixel_color = LabColor(pixel_lab[0], pixel_lab[1], pixel_lab[2])
                delta_e_map[i, j] = delta_e_cie2000(pixel_color, cheek_color)
        
        return delta_e_map