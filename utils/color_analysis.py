"""
Color analysis utilities for RGB/HSV feature extraction
Based on research showing HSV color space effectiveness for bilirubin detection
"""

import cv2
import numpy as np
from typing import Dict, Tuple
import colorsys


class ColorAnalyzer:
    """Analyzes color features relevant to bilirubin detection"""
    
    def __init__(self):
        # Yellow hue range in HSV (in OpenCV scale: 0-180)
        self.yellow_hue_range = (20, 40)  # Approximately 40-80 degrees
        
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Extract color features from image region
        
        Args:
            image: Input image region (BGR format)
            
        Returns:
            Dictionary of color features
        """
        if image is None or image.size == 0:
            return self._empty_features()
        
        # Convert to different color spaces
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract features
        features = {}
        
        # RGB features
        features.update(self._extract_rgb_features(rgb))
        
        # HSV features (most important for bilirubin)
        features.update(self._extract_hsv_features(hsv))
        
        # LAB features
        features.update(self._extract_lab_features(lab))
        
        # Combined features
        features.update(self._extract_combined_features(rgb, hsv, lab))
        
        return features
    
    def _extract_rgb_features(self, rgb: np.ndarray) -> Dict:
        """Extract features from RGB color space"""
        # Flatten non-zero pixels
        mask = np.any(rgb > 0, axis=2)
        pixels = rgb[mask]
        
        if len(pixels) == 0:
            return {
                'rgb_mean_r': 0, 'rgb_mean_g': 0, 'rgb_mean_b': 0,
                'rgb_std_r': 0, 'rgb_std_g': 0, 'rgb_std_b': 0,
                'rgb_red_blue_ratio': 1.0
            }
        
        # Calculate statistics
        mean_values = np.mean(pixels, axis=0)
        std_values = np.std(pixels, axis=0)
        
        # Red/Blue ratio (higher in jaundiced patients)
        rb_ratio = mean_values[0] / (mean_values[2] + 1e-6)
        
        return {
            'rgb_mean_r': float(mean_values[0]),
            'rgb_mean_g': float(mean_values[1]),
            'rgb_mean_b': float(mean_values[2]),
            'rgb_std_r': float(std_values[0]),
            'rgb_std_g': float(std_values[1]),
            'rgb_std_b': float(std_values[2]),
            'rgb_red_blue_ratio': float(rb_ratio)
        }
    
    def _extract_hsv_features(self, hsv: np.ndarray) -> Dict:
        """Extract features from HSV color space"""
        # Flatten non-zero pixels
        mask = hsv[:, :, 2] > 0  # Value channel > 0
        pixels = hsv[mask]
        
        if len(pixels) == 0:
            return {
                'hsv_mean_h': 0, 'hsv_mean_s': 0, 'hsv_mean_v': 0,
                'hsv_yellow_ratio': 0, 'saturation_mean': 0
            }
        
        # Calculate statistics
        mean_h = np.mean(pixels[:, 0])
        mean_s = np.mean(pixels[:, 1])
        mean_v = np.mean(pixels[:, 2])
        
        # Calculate yellow pixel ratio (key feature)
        yellow_mask = (
            (pixels[:, 0] >= self.yellow_hue_range[0]) & 
            (pixels[:, 0] <= self.yellow_hue_range[1]) &
            (pixels[:, 1] > 30) &  # Minimum saturation
            (pixels[:, 2] > 50)    # Minimum value
        )
        yellow_ratio = np.sum(yellow_mask) / len(pixels)
        
        return {
            'hsv_mean_h': float(mean_h),
            'hsv_mean_s': float(mean_s),
            'hsv_mean_v': float(mean_v),
            'hsv_yellow_ratio': float(yellow_ratio),
            'saturation_mean': float(mean_s) / 255.0  # Normalized
        }
    
    def _extract_lab_features(self, lab: np.ndarray) -> Dict:
        """Extract features from LAB color space"""
        # Flatten non-zero pixels
        mask = lab[:, :, 0] > 0  # L channel > 0
        pixels = lab[mask]
        
        if len(pixels) == 0:
            return {
                'lab_mean_l': 0, 'lab_mean_a': 0, 'lab_mean_b': 0,
                'lab_yellowness': 0
            }
        
        # Calculate statistics
        mean_values = np.mean(pixels, axis=0)
        
        # b* channel indicates yellow-blue axis (positive = yellow)
        yellowness = mean_values[2] - 128  # Center around 0
        
        return {
            'lab_mean_l': float(mean_values[0]),
            'lab_mean_a': float(mean_values[1]),
            'lab_mean_b': float(mean_values[2]),
            'lab_yellowness': float(yellowness)
        }
    
    def _extract_combined_features(self, rgb: np.ndarray, hsv: np.ndarray, 
                                  lab: np.ndarray) -> Dict:
        """Extract combined features across color spaces"""
        features = {}
        
        # Yellowness index (custom metric)
        # Combines multiple indicators of yellow coloration
        mask = np.any(rgb > 0, axis=2)
        if np.sum(mask) > 0:
            rgb_pixels = rgb[mask]
            hsv_pixels = hsv[mask]
            
            # RGB yellowness: high R and G, low B
            rgb_yellow = (rgb_pixels[:, 0] + rgb_pixels[:, 1]) / (2 * (rgb_pixels[:, 2] + 1))
            
            # HSV yellowness: specific hue range
            hsv_yellow = (
                (hsv_pixels[:, 0] >= self.yellow_hue_range[0]) & 
                (hsv_pixels[:, 0] <= self.yellow_hue_range[1])
            ).astype(float)
            
            # Combined yellowness index
            yellowness_index = np.mean(rgb_yellow * hsv_yellow)
            features['yellowness_index'] = float(yellowness_index)
        else:
            features['yellowness_index'] = 0.0
        
        return features
    
    def _empty_features(self) -> Dict:
        """Return empty feature dictionary"""
        return {
            'rgb_mean_r': 0, 'rgb_mean_g': 0, 'rgb_mean_b': 0,
            'rgb_std_r': 0, 'rgb_std_g': 0, 'rgb_std_b': 0,
            'rgb_red_blue_ratio': 1.0,
            'hsv_mean_h': 0, 'hsv_mean_s': 0, 'hsv_mean_v': 0,
            'hsv_yellow_ratio': 0, 'saturation_mean': 0,
            'lab_mean_l': 0, 'lab_mean_a': 0, 'lab_mean_b': 0,
            'lab_yellowness': 0,
            'yellowness_index': 0
        }


def extract_color_features(image: np.ndarray) -> Dict:
    """
    Convenience function to extract color features
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Dictionary of color features
    """
    analyzer = ColorAnalyzer()
    return analyzer.analyze(image)


def visualize_color_analysis(image: np.ndarray, features: Dict) -> np.ndarray:
    """
    Create visualization of color analysis results
    
    Args:
        image: Original image
        features: Extracted color features
        
    Returns:
        Visualization image
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # HSV channels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    axes[0, 1].imshow(hsv[:, :, 0], cmap='hsv')
    axes[0, 1].set_title(f'Hue (Yellow ratio: {features["hsv_yellow_ratio"]:.2%})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(hsv[:, :, 1], cmap='gray')
    axes[0, 2].set_title(f'Saturation (Mean: {features["saturation_mean"]:.2f})')
    axes[0, 2].axis('off')
    
    # LAB b* channel (yellow-blue)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    axes[1, 0].imshow(lab[:, :, 2], cmap='RdYlBu_r')
    axes[1, 0].set_title(f'LAB b* (Yellowness: {features["lab_yellowness"]:.1f})')
    axes[1, 0].axis('off')
    
    # RGB ratios
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rb_ratio_img = rgb[:, :, 0].astype(float) / (rgb[:, :, 2].astype(float) + 1)
    axes[1, 1].imshow(rb_ratio_img, cmap='hot')
    axes[1, 1].set_title(f'R/B Ratio (Mean: {features["rgb_red_blue_ratio"]:.2f})')
    axes[1, 1].axis('off')
    
    # Feature summary
    axes[1, 2].text(0.1, 0.8, 'Key Features:', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f'Yellowness Index: {features["yellowness_index"]:.3f}', 
                   fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f'HSV Yellow %: {features["hsv_yellow_ratio"]:.1%}', 
                   fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.2, f'RGB R/B Ratio: {features["rgb_red_blue_ratio"]:.2f}', 
                   fontsize=10, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Convert to image
    fig.canvas.draw()
    viz_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    viz_image = viz_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return viz_image