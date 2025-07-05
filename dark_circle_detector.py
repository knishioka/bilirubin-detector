#!/usr/bin/env python3
"""
Dark Circle (Periorbital Hyperpigmentation) Detection System
Detects and quantifies dark circles under eyes using color analysis
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple, Optional
from skimage import color

from utils.periorbital_detection import PerioribitalDetector
from utils.dark_circle_analysis import DarkCircleAnalyzer
from utils.dark_circle_segmentation import DarkCircleSegmenter


class DarkCircleDetector:
    """Main class for dark circle detection and quantification"""
    
    def __init__(self):
        self.periorbital_detector = PerioribitalDetector()
        self.analyzer = DarkCircleAnalyzer()
        self.segmenter = DarkCircleSegmenter()
        
        # Severity thresholds based on ΔE (CIE2000)
        self.severity_thresholds = {
            'none': 3.0,      # ΔE < 3: Not noticeable
            'mild': 5.0,      # ΔE 3-5: Slightly noticeable
            'moderate': 8.0,  # ΔE 5-8: Clearly visible
            'severe': float('inf')  # ΔE > 8: Very prominent
        }
    
    def detect_dark_circles(self, image_path: str) -> Dict:
        """
        Main detection pipeline for dark circles
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing detection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        # Detect periorbital regions
        detection_result = self.periorbital_detector.detect_periorbital_regions(image)
        
        if not detection_result['success']:
            return {
                'success': False,
                'error': detection_result.get('error', 'Failed to detect eye regions')
            }
        
        # Extract regions
        left_eye = detection_result['left_eye']
        right_eye = detection_result['right_eye']
        left_infraorbital = detection_result['left_infraorbital']
        right_infraorbital = detection_result['right_infraorbital']
        left_cheek = detection_result['left_cheek']
        right_cheek = detection_result['right_cheek']
        
        # Analyze left eye
        left_results = self._analyze_single_eye(
            left_infraorbital, left_cheek, 'left'
        )
        
        # Analyze right eye
        right_results = self._analyze_single_eye(
            right_infraorbital, right_cheek, 'right'
        )
        
        # Segment dark circle regions
        left_mask = self.segmenter.segment_dark_circle(
            left_eye, left_infraorbital, left_results['delta_e']
        )
        right_mask = self.segmenter.segment_dark_circle(
            right_eye, right_infraorbital, right_results['delta_e']
        )
        
        # Calculate overall metrics
        avg_delta_e = (left_results['delta_e'] + right_results['delta_e']) / 2
        severity = self._assess_severity(avg_delta_e)
        symmetry = self._calculate_symmetry(left_results, right_results)
        
        return {
            'success': True,
            'left_eye': left_results,
            'right_eye': right_results,
            'average_delta_e': round(avg_delta_e, 2),
            'severity': severity,
            'symmetry_score': round(symmetry, 2),
            'masks': {
                'left': left_mask,
                'right': right_mask
            },
            'face_bbox': detection_result.get('face_bbox')
        }
    
    def _analyze_single_eye(self, infraorbital: np.ndarray, 
                           cheek: np.ndarray, side: str) -> Dict:
        """Analyze dark circle for a single eye"""
        # Calculate color metrics
        delta_e = self.analyzer.calculate_delta_e(infraorbital, cheek)
        
        # Get LAB values
        infraorbital_lab = self.analyzer.get_mean_lab(infraorbital)
        cheek_lab = self.analyzer.get_mean_lab(cheek)
        
        # Calculate ITA for skin tone independence
        ita_infraorbital = self.analyzer.calculate_ita(infraorbital_lab)
        ita_cheek = self.analyzer.calculate_ita(cheek_lab)
        
        # Additional metrics
        darkness_ratio = self.analyzer.calculate_darkness_ratio(
            infraorbital_lab[0], cheek_lab[0]
        )
        
        redness = self.analyzer.calculate_redness_index(infraorbital)
        blueness = self.analyzer.calculate_blueness_index(infraorbital)
        
        return {
            'side': side,
            'delta_e': round(delta_e, 2),
            'darkness_ratio': round(darkness_ratio, 3),
            'ita_infraorbital': round(ita_infraorbital, 1),
            'ita_cheek': round(ita_cheek, 1),
            'lab_infraorbital': {
                'L': round(infraorbital_lab[0], 1),
                'a': round(infraorbital_lab[1], 1),
                'b': round(infraorbital_lab[2], 1)
            },
            'lab_cheek': {
                'L': round(cheek_lab[0], 1),
                'a': round(cheek_lab[1], 1),
                'b': round(cheek_lab[2], 1)
            },
            'redness_index': round(redness, 2),
            'blueness_index': round(blueness, 2)
        }
    
    def _assess_severity(self, delta_e: float) -> str:
        """Assess dark circle severity based on ΔE value"""
        if delta_e < self.severity_thresholds['none']:
            return 'none'
        elif delta_e < self.severity_thresholds['mild']:
            return 'mild'
        elif delta_e < self.severity_thresholds['moderate']:
            return 'moderate'
        else:
            return 'severe'
    
    def _calculate_symmetry(self, left_results: Dict, right_results: Dict) -> float:
        """Calculate symmetry score between left and right eyes"""
        delta_e_diff = abs(left_results['delta_e'] - right_results['delta_e'])
        
        # Normalize to 0-1 scale (1 = perfect symmetry)
        symmetry = 1.0 - min(delta_e_diff / 10.0, 1.0)
        return symmetry
    
    def visualize_results(self, image_path: str, results: Dict, output_path: str):
        """Create visualization of detection results"""
        import matplotlib.pyplot as plt
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image with bounding box
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Dark circle masks overlay
        if results['success']:
            overlay = image_rgb.copy()
            
            # Apply masks with transparency
            if 'masks' in results:
                left_mask = results['masks']['left']
                right_mask = results['masks']['right']
                
                # Create colored overlay
                overlay[left_mask > 0] = [255, 0, 0]  # Red for left
                overlay[right_mask > 0] = [0, 0, 255]  # Blue for right
                
            blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
            axes[0, 1].imshow(blended)
            axes[0, 1].set_title('Dark Circle Detection')
            axes[0, 1].axis('off')
        
        # Metrics display
        axes[1, 0].text(0.1, 0.9, 'Detection Results', fontsize=16, 
                       weight='bold', transform=axes[1, 0].transAxes)
        
        if results['success']:
            metrics_text = f"""
Average ΔE: {results['average_delta_e']}
Severity: {results['severity'].upper()}
Symmetry: {results['symmetry_score']:.0%}

Left Eye:
  ΔE: {results['left_eye']['delta_e']}
  Darkness Ratio: {results['left_eye']['darkness_ratio']}
  
Right Eye:
  ΔE: {results['right_eye']['delta_e']}
  Darkness Ratio: {results['right_eye']['darkness_ratio']}
"""
            axes[1, 0].text(0.1, 0.7, metrics_text, fontsize=10,
                          transform=axes[1, 0].transAxes, verticalalignment='top')
        else:
            axes[1, 0].text(0.1, 0.5, f"Detection failed: {results.get('error', 'Unknown error')}", 
                          fontsize=12, color='red', transform=axes[1, 0].transAxes)
        
        axes[1, 0].axis('off')
        
        # Color analysis chart
        if results['success']:
            categories = ['Left\nInfraorbital', 'Left\nCheek', 'Right\nInfraorbital', 'Right\nCheek']
            l_values = [
                results['left_eye']['lab_infraorbital']['L'],
                results['left_eye']['lab_cheek']['L'],
                results['right_eye']['lab_infraorbital']['L'],
                results['right_eye']['lab_cheek']['L']
            ]
            
            axes[1, 1].bar(categories, l_values, color=['darkred', 'pink', 'darkblue', 'lightblue'])
            axes[1, 1].set_ylabel('L* (Lightness)')
            axes[1, 1].set_title('Lightness Comparison')
            axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Detect dark circles under eyes')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--output', '-o', help='Output path for visualization')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DarkCircleDetector()
    
    try:
        # Run detection
        results = detector.detect_dark_circles(args.image)
        
        if args.json:
            # Remove numpy arrays from results for JSON serialization
            json_results = {k: v for k, v in results.items() if k != 'masks'}
            print(json.dumps(json_results, indent=2))
        else:
            if results['success']:
                print(f"\nDark Circle Detection Results:")
                print(f"  Average ΔE: {results['average_delta_e']}")
                print(f"  Severity: {results['severity']}")
                print(f"  Symmetry: {results['symmetry_score']:.0%}")
                print(f"\nLeft Eye:")
                print(f"  ΔE: {results['left_eye']['delta_e']}")
                print(f"  Darkness Ratio: {results['left_eye']['darkness_ratio']}")
                print(f"\nRight Eye:")
                print(f"  ΔE: {results['right_eye']['delta_e']}")
                print(f"  Darkness Ratio: {results['right_eye']['darkness_ratio']}")
            else:
                print(f"\nDetection failed: {results.get('error', 'Unknown error')}")
        
        # Generate visualization if requested
        if args.output:
            detector.visualize_results(args.image, results, args.output)
            print(f"\nVisualization saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())