#!/usr/bin/env python3
"""
Bilirubin Detection System - Main Script
Detects bilirubin levels from eye/conjunctiva images using color analysis
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple, Optional

from utils.image_processing import EyeDetector, preprocess_image
from utils.color_analysis import ColorAnalyzer, extract_color_features
from utils.calibration import ColorCalibrator


class BilirubinDetector:
    """Main class for bilirubin level detection from eye images"""
    
    def __init__(self, calibration_mode: bool = False):
        self.eye_detector = EyeDetector()
        self.color_analyzer = ColorAnalyzer()
        self.calibrator = ColorCalibrator() if calibration_mode else None
        
        # Simple linear regression coefficients (placeholder)
        # In production, these would be trained from clinical data
        self.model_coefficients = {
            'hsv_yellow_ratio': 50.0,  # Higher yellow = higher bilirubin
            'rgb_red_blue_ratio': 25.0,  # R/B ratio contribution
            'lab_yellowness': 0.8,     # LAB b* channel contribution
            'yellowness_index': 30.0,  # Combined yellowness
            'saturation_mean': 10.0,
            'intercept': -5.0
        }
        
    def detect_bilirubin(self, image_path: str) -> Dict:
        """
        Main detection pipeline
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing bilirubin level and analysis results
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
            
        processed_image = preprocess_image(image)
        
        # Detect eye region
        eye_region, confidence = self.eye_detector.detect_conjunctiva(processed_image)
        if eye_region is None:
            return {
                'success': False,
                'error': 'Could not detect eye region',
                'confidence': 0.0
            }
        
        # Apply color calibration if available
        if self.calibrator and self.calibrator.is_calibrated:
            eye_region = self.calibrator.correct_colors(eye_region)
        
        # Extract color features
        color_features = self.color_analyzer.analyze(eye_region)
        
        # Estimate bilirubin level
        bilirubin_level = self._estimate_bilirubin(color_features)
        
        # Determine risk level
        risk_level = self._assess_risk(bilirubin_level)
        
        return {
            'success': True,
            'bilirubin_level_mg_dl': round(bilirubin_level, 2),
            'risk_level': risk_level,
            'confidence': confidence,
            'color_features': color_features,
            'calibrated': self.calibrator is not None and self.calibrator.is_calibrated
        }
    
    def _estimate_bilirubin(self, features: Dict) -> float:
        """
        Estimate bilirubin level from color features
        Simple linear model for prototype
        """
        # Extract relevant features
        hsv_yellow = features.get('hsv_yellow_ratio', 0)
        rgb_ratio = features.get('rgb_red_blue_ratio', 1)
        lab_yellowness = features.get('lab_yellowness', 0)
        yellowness_index = features.get('yellowness_index', 0)
        saturation = features.get('saturation_mean', 0)
        
        # Simple linear combination (placeholder model)
        bilirubin = (
            self.model_coefficients['hsv_yellow_ratio'] * hsv_yellow +
            self.model_coefficients['rgb_red_blue_ratio'] * (rgb_ratio - 1.0) +  # Normalize around 1
            self.model_coefficients['lab_yellowness'] * lab_yellowness +
            self.model_coefficients['yellowness_index'] * yellowness_index +
            self.model_coefficients['saturation_mean'] * saturation +
            self.model_coefficients['intercept']
        )
        
        # Clamp to realistic range (0-30 mg/dL)
        return max(0, min(30, bilirubin))
    
    def _assess_risk(self, bilirubin_level: float) -> str:
        """
        Assess jaundice risk based on bilirubin level
        Based on medical guidelines
        """
        if bilirubin_level < 3:
            return "low"
        elif bilirubin_level < 12:
            return "moderate"
        elif bilirubin_level < 20:
            return "high"
        else:
            return "critical"
    
    def visualize_results(self, image_path: str, results: Dict, output_path: str):
        """Create visualization of detection results"""
        import matplotlib.pyplot as plt
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Results panel
        axes[1].text(0.1, 0.8, f"Bilirubin Level: {results.get('bilirubin_level_mg_dl', 'N/A')} mg/dL", 
                    fontsize=14, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.6, f"Risk Level: {results.get('risk_level', 'N/A')}", 
                    fontsize=12, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.4, f"Confidence: {results.get('confidence', 0):.2%}", 
                    fontsize=12, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.2, f"Calibrated: {'Yes' if results.get('calibrated') else 'No'}", 
                    fontsize=10, transform=axes[1].transAxes)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Detect bilirubin levels from eye images')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--calibrate', action='store_true', 
                       help='Enable color calibration mode')
    parser.add_argument('--output', '-o', help='Output path for results visualization')
    parser.add_argument('--json', action='store_true', 
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BilirubinDetector(calibration_mode=args.calibrate)
    
    try:
        # Run detection
        results = detector.detect_bilirubin(args.image)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if results['success']:
                print(f"\nBilirubin Detection Results:")
                print(f"  Bilirubin Level: {results['bilirubin_level_mg_dl']} mg/dL")
                print(f"  Risk Level: {results['risk_level']}")
                print(f"  Confidence: {results['confidence']:.2%}")
                print(f"  Calibrated: {'Yes' if results['calibrated'] else 'No'}")
            else:
                print(f"\nDetection failed: {results.get('error', 'Unknown error')}")
        
        # Generate visualization if requested
        if args.output and results['success']:
            detector.visualize_results(args.image, results, args.output)
            print(f"\nVisualization saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())