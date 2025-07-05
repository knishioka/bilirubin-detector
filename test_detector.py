#!/usr/bin/env python3
"""
Test script for bilirubin detection system
Provides sample usage and testing utilities
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

from bilirubin_detector import BilirubinDetector
from utils.calibration import create_calibration_card_reference


def create_sample_eye_image():
    """Create a more realistic synthetic eye image for testing"""
    # Create larger base image for better detection
    height, width = 480, 640
    # Create face-like background (skin tone)
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array([150, 180, 210], dtype=np.uint8)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 5, (height, width, 3))
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Create face region
    face_center = (width // 2, height // 2)
    face_width, face_height = 300, 400
    
    # Draw eye region
    eye_y = face_center[1] - 50
    left_eye_center = (face_center[0] - 80, eye_y)
    right_eye_center = (face_center[0] + 80, eye_y)
    
    # Draw more realistic eyes
    for eye_center in [left_eye_center, right_eye_center]:
        # Eye white (sclera) - elliptical shape
        axes = (50, 25)
        
        # Base white color with slight yellowish tint for jaundice simulation
        sclera_color = [210, 240, 250]  # Slight yellow in BGR
        cv2.ellipse(image, eye_center, axes, 0, 0, 360, sclera_color, -1)
        
        # Add subtle gradient
        overlay = image.copy()
        cv2.ellipse(overlay, eye_center, (axes[0]-5, axes[1]-5), 0, 0, 360, [230, 250, 255], -1)
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
        
        # Draw iris
        iris_radius = 20
        iris_color = [80, 50, 30]  # Brown iris
        cv2.circle(image, eye_center, iris_radius, iris_color, -1)
        
        # Add iris texture
        for r in range(5, iris_radius, 3):
            color_var = [c + np.random.randint(-20, 20) for c in iris_color]
            cv2.circle(image, eye_center, r, color_var, 1)
        
        # Draw pupil
        pupil_radius = 8
        cv2.circle(image, eye_center, pupil_radius, (0, 0, 0), -1)
        
        # Add highlight for realism
        highlight_pos = (eye_center[0] - 5, eye_center[1] - 5)
        cv2.circle(image, highlight_pos, 3, (255, 255, 255), -1)
        
        # Add blood vessels to sclera
        for _ in range(3):
            start_x = eye_center[0] + np.random.randint(-40, -20)
            start_y = eye_center[1] + np.random.randint(-10, 10)
            end_x = eye_center[0] + np.random.randint(-10, 10)
            end_y = eye_center[1] + np.random.randint(-5, 5)
            cv2.line(image, (start_x, start_y), (end_x, end_y), [180, 190, 210], 1)
    
    # Add eyebrows for better face detection
    eyebrow_y = eye_y - 30
    cv2.ellipse(image, (left_eye_center[0], eyebrow_y), (40, 8), 0, 0, 180, (50, 50, 50), -1)
    cv2.ellipse(image, (right_eye_center[0], eyebrow_y), (40, 8), 0, 0, 180, (50, 50, 50), -1)
    
    # Add nose hint
    nose_tip = (face_center[0], face_center[1] + 20)
    cv2.circle(image, nose_tip, 15, [140, 170, 200], -1)
    
    # Apply slight blur for realism
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image


def test_with_sample_image(detector, output_dir):
    """Test detector with synthetic sample image"""
    print("\n=== Testing with synthetic eye image ===")
    
    # Create sample image
    sample_image = create_sample_eye_image()
    sample_path = os.path.join(output_dir, "sample_eye.jpg")
    cv2.imwrite(sample_path, sample_image)
    print(f"Created sample image: {sample_path}")
    
    # Run detection
    try:
        results = detector.detect_bilirubin(sample_path)
        
        if results['success']:
            print(f"\nDetection Results:")
            print(f"  Bilirubin Level: {results['bilirubin_level_mg_dl']} mg/dL")
            print(f"  Risk Level: {results['risk_level']}")
            print(f"  Confidence: {results['confidence']:.2%}")
            
            # Save visualization
            viz_path = os.path.join(output_dir, "sample_result.png")
            detector.visualize_results(sample_path, results, viz_path)
            print(f"\nVisualization saved to: {viz_path}")
        else:
            print(f"\nDetection failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\nError during detection: {str(e)}")


def test_color_features():
    """Test color feature extraction"""
    print("\n=== Testing color feature extraction ===")
    
    from utils.color_analysis import ColorAnalyzer
    
    # Create test patches with known colors
    analyzer = ColorAnalyzer()
    
    # Yellow patch
    yellow_patch = np.full((50, 50, 3), [0, 255, 255], dtype=np.uint8)  # BGR
    features = analyzer.analyze(yellow_patch)
    print(f"\nYellow patch features:")
    print(f"  HSV Yellow Ratio: {features['hsv_yellow_ratio']:.2%}")
    print(f"  RGB R/B Ratio: {features['rgb_red_blue_ratio']:.2f}")
    print(f"  Yellowness Index: {features['yellowness_index']:.3f}")
    
    # White patch
    white_patch = np.full((50, 50, 3), [255, 255, 255], dtype=np.uint8)
    features = analyzer.analyze(white_patch)
    print(f"\nWhite patch features:")
    print(f"  HSV Yellow Ratio: {features['hsv_yellow_ratio']:.2%}")
    print(f"  Saturation Mean: {features['saturation_mean']:.2f}")


def test_calibration_card():
    """Test calibration card creation and detection"""
    print("\n=== Testing calibration card ===")
    
    # Create calibration card
    card = create_calibration_card_reference()
    card_path = "calibration_card.png"
    cv2.imwrite(card_path, card)
    print(f"Calibration card saved to: {card_path}")
    
    # Test calibration
    from utils.calibration import ColorCalibrator
    calibrator = ColorCalibrator()
    
    # Simulate calibration
    success = calibrator.calibrate_from_card(card)
    print(f"Calibration {'successful' if success else 'failed'}")


def main():
    parser = argparse.ArgumentParser(description='Test bilirubin detection system')
    parser.add_argument('--image', help='Path to test image (optional)')
    parser.add_argument('--output-dir', default='test_output', 
                       help='Directory for test outputs')
    parser.add_argument('--create-calibration-card', action='store_true',
                       help='Create calibration card reference')
    parser.add_argument('--test-all', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    detector = BilirubinDetector()
    
    print("Bilirubin Detection System - Test Suite")
    print("=" * 40)
    
    if args.create_calibration_card:
        test_calibration_card()
        return
    
    if args.test_all:
        # Run all tests
        test_with_sample_image(detector, args.output_dir)
        test_color_features()
        test_calibration_card()
    elif args.image:
        # Test with provided image
        print(f"\n=== Testing with image: {args.image} ===")
        
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        try:
            results = detector.detect_bilirubin(args.image)
            
            if results['success']:
                print(f"\nDetection Results:")
                print(f"  Bilirubin Level: {results['bilirubin_level_mg_dl']} mg/dL")
                print(f"  Risk Level: {results['risk_level']}")
                print(f"  Confidence: {results['confidence']:.2%}")
                
                # Color features
                print(f"\nKey Color Features:")
                features = results.get('color_features', {})
                print(f"  HSV Yellow Ratio: {features.get('hsv_yellow_ratio', 0):.2%}")
                print(f"  RGB R/B Ratio: {features.get('rgb_red_blue_ratio', 0):.2f}")
                print(f"  LAB Yellowness: {features.get('lab_yellowness', 0):.1f}")
                print(f"  Yellowness Index: {features.get('yellowness_index', 0):.3f}")
                
                # Save visualization
                output_path = os.path.join(args.output_dir, 
                                         f"result_{Path(args.image).stem}.png")
                detector.visualize_results(args.image, results, output_path)
                print(f"\nVisualization saved to: {output_path}")
            else:
                print(f"\nDetection failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        # Default: test with sample image
        test_with_sample_image(detector, args.output_dir)
    
    print("\n" + "=" * 40)
    print("Testing complete!")


if __name__ == "__main__":
    main()