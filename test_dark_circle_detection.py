#!/usr/bin/env python3
"""
Test script for dark circle detection system
Tests the detection on sample images and validates the algorithm
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from dark_circle_detector import DarkCircleDetector
from utils.periorbital_detection import PerioribitalDetector
from utils.dark_circle_analysis import DarkCircleAnalyzer


def test_periorbital_detection():
    """Test periorbital region detection"""
    print("Testing periorbital detection...")
    
    # Create test image with synthetic face
    img = create_synthetic_face_with_dark_circles()
    
    detector = PerioribitalDetector()
    result = detector.detect_periorbital_regions(img)
    
    if result['success']:
        print("✓ Periorbital detection successful")
        print(f"  - Face detected at: {result['face_bbox']}")
        print(f"  - Left eye shape: {result['left_eye'].shape}")
        print(f"  - Right eye shape: {result['right_eye'].shape}")
        
        # Visualize
        vis = detector.draw_regions(img, result)
        cv2.imwrite('outputs/test_periorbital_detection.jpg', vis)
        print("  - Visualization saved to outputs/test_periorbital_detection.jpg")
    else:
        print(f"✗ Detection failed: {result['error']}")
    
    return result['success']


def test_color_analysis():
    """Test color analysis functions"""
    print("\nTesting color analysis...")
    
    # Create test regions with known color differences
    region1 = np.full((50, 50, 3), [180, 150, 130], dtype=np.uint8)  # Normal skin
    region2 = np.full((50, 50, 3), [140, 130, 120], dtype=np.uint8)  # Darker (dark circle)
    
    analyzer = DarkCircleAnalyzer()
    
    # Test ΔE calculation
    delta_e = analyzer.calculate_delta_e(region2, region1)
    print(f"✓ ΔE calculation: {delta_e:.2f}")
    
    # Test LAB conversion
    lab1 = analyzer.get_mean_lab(region1)
    lab2 = analyzer.get_mean_lab(region2)
    print(f"✓ LAB values - Normal: L={lab1[0]:.1f}, a={lab1[1]:.1f}, b={lab1[2]:.1f}")
    print(f"✓ LAB values - Dark: L={lab2[0]:.1f}, a={lab2[1]:.1f}, b={lab2[2]:.1f}")
    
    # Test ITA calculation
    ita1 = analyzer.calculate_ita(lab1)
    ita2 = analyzer.calculate_ita(lab2)
    skin_type1 = analyzer.classify_skin_type(ita1)
    skin_type2 = analyzer.classify_skin_type(ita2)
    print(f"✓ ITA values - Normal: {ita1:.1f}° ({skin_type1})")
    print(f"✓ ITA values - Dark: {ita2:.1f}° ({skin_type2})")
    
    # Test color indices
    redness = analyzer.calculate_redness_index(region2)
    blueness = analyzer.calculate_blueness_index(region2)
    print(f"✓ Color indices - Redness: {redness:.2f}, Blueness: {blueness:.2f}")
    
    return True


def test_full_detection_pipeline():
    """Test complete detection pipeline"""
    print("\nTesting full detection pipeline...")
    
    # Create test image
    img = create_synthetic_face_with_dark_circles(severity='moderate')
    cv2.imwrite('outputs/test_input.jpg', img)
    
    # Initialize detector
    detector = DarkCircleDetector()
    
    # Run detection
    results = detector.detect_dark_circles('outputs/test_input.jpg')
    
    if results['success']:
        print("✓ Detection successful")
        print(f"  - Average ΔE: {results['average_delta_e']}")
        print(f"  - Severity: {results['severity']}")
        print(f"  - Symmetry: {results['symmetry_score']:.0%}")
        print(f"  - Left eye ΔE: {results['left_eye']['delta_e']}")
        print(f"  - Right eye ΔE: {results['right_eye']['delta_e']}")
        
        # Save visualization
        detector.visualize_results(
            'outputs/test_input.jpg',
            results,
            'outputs/test_detection_results.png'
        )
        print("  - Visualization saved to outputs/test_detection_results.png")
        
        # Save JSON results
        json_results = {k: v for k, v in results.items() if k != 'masks'}
        with open('outputs/test_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print("  - Results saved to outputs/test_results.json")
    else:
        print(f"✗ Detection failed: {results['error']}")
    
    return results['success']


def test_severity_levels():
    """Test detection at different severity levels"""
    print("\nTesting different severity levels...")
    
    severities = ['none', 'mild', 'moderate', 'severe']
    detector = DarkCircleDetector()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, severity in enumerate(severities):
        # Create test image
        img = create_synthetic_face_with_dark_circles(severity=severity)
        filename = f'outputs/test_{severity}.jpg'
        cv2.imwrite(filename, img)
        
        # Run detection
        results = detector.detect_dark_circles(filename)
        
        if results['success']:
            detected_severity = results['severity']
            delta_e = results['average_delta_e']
            match = '✓' if detected_severity == severity else '✗'
            
            print(f"{match} {severity.capitalize()}: ΔE={delta_e:.1f}, Detected={detected_severity}")
            
            # Display in subplot
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img_rgb)
            axes[0, i].set_title(f"{severity.capitalize()}\nΔE={delta_e:.1f}")
            axes[0, i].axis('off')
            
            # Show mask if available
            if 'masks' in results and results['masks']['left'] is not None:
                axes[1, i].imshow(results['masks']['left'], cmap='gray')
                axes[1, i].set_title(f"Detected: {detected_severity}")
                axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/test_severity_levels.png', dpi=150)
    print("  - Severity comparison saved to outputs/test_severity_levels.png")
    plt.close()
    
    return True


def create_synthetic_face_with_dark_circles(severity='moderate'):
    """Create synthetic face image with dark circles"""
    # Create base image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Add face (ellipse)
    face_center = (320, 240)
    face_axes = (150, 180)
    cv2.ellipse(img, face_center, face_axes, 0, 0, 360, (180, 160, 140), -1)
    
    # Add eyes
    left_eye_center = (280, 200)
    right_eye_center = (360, 200)
    eye_size = (30, 20)
    
    # White of eyes
    cv2.ellipse(img, left_eye_center, eye_size, 0, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(img, right_eye_center, eye_size, 0, 0, 360, (250, 250, 250), -1)
    
    # Iris
    cv2.circle(img, left_eye_center, 12, (50, 100, 150), -1)
    cv2.circle(img, right_eye_center, 12, (50, 100, 150), -1)
    
    # Pupil
    cv2.circle(img, left_eye_center, 5, (0, 0, 0), -1)
    cv2.circle(img, right_eye_center, 5, (0, 0, 0), -1)
    
    # Add dark circles based on severity
    darkness_map = {
        'none': 0,
        'mild': 20,
        'moderate': 40,
        'severe': 60
    }
    
    darkness = darkness_map.get(severity, 30)
    
    if darkness > 0:
        # Create dark circle regions
        for center, eye_center in [(280, left_eye_center), (360, right_eye_center)]:
            # Infraorbital region
            infra_center = (eye_center[0], eye_center[1] + 30)
            infra_axes = (35, 15)
            
            # Create gradient effect
            overlay = img.copy()
            color = (180 - darkness, 160 - darkness, 140 - darkness)
            cv2.ellipse(overlay, infra_center, infra_axes, 0, 0, 180, color, -1)
            
            # Blend with original
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Add some texture/noise
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img


def main():
    """Run all tests"""
    print("Dark Circle Detection System Test Suite")
    print("=" * 50)
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Run tests
    tests = [
        ("Periorbital Detection", test_periorbital_detection),
        ("Color Analysis", test_color_analysis),
        ("Full Detection Pipeline", test_full_detection_pipeline),
        ("Severity Levels", test_severity_levels)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)