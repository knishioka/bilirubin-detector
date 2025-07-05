#!/usr/bin/env python3
"""
Evaluate bilirubin detection on sample images
"""

import os
from bilirubin_detector import BilirubinDetector
import json

def evaluate_all_samples():
    """Evaluate all sample images and display results"""
    detector = BilirubinDetector()
    sample_dir = 'sample_images'
    
    results = []
    
    print("Evaluating bilirubin detection on sample images...")
    print("=" * 60)
    
    # Expected values for each level (approximate)
    expected_values = {
        'none': 1.0,      # Normal
        'mild': 5.0,      # Mild jaundice
        'moderate': 15.0, # Moderate jaundice  
        'severe': 25.0    # Severe jaundice
    }
    
    for filename in sorted(os.listdir(sample_dir)):
        if filename.endswith('.jpg'):
            filepath = os.path.join(sample_dir, filename)
            
            # Extract jaundice level from filename
            level = filename.split('_')[1]
            
            print(f"\nImage: {filename}")
            print("-" * 40)
            
            # Run detection
            result = detector.detect_bilirubin(filepath)
            
            if result['success']:
                detected_value = result['bilirubin_level_mg_dl']
                expected = expected_values.get(level, 0)
                difference = abs(detected_value - expected)
                
                print(f"  Expected Level: {level.upper()}")
                print(f"  Expected Value: ~{expected} mg/dL")
                print(f"  Detected Value: {detected_value} mg/dL")
                print(f"  Risk Level: {result['risk_level']}")
                print(f"  Confidence: {result['confidence']:.2%}")
                print(f"  Difference: {difference:.1f} mg/dL")
                
                # Color features
                features = result['color_features']
                print(f"\n  Color Features:")
                print(f"    HSV Yellow Ratio: {features['hsv_yellow_ratio']:.3f}")
                print(f"    RGB R/B Ratio: {features['rgb_red_blue_ratio']:.3f}")
                print(f"    LAB Yellowness: {features['lab_yellowness']:.1f}")
                print(f"    Yellowness Index: {features['yellowness_index']:.3f}")
                
                # Save visualization
                output_path = filepath.replace('.jpg', '_result.png')
                detector.visualize_results(filepath, result, output_path)
                print(f"\n  Visualization saved: {output_path}")
                
                results.append({
                    'image': filename,
                    'level': level,
                    'expected': expected,
                    'detected': detected_value,
                    'risk_level': result['risk_level'],
                    'confidence': result['confidence'],
                    'error': difference
                })
            else:
                print(f"  Detection failed: {result['error']}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        avg_error = sum(r['error'] for r in results) / len(results)
        print(f"Average Error: {avg_error:.2f} mg/dL")
        
        print("\nDetailed Results:")
        print(f"{'Level':<10} {'Expected':<10} {'Detected':<10} {'Risk':<10} {'Confidence':<10}")
        print("-" * 50)
        for r in results:
            print(f"{r['level']:<10} {r['expected']:<10.1f} {r['detected']:<10.1f} {r['risk_level']:<10} {r['confidence']:<10.2%}")
    
    # Save results to JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: evaluation_results.json")

if __name__ == "__main__":
    evaluate_all_samples()