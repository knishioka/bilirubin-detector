#!/usr/bin/env python3
"""
Create sample eye images with different levels of jaundice for testing
"""

import cv2
import numpy as np
import os

def create_eye_image_with_jaundice(jaundice_level='none'):
    """
    Create synthetic eye image with specified jaundice level
    
    Args:
        jaundice_level: 'none', 'mild', 'moderate', 'severe'
    """
    # Jaundice color settings (BGR format)
    jaundice_colors = {
        'none': [240, 245, 250],      # Normal white sclera
        'mild': [200, 230, 250],       # Slight yellow tint
        'moderate': [150, 200, 240],   # Moderate yellow
        'severe': [100, 180, 230]      # Strong yellow
    }
    
    sclera_color = jaundice_colors.get(jaundice_level, jaundice_colors['none'])
    
    height, width = 480, 640
    # Face background
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array([150, 180, 210], dtype=np.uint8)
    
    # Add noise
    noise = np.random.normal(0, 5, (height, width, 3))
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Face center
    face_center = (width // 2, height // 2)
    
    # Eye positions
    eye_y = face_center[1] - 50
    left_eye_center = (face_center[0] - 80, eye_y)
    right_eye_center = (face_center[0] + 80, eye_y)
    
    # Draw eyes
    for eye_center in [left_eye_center, right_eye_center]:
        # Sclera (white part)
        axes = (50, 25)
        cv2.ellipse(image, eye_center, axes, 0, 0, 360, sclera_color, -1)
        
        # Add gradient for realism
        overlay = image.copy()
        gradient_color = [min(255, c + 20) for c in sclera_color]
        cv2.ellipse(overlay, eye_center, (axes[0]-5, axes[1]-5), 0, 0, 360, gradient_color, -1)
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
        
        # Iris
        iris_radius = 20
        iris_color = [80, 50, 30]
        cv2.circle(image, eye_center, iris_radius, iris_color, -1)
        
        # Iris texture
        for r in range(5, iris_radius, 3):
            color_var = [c + np.random.randint(-20, 20) for c in iris_color]
            cv2.circle(image, eye_center, r, color_var, 1)
        
        # Pupil
        cv2.circle(image, eye_center, 8, (0, 0, 0), -1)
        
        # Highlight
        highlight_pos = (eye_center[0] - 5, eye_center[1] - 5)
        cv2.circle(image, highlight_pos, 3, (255, 255, 255), -1)
        
        # Blood vessels (more visible in jaundice)
        if jaundice_level in ['moderate', 'severe']:
            vessel_count = 5 if jaundice_level == 'severe' else 3
            for _ in range(vessel_count):
                start_x = eye_center[0] + np.random.randint(-40, -20)
                start_y = eye_center[1] + np.random.randint(-10, 10)
                end_x = eye_center[0] + np.random.randint(-10, 10)
                end_y = eye_center[1] + np.random.randint(-5, 5)
                # More prominent vessels in jaundice
                vessel_color = [150, 170, 200] if jaundice_level == 'severe' else [170, 185, 210]
                cv2.line(image, (start_x, start_y), (end_x, end_y), vessel_color, 1)
    
    # Eyebrows
    eyebrow_y = eye_y - 30
    cv2.ellipse(image, (left_eye_center[0], eyebrow_y), (40, 8), 0, 0, 180, (50, 50, 50), -1)
    cv2.ellipse(image, (right_eye_center[0], eyebrow_y), (40, 8), 0, 0, 180, (50, 50, 50), -1)
    
    # Nose
    nose_tip = (face_center[0], face_center[1] + 20)
    cv2.circle(image, nose_tip, 15, [140, 170, 200], -1)
    
    # Apply blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image

def main():
    """Create sample images with different jaundice levels"""
    output_dir = 'sample_images'
    os.makedirs(output_dir, exist_ok=True)
    
    jaundice_levels = ['none', 'mild', 'moderate', 'severe']
    
    for level in jaundice_levels:
        print(f"Creating {level} jaundice sample...")
        image = create_eye_image_with_jaundice(level)
        filename = f'{output_dir}/eye_{level}_jaundice.jpg'
        cv2.imwrite(filename, image)
        print(f"  Saved: {filename}")
    
    print("\nSample images created successfully!")

if __name__ == "__main__":
    main()