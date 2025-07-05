#!/usr/bin/env python3
"""
Generate sample images with varying dark circle severities
Creates realistic synthetic images for testing the detection algorithm
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict


class DarkCircleSampleGenerator:
    """Generates synthetic face images with dark circles"""
    
    def __init__(self, output_dir: str = 'samples/dark_circles'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Skin tone variations (BGR format)
        self.skin_tones = {
            'light': (210, 190, 170),
            'medium': (180, 160, 140),
            'tan': (160, 140, 120),
            'dark': (120, 100, 80)
        }
        
        # Dark circle types
        self.dark_circle_types = {
            'pigmentation': {'color_shift': -30, 'redness': 0, 'blueness': 0},
            'vascular': {'color_shift': -20, 'redness': 10, 'blueness': 5},
            'structural': {'color_shift': -25, 'redness': 5, 'blueness': 10},
            'mixed': {'color_shift': -25, 'redness': 8, 'blueness': 8}
        }
    
    def generate_face_base(self, skin_tone: Tuple[int, int, int],
                          img_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Generate base face image"""
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 240
        
        # Face parameters
        face_center = (img_size[0] // 2, img_size[1] // 2)
        face_width = int(img_size[0] * 0.4)
        face_height = int(img_size[1] * 0.5)
        
        # Draw face ellipse
        cv2.ellipse(img, face_center, (face_width // 2, face_height // 2),
                   0, 0, 360, skin_tone, -1)
        
        # Add facial features
        # Eyes
        eye_y = face_center[1] - face_height // 6
        left_eye_x = face_center[0] - face_width // 4
        right_eye_x = face_center[0] + face_width // 4
        
        # Eye whites
        eye_width = face_width // 8
        eye_height = face_height // 12
        
        for eye_x in [left_eye_x, right_eye_x]:
            # White of eye
            cv2.ellipse(img, (eye_x, eye_y), (eye_width, eye_height),
                       0, 0, 360, (250, 250, 250), -1)
            
            # Add slight shadow to eye socket
            socket_overlay = img.copy()
            cv2.ellipse(socket_overlay, (eye_x, eye_y - eye_height),
                       (eye_width + 10, eye_height + 5),
                       0, 0, 360, 
                       (skin_tone[0] - 20, skin_tone[1] - 20, skin_tone[2] - 20), -1)
            cv2.addWeighted(socket_overlay, 0.3, img, 0.7, 0, img)
            
            # Iris
            cv2.circle(img, (eye_x, eye_y), eye_height // 2, (60, 90, 120), -1)
            
            # Pupil
            cv2.circle(img, (eye_x, eye_y), eye_height // 4, (0, 0, 0), -1)
            
            # Eye highlight
            cv2.circle(img, (eye_x - 2, eye_y - 2), 2, (255, 255, 255), -1)
        
        # Nose (simple)
        nose_tip = (face_center[0], face_center[1] + face_height // 12)
        nose_width = face_width // 10
        
        # Nose shadow
        nose_pts = np.array([
            [nose_tip[0] - nose_width, nose_tip[1] - 10],
            [nose_tip[0], nose_tip[1]],
            [nose_tip[0] + nose_width, nose_tip[1] - 10]
        ], np.int32)
        
        overlay = img.copy()
        cv2.fillPoly(overlay, [nose_pts], 
                    (skin_tone[0] - 15, skin_tone[1] - 15, skin_tone[2] - 15))
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        # Mouth
        mouth_y = face_center[1] + face_height // 4
        mouth_width = face_width // 4
        cv2.ellipse(img, (face_center[0], mouth_y), (mouth_width, 8),
                   0, 0, 180, (150, 100, 100), -1)
        
        # Add some skin texture
        texture = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        
        # Store eye positions for dark circle placement
        self.eye_positions = {
            'left': (left_eye_x, eye_y),
            'right': (right_eye_x, eye_y),
            'eye_width': eye_width,
            'eye_height': eye_height
        }
        
        return img
    
    def add_dark_circles(self, img: np.ndarray, severity: str,
                        dc_type: str = 'mixed') -> np.ndarray:
        """Add dark circles to face image"""
        severity_params = {
            'none': {'intensity': 0, 'spread': 0, 'blur': 0},
            'mild': {'intensity': 0.3, 'spread': 1.2, 'blur': 5},
            'moderate': {'intensity': 0.5, 'spread': 1.5, 'blur': 7},
            'severe': {'intensity': 0.7, 'spread': 1.8, 'blur': 9}
        }
        
        if severity not in severity_params:
            return img
        
        params = severity_params[severity]
        dc_params = self.dark_circle_types[dc_type]
        
        if params['intensity'] == 0:
            return img
        
        result = img.copy()
        
        for side in ['left', 'right']:
            eye_x, eye_y = self.eye_positions[side]
            eye_w = self.eye_positions['eye_width']
            eye_h = self.eye_positions['eye_height']
            
            # Define infraorbital region
            infra_y = eye_y + int(eye_h * 0.8)
            infra_w = int(eye_w * params['spread'])
            infra_h = int(eye_h * 0.8 * params['spread'])
            
            # Create mask for dark circle
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            # Draw semi-ellipse for infraorbital area
            cv2.ellipse(mask, (eye_x, infra_y), (infra_w, infra_h),
                       0, 0, 180, 255, -1)
            
            # Add tear trough effect
            if params['spread'] > 1.3:
                trough_pts = np.array([
                    [eye_x - infra_w, infra_y],
                    [eye_x - infra_w // 2, infra_y + infra_h // 2],
                    [eye_x, infra_y + infra_h],
                    [eye_x + infra_w // 2, infra_y + infra_h // 2],
                    [eye_x + infra_w, infra_y]
                ], np.int32)
                cv2.fillPoly(mask, [trough_pts], 255)
            
            # Blur mask for smooth transition
            mask = cv2.GaussianBlur(mask, (params['blur'] * 2 + 1, params['blur'] * 2 + 1), 0)
            
            # Create color overlay
            overlay = result.copy()
            
            # Apply color changes based on dark circle type
            b, g, r = cv2.split(overlay)
            
            # Darken
            color_shift = dc_params['color_shift'] * params['intensity']
            b = np.clip(b + color_shift, 0, 255).astype(np.uint8)
            g = np.clip(g + color_shift, 0, 255).astype(np.uint8)
            r = np.clip(r + color_shift, 0, 255).astype(np.uint8)
            
            # Add redness (vascular component)
            if dc_params['redness'] > 0:
                r = np.clip(r + dc_params['redness'] * params['intensity'], 0, 255).astype(np.uint8)
            
            # Add blueness (venous pooling)
            if dc_params['blueness'] > 0:
                b = np.clip(b + dc_params['blueness'] * params['intensity'], 0, 255).astype(np.uint8)
            
            overlay = cv2.merge([b, g, r])
            
            # Apply masked overlay
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (result * (1 - mask_3ch) + overlay * mask_3ch).astype(np.uint8)
        
        return result
    
    def generate_sample_set(self):
        """Generate complete set of sample images"""
        metadata = []
        sample_id = 0
        
        for skin_name, skin_tone in self.skin_tones.items():
            for dc_type in self.dark_circle_types.keys():
                for severity in ['none', 'mild', 'moderate', 'severe']:
                    # Generate base face
                    img = self.generate_face_base(skin_tone)
                    
                    # Add dark circles
                    img = self.add_dark_circles(img, severity, dc_type)
                    
                    # Add some variation
                    if np.random.random() > 0.5:
                        # Add slight rotation
                        angle = np.random.uniform(-5, 5)
                        center = (img.shape[1] // 2, img.shape[0] // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                           borderValue=(240, 240, 240))
                    
                    # Save image
                    filename = f"sample_{sample_id:03d}_{skin_name}_{dc_type}_{severity}.jpg"
                    filepath = self.output_dir / filename
                    cv2.imwrite(str(filepath), img)
                    
                    # Store metadata
                    metadata.append({
                        'id': sample_id,
                        'filename': filename,
                        'skin_tone': skin_name,
                        'dark_circle_type': dc_type,
                        'severity': severity,
                        'expected_delta_e': self._estimate_delta_e(severity)
                    })
                    
                    sample_id += 1
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Generated {sample_id} sample images in {self.output_dir}")
        return metadata
    
    def _estimate_delta_e(self, severity: str) -> float:
        """Estimate expected ΔE value for severity level"""
        estimates = {
            'none': 1.5,
            'mild': 4.0,
            'moderate': 6.5,
            'severe': 10.0
        }
        return estimates.get(severity, 5.0)
    
    def generate_comparison_grid(self):
        """Generate comparison grid showing all severity levels"""
        fig_width = len(self.dark_circle_types) * 3
        fig_height = len(self.skin_tones) * 3
        
        # Create grid image
        cell_size = 150
        grid_img = np.ones((len(self.skin_tones) * cell_size * 4,
                           len(self.dark_circle_types) * cell_size,
                           3), dtype=np.uint8) * 255
        
        for i, (skin_name, skin_tone) in enumerate(self.skin_tones.items()):
            for j, dc_type in enumerate(self.dark_circle_types.keys()):
                for k, severity in enumerate(['none', 'mild', 'moderate', 'severe']):
                    # Generate face
                    img = self.generate_face_base(skin_tone, (cell_size, cell_size))
                    img = self.add_dark_circles(img, severity, dc_type)
                    
                    # Place in grid
                    y_start = (i * 4 + k) * cell_size
                    x_start = j * cell_size
                    grid_img[y_start:y_start + cell_size,
                            x_start:x_start + cell_size] = img
                    
                    # Add labels
                    if k == 0:  # Top of each group
                        cv2.putText(grid_img, f"{skin_name}", 
                                  (x_start + 5, y_start + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    cv2.putText(grid_img, severity,
                              (x_start + 5, y_start + cell_size - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                    
                    if i == 0 and k == 0:  # Top row headers
                        cv2.putText(grid_img, dc_type,
                                  (x_start + 5, 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Save grid
        cv2.imwrite(str(self.output_dir / 'comparison_grid.jpg'), grid_img)
        print(f"Saved comparison grid to {self.output_dir}/comparison_grid.jpg")


def main():
    """Generate sample images"""
    generator = DarkCircleSampleGenerator()
    
    print("Generating dark circle sample images...")
    metadata = generator.generate_sample_set()
    
    print("\nGenerating comparison grid...")
    generator.generate_comparison_grid()
    
    # Summary statistics
    print("\nGeneration Summary:")
    print(f"Total samples: {len(metadata)}")
    print(f"Skin tones: {len(generator.skin_tones)}")
    print(f"Dark circle types: {len(generator.dark_circle_types)}")
    print(f"Severity levels: 4")
    print(f"\nSamples saved to: {generator.output_dir}")


if __name__ == "__main__":
    main()