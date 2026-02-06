"""
ReActor V5 - Realism Enhancement Module

Implements advanced realism improvements including:
- Identity/texture separation
- Adaptive face restoration
- Controlled noise injection
- Frequency-aware blending
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List
try:
    from scipy import ndimage
    from skimage import measure, filters
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[ReActor V5] Warning: scipy/skimage not available, using basic processing")
import random


class RealismEnhancer:
    """
    Advanced realism enhancement for face swapping.
    
    Key features:
    - Separates identity (geometry) from texture (skin details)
    - Applies adaptive restoration only when blur is detected
    - Injects controlled noise to prevent plastic appearance
    - Uses frequency-aware blending for natural integration
    """
    
    def __init__(self):
        # Blur detection parameters
        self.blur_threshold = 100.0  # Laplacian variance threshold
        self.max_restoration_strength = 0.35  # Cap restoration strength
        
        # Skin texture parameters - adds realistic micro-texture to combat plasticky look
        self.skin_texture_strength = 0.08  # Subtle texture overlay
        self.skin_noise_scale = 0.015  # Fine grain noise
        self.pore_noise_scale = 0.025  # Larger pore-like texture
        
        # Frequency separation parameters
        self.low_freq_sigma = 3.0  # For identity/geometry
        self.high_freq_preserve = 0.8  # How much texture to preserve
        
        print("[ReActor V5] Realism enhancer initialized")
    
    def detect_blur(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, bool]:
        """
        Detect blur in image using Laplacian variance.
        
        Args:
            image: Input image (BGR format)
            mask: Optional mask to focus blur detection on specific region
            
        Returns:
            Tuple[blur_score, is_blurry]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask if provided
        if mask is not None:
            gray = gray * (mask / 255.0)
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_blurry = laplacian_var < self.blur_threshold
        
        print(f"[ReActor V5] Blur detection - Score: {laplacian_var:.1f}, Threshold: {self.blur_threshold}, Blurry: {is_blurry}")
        
        return laplacian_var, is_blurry
    
    def separate_identity_texture(self, image: np.ndarray, sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate image into low-frequency (identity/geometry) and high-frequency (texture) components.
        
        Args:
            image: Input image (BGR format)
            sigma: Gaussian blur sigma for separation
            
        Returns:
            Tuple[low_freq_identity, high_freq_texture]
        """
        # Convert to float
        image_float = image.astype(np.float32) / 255.0
        
        # Low frequency: identity/geometry (blurred)
        low_freq = cv2.GaussianBlur(image_float, (0, 0), sigma)
        
        # High frequency: texture details (residual)
        high_freq = image_float - low_freq
        
        # Convert back to uint8
        low_freq = np.clip(low_freq * 255, 0, 255).astype(np.uint8)
        high_freq = np.clip((high_freq + 0.5) * 255, 0, 255).astype(np.uint8)
        
        return low_freq, high_freq
    
    def add_skin_texture(self, 
                        image: np.ndarray, 
                        face_mask: np.ndarray,
                        strength: float = 0.08) -> np.ndarray:
        """
        Add realistic skin micro-texture to combat plasticky GPEN output.
        
        Uses multi-scale noise to simulate:
        - Fine grain (pores)
        - Medium texture (skin surface)
        - Color variation (natural skin tone inconsistency)
        
        Args:
            image: Input image (BGR format)
            face_mask: Face region mask
            strength: Texture strength (0.05-0.15 recommended)
            
        Returns:
            Image with realistic skin texture
        """
        h, w = image.shape[:2]
        
        # Resize mask if needed
        if face_mask.shape[:2] != (h, w):
            face_mask = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize mask
        if len(face_mask.shape) == 2:
            mask_norm = face_mask.astype(np.float32) / 255.0
        else:
            mask_norm = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Create skin mask (only skin tones, not eyes/lips)
        skin_mask = self.create_skin_mask(image, (face_mask * 255).astype(np.uint8) if face_mask.max() <= 1 else face_mask)
        skin_mask_norm = skin_mask.astype(np.float32) / 255.0
        
        # Resolution-aware scaling
        resolution_factor = np.sqrt((w * h) / (512 * 512))
        
        # === Layer 1: Fine grain noise (pores) ===
        fine_noise = np.random.normal(0, self.skin_noise_scale / resolution_factor, (h, w, 3)).astype(np.float32)
        
        # === Layer 2: Medium texture (skin surface variation) ===
        # Create larger-scale perlin-like noise by upscaling smaller noise
        small_h, small_w = h // 8, w // 8
        medium_noise_small = np.random.normal(0, self.pore_noise_scale, (small_h, small_w, 3)).astype(np.float32)
        medium_noise = cv2.resize(medium_noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # === Layer 3: Color micro-variation (natural skin tone inconsistency) ===
        # Slightly vary hue/saturation in different areas
        color_var_small = np.random.normal(0, 0.01, (h // 16, w // 16, 3)).astype(np.float32)
        color_variation = cv2.resize(color_var_small, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Combine texture layers
        combined_texture = fine_noise * 0.5 + medium_noise * 0.35 + color_variation * 0.15
        
        # Apply skin mask (only add texture to skin, not eyes/lips/hair)
        combined_texture = combined_texture * skin_mask_norm[:, :, np.newaxis]
        
        # Apply to image
        image_float = image.astype(np.float32) / 255.0
        textured_image = image_float + combined_texture * strength
        
        # Clip and convert
        result = np.clip(textured_image * 255, 0, 255).astype(np.uint8)
        
        print(f"[ReActor V5] Added skin texture - Strength: {strength:.2f}, Resolution factor: {resolution_factor:.2f}")
        
        return result
    
    def create_skin_mask(self, image: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """
        Create skin mask using color-based segmentation within face region.
        
        Args:
            image: Input image (BGR format)
            face_mask: Face region mask
            
        Returns:
            Skin mask (more precise than face mask)
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color ranges in HSV
        # These ranges work well for various skin tones
        lower_skin1 = np.array([0, 20, 70])
        upper_skin1 = np.array([20, 255, 255])
        
        lower_skin2 = np.array([170, 20, 70])
        upper_skin2 = np.array([180, 255, 255])
        
        # Create skin masks
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Intersect with face mask
        if len(face_mask.shape) == 3:
            face_mask = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY)
        
        skin_mask = cv2.bitwise_and(skin_mask, face_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Gaussian blur for smooth edges
        skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 3)
        
        return skin_mask
    
    def frequency_aware_blending(self, 
                                swapped_image: np.ndarray,
                                original_texture: np.ndarray,
                                face_mask: np.ndarray,
                                texture_preservation: float = 0.8) -> np.ndarray:
        """
        Blend swapped face back with frequency separation.
        
        Args:
            swapped_image: Face-swapped image
            original_texture: Original high-frequency texture
            face_mask: Face region mask
            texture_preservation: How much original texture to preserve (0-1)
            
        Returns:
            Blended image with preserved texture details
        """
        # Get target dimensions from swapped image
        target_h, target_w = swapped_image.shape[:2]
        
        # Resize original_texture to match swapped_image dimensions if needed
        if original_texture.shape[:2] != (target_h, target_w):
            print(f"[ReActor V5] Resizing texture from {original_texture.shape[:2]} to {(target_h, target_w)}")
            original_texture = cv2.resize(original_texture, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Resize face_mask to match if needed
        if face_mask.shape[:2] != (target_h, target_w):
            face_mask = cv2.resize(face_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Separate frequencies in swapped image
        swapped_low, swapped_high = self.separate_identity_texture(swapped_image)
        
        # Create blending weights
        if len(face_mask.shape) == 2:
            blend_mask = np.stack([face_mask] * 3, axis=2) / 255.0
        else:
            blend_mask = face_mask / 255.0
        
        # Ensure all arrays are float32 for computation
        original_texture = original_texture.astype(np.float32)
        swapped_high = swapped_high.astype(np.float32)
        
        # Blend high frequencies (texture)
        texture_blend = (
            original_texture * texture_preservation * blend_mask +
            swapped_high * (1 - texture_preservation * blend_mask)
        )
        
        # Combine with low frequencies (identity from swapped)
        result_low = swapped_low.astype(np.float32) / 255.0
        result_high = texture_blend.astype(np.float32) / 255.0 - 0.5
        
        # Recombine frequencies
        final_result = result_low + result_high
        
        # Clip and convert
        return np.clip(final_result * 255, 0, 255).astype(np.uint8)
    
    def adaptive_face_restoration(self, 
                                 image: np.ndarray,
                                 face_mask: np.ndarray,
                                 restorer: Any = None) -> Tuple[np.ndarray, bool]:
        """
        Check if face needs restoration (but don't apply here).
        
        The actual GPEN restoration is now handled in the main swapper
        with proper face region detection for better quality.
        
        Args:
            image: Input image
            face_mask: Face region mask
            restorer: Face restoration model (not used here anymore)
            
        Returns:
            Tuple[image, needs_restoration_flag]
        """
        # Detect blur in face region
        blur_score, is_blurry = self.detect_blur(image, face_mask)
        
        # Just return the blur detection result
        # Actual restoration happens in main swapper with proper face detection
        print(f"[ReActor V5] Blur check - Score: {blur_score:.1f}, Needs restoration: {is_blurry}")
        
        return image, is_blurry
    
    def enhance_realism(self, 
                       original_target: np.ndarray,
                       swapped_image: np.ndarray,
                       face_mask: np.ndarray,
                       config: Dict[str, Any],
                       restorer: Any = None) -> np.ndarray:
        """
        Apply comprehensive realism enhancement pipeline.
        
        Pipeline:
        1. Create skin mask
        2. Add realistic skin texture (combats plasticky GPEN output)
        
        Args:
            original_target: Original TARGET image (for texture preservation)
            swapped_image: Face-swapped result
            face_mask: Face region mask
            config: Enhancement configuration
            restorer: Optional face restoration model
            
        Returns:
            Enhanced realistic result
        """
        print("[ReActor V5] Starting realism enhancement pipeline...")
        
        # Get target dimensions from swapped image
        target_h, target_w = swapped_image.shape[:2]
        
        # Resize face_mask to match if needed
        face_mask_resized = face_mask
        if face_mask.shape[:2] != (target_h, target_w):
            face_mask_resized = cv2.resize(face_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        enhanced_image = swapped_image.copy()
        
        # Add skin texture to combat plasticky look from GPEN
        texture_strength = config.get('texture_strength', self.skin_texture_strength)
        if texture_strength > 0:
            print(f"[ReActor V5] Adding skin texture (strength={texture_strength:.2f})...")
            enhanced_image = self.add_skin_texture(enhanced_image, face_mask_resized, texture_strength)
        
        print("[ReActor V5] Realism enhancement complete")
        
        return enhanced_image


# Global realism enhancer
realism_enhancer = None

def get_realism_enhancer() -> RealismEnhancer:
    """Get global realism enhancer instance"""
    global realism_enhancer
    if realism_enhancer is None:
        realism_enhancer = RealismEnhancer()
    return realism_enhancer


def enhance_face_realism(original_target: np.ndarray,
                        swapped_image: np.ndarray,  
                        face_mask: np.ndarray,
                        config: Dict[str, Any],
                        restorer: Any = None) -> np.ndarray:
    """
    Convenience function for face realism enhancement.
    
    Args:
        original_target: Original TARGET image (for texture preservation)
        swapped_image: Face-swapped result
        face_mask: Face region mask  
        config: Enhancement configuration
        restorer: Optional face restoration model
        
    Returns:
        Enhanced realistic result
    """
    enhancer = get_realism_enhancer()
    return enhancer.enhance_realism(
        original_target, swapped_image, face_mask, config, restorer
    )